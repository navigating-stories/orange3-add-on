import os
import pathlib
import re
import sys

from Orange.data import Table
from Orange.widgets.settings import Setting, DomainContextHandler, ContextSetting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import Input, Output, OWWidget
from orangecontrib.text.corpus import Corpus
from AnyQt.QtWidgets import QSizePolicy, QSplitter
from AnyQt.QtCore import Qt
from Orange.widgets import gui
from Orange.data.pandas_compat import table_from_frame

import storynavigation.modules.constants as constants
from storynavigation.modules.purposeanalysis import PurposeAnalyzer
import storynavigation.modules.util as util

from AnyQt.QtWidgets import QAbstractItemView, QHeaderView, QTableView
from storynavigation.widgets.OWSNActorAnalysis import DocumentTableView, DocumentListModel, DocumentsFilterProxyModel, _count_matches
from typing import Set
from orangecontrib.text.widgets.utils import widgets
from Orange.data.io import FileFormat


class OWSNPurposeAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = 'Purpose Analysis'
    description = "Provides tools to help identify the main purpose of a story."
    icon = "icons/purpose_analysis_icon.png"
    priority = 6480
    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    n_segments = 1
    entity_colors = {"CONTEXT": "salmon",
                     "PURPOSE": "lightgreen",
                     "ADVERB": "lightblue"}
    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    class Inputs:
        story_elements = Input("Story elements", Table)


    class Outputs:
        dataset_level_data = Output('Intermediate purpose', Table)


    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories_selected = []
        self.story_elements = []
        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)
        self.verb_frames = {}

        self.__initialize_strategy()
        self.read_strategy_file(self.strategy_file_name)

        self.__make_language_selection_menu()
        self.__make_strategy_selection_menu()
        self.__make_strategy_file_dialog()
        self.__make_regexp_filter_dialog()
        self.__make_document_viewer()


    def __initialize_strategy(self):
        self.adverbs_strategy_file_name = os.path.join(
            str(constants.PKG),
            str(constants.RESOURCES_SUBPACKAGE),
            ("dutch" if self.language == "nl" else "english") + "_purpose_adverbs.csv")
        self.verbs_strategy_file_name = os.path.join(
            str(constants.PKG),
            str(constants.RESOURCES_SUBPACKAGE),
            ("dutch" if self.language == "nl" else "english") + "_purpose_verbs.csv")
        self.recent_strategy_files = [self.verbs_strategy_file_name,
                                      self.adverbs_strategy_file_name]
        self.strategy_file_name = self.recent_strategy_files[0]
        self.purpose_strategy = constants.PURPOSE_STRATEGY_VERBS


    def __make_language_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language:",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            currentIndex=1,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)
        self.select_language_combo.setEnabled(True)


    def __make_strategy_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Strategy:",
            value="purpose_strategy",
            items=[constants.PURPOSE_STRATEGY_VERBS, constants.PURPOSE_STRATEGY_ADVERBS],
            sendSelectedValue=True,
            currentIndex=0,
            callback=self.__process_purpose_strategy_change,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)
        self.select_language_combo.setEnabled(True)


    def __make_strategy_file_dialog(self):
        # code copied from Corpus widget
        fbox = gui.widgetBox(self.controlArea, "Verb frames file:", orientation=0)
        self.file_widget = widgets.FileWidget(
            recent_files=self.recent_strategy_files, icon_size=(16, 16),
            on_open=self.read_strategy_file, dialog_format=self.dlgFormats,
            dialog_title='Choose strategy file',
            reload_label='Reload', browse_label='Browse',
            allow_empty=False, minimal_width=150,
        )
        fbox.layout().addWidget(self.file_widget)


    def __make_regexp_filter_dialog(self):
        # code copied from CorpusViewer widget
        self.regexp_filter = ""
        self.filter_input = gui.lineEdit(
            self.mainArea,
            self,
            "regexp_filter",
            orientation=Qt.Horizontal,
            sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed),
            label="RegExp Filter:",
            callback=self.refresh_search,
        )


    def __make_document_viewer(self):
        # code copied from CorpusViewer widget
        self.splitter = QSplitter(orientation=Qt.Horizontal, childrenCollapsible=False)
        self.doc_list = DocumentTableView()
        self.doc_list.setSelectionBehavior(QTableView.SelectRows)
        self.doc_list.setSelectionMode(QTableView.ExtendedSelection)
        self.doc_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.doc_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.doc_list.horizontalHeader().setVisible(False)
        self.splitter.addWidget(self.doc_list)

        self.doc_list_model = DocumentListModel()
        proxy_model = DocumentsFilterProxyModel()
        proxy_model.setSourceModel(self.doc_list_model)
        self.doc_list.setModel(proxy_model)
        self.doc_list.selectionModel().selectionChanged.connect(self.selection_changed)
        self.doc_webview = gui.WebviewWidget(self.splitter, debug=False)
        self.doc_webview.setHtml("<div style=\"max-width:600px\" />")
        self.mainArea.layout().addWidget(self.splitter)
        total_size = self.splitter.size().width()
        self.splitter.setSizes([int(0.2 * total_size), int(0.8 * total_size)])


    def __update_stories_selected(self):
        self.stories_selected = []
        regexp = re.compile(self.regexp_filter)
        for i in range(0, len(self.text_tuples)):
            if regexp.search(self.text_tuples[i][0]):
                self.stories_selected.append(i)


    def refresh_search(self):
        if self.text_tuples is not None:
            self.doc_list.model().set_filter_string(self.regexp_filter)
            self.__update_stories_selected()
            self.__visualize_text_data()


    def read_strategy_file(self, strategy_file_name):
        self.strategy_file_name = strategy_file_name
        self.verb_frames = []
        try:
            verb_frames_lines = pathlib.Path(strategy_file_name).read_text(encoding="utf-8").strip().split("\n")
            for line in verb_frames_lines:
                self.verb_frames.append([token.strip() for token in line.strip().split(",")])
            if re.search("adverb", strategy_file_name):
                self.purpose_strategy = constants.PURPOSE_STRATEGY_ADVERBS
            else:
                self.purpose_strategy = constants.PURPOSE_STRATEGY_VERBS
        except Exception as e:
            print("read_strategy_file", str(e))
        if self.story_elements:
            self.reset_story_elements(self.story_elements)


    def __process_purpose_strategy_change(self):
        if re.search("adverb", self.purpose_strategy):
            self.strategy_file_name = self.adverbs_strategy_file_name
        else:
            self.strategy_file_name = self.verbs_strategy_file_name
        self.read_strategy_file(self.strategy_file_name)


    def get_selected_indexes(self) -> Set[int]:
        m = self.doc_list.model().mapToSource
        return [int(m(i).row()) for i in self.doc_list.selectionModel().selectedRows()]


    def selection_changed(self) -> None:
        self.stories_selected = self.get_selected_indexes()
        self.__visualize_text_data()


    @Inputs.story_elements
    def set_story_elements(self, story_elements=None):
        """Story elements expects a table. Because Corpus is a subclass of Table, Orange type checking 
        misses wrongly connected inputs."""

        if story_elements is not None:
            self.stories_selected = []
            self.story_elements = story_elements
            self.reset_story_elements(story_elements)


    def reset_story_elements(self, story_elements=None):
        if story_elements is not None:
            self.text_tuples = self.make_text_tuples(story_elements)
            self.__action_analyze_purpose_wrapper()

            self.doc_list_model.setup_data(self.make_document_names(self.text_tuples), [text for text, text_id in self.text_tuples])


    def make_document_names(self, text_tuples):
        document_names = []
        for text, text_id in text_tuples:
            document_names.append("Document " + str(int(text_id) + 1))
        return document_names


    def make_text_tuples(self, story_elements):
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        current_story = ""
        current_story_id = ""
        last_sentence = ""
        text_tuples = []
        for index, row in story_elements_df.iterrows():
            story_id = row["storyid"]
            if story_id != current_story_id:
                if current_story_id != "":
                    text_tuples.append((current_story, current_story_id))
                current_story = ""
                current_story_id = story_id
                last_sentence = ""
            if row["sentence"] != last_sentence:
                current_story += row["sentence"] if current_story == "" else " " + row["sentence"]
                last_sentence = row["sentence"]
        if current_story != "":
            text_tuples.append((current_story, current_story_id))
        return text_tuples


    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()


    def __action_analyze_purpose_wrapper(self):
        self.start(self.action_analyze_purpose)


    def action_analyze_purpose(self, state: TaskState):
        def move_progress_bar(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.analyzer = PurposeAnalyzer(
             language=self.language,
             story_elements=self.story_elements,
             verb_frames=self.verb_frames,
             purpose_strategy=self.purpose_strategy,
             callback=move_progress_bar
        )


    def on_done(self, result) -> None:
        self.refresh_search()
        try:
            self.Outputs.dataset_level_data.send(table_from_frame(self.analyzer.purpose_analysis))
        except Exception as e:
            print("on_done", e)


    def __make_entity_bar_for_html(self):
        return " ".join(['<mark style="background-color:' +
                           self.entity_colors[entity] +
                           f'">{entity}</mark>' for entity in self.entity_colors ])


    def __insert_entity_color_in_story_text(self, story_text, start, end, label):
        story_text = story_text[:end] + "</mark>" + story_text[end:]
        story_text = (story_text[:start] +
                      '<mark style="background-color:' +
                      self.entity_colors[label] +
                      ';">' +
                      story_text[start:])
        return story_text


    def __add_entity_colors_to_story_text(self, story_text, story_id):
        entity_data = []
        first_id = sys.maxsize
        try:
            for index, row in self.analyzer.purpose_analysis.loc[
                             self.analyzer.purpose_analysis["text_id"] == story_id].iloc[::-1].iterrows():
                start = int(row["character_id"])
                end = start + len(row["text"])
                if end >= first_id:
                    print(f"cannot visualize story {story_id}'s overlapping {start} {end} ({first_id})")
                else:
                    story_text = self.__insert_entity_color_in_story_text(
                        story_text, start, end, row["label"])
                    first_id = start
        except Exception as e:
            print("__add_entity_colors_to_story_text", e)
        return story_text


    def __add_paragraphs_to_story_text(self, story_text):
        return "<p>" + re.sub("\n\n", "<p>", story_text)


    def __visualize_text_data(self):
        html_text = "<html><body>"
        html_text += self.__make_entity_bar_for_html()
        for story_text, story_id in self.text_tuples:
            if len(self.stories_selected) == 0 or int(story_id) in self.stories_selected:
                story_text = self.__add_entity_colors_to_story_text(story_text, int(story_id))
                html_text += "<hr>" + self.__add_paragraphs_to_story_text(story_text)
        html_text += "</body></html>"
        self.doc_webview.setHtml(html_text)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    corpus_ = Corpus.from_file("tests/storynavigator-testdata.tab")
    previewer = WidgetPreview(OWSNPurposeAnalysis)

