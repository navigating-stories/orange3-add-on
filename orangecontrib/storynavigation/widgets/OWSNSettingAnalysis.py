import os
import pathlib
import re

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
from storynavigation.modules.settinganalysis import SettingAnalyzer
import storynavigation.modules.util as util

from AnyQt.QtWidgets import QAbstractItemView, QHeaderView, QTableView
from storynavigation.widgets.OWSNActorAnalysis import DocumentTableView, DocumentListModel, DocumentsFilterProxyModel, _count_matches
from typing import Set
from orangecontrib.text.widgets.utils import widgets
from Orange.data.io import FileFormat


class OWSNSettingAnalysis(OWWidget, ConcurrentWidgetMixin):
    name = 'Setting Analysis'
    description = "Provides tools to help identify the main setting of a story."
    icon = "icons/setting_analysis_icon.png"
    priority = 6480
    settingsHandler = DomainContextHandler()
    settings_version = 2
    autocommit = Setting(True)
    language = 'nl'
    n_segments = 1
    user_defined_entities_file_name = os.path.join(
        str(constants.PKG),
        str(constants.RESOURCES_SUBPACKAGE),
        ("dutch" if language == "nl" else "english") + "_entities.csv")
    recent_files = [user_defined_entities_file_name]
    ENTITIES_FILE_YES = "yes: use this file"
    ENTITIES_FILE_NO = "no: skip this file"
    entity_colors = { "DATE": "lightblue",
                      "EVENT": "salmon",
                      "FAC": "silver",
                      "GPE": "lemonchiffon",
                      "LOC": "lightgreen",
                      "TIME": "thistle", }
    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(FileFormat.readers.values()),
                                  key=list(FileFormat.readers.values()).index)))

    class Inputs:
        story_elements = Input("Story elements", Table)


    class Outputs:
        dataset_level_data = Output('Intermediate settings', Table)


    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.stories_selected = []
        self.story_elements = []
        size_policy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.controlArea.setSizePolicy(size_policy)
        self.user_defined_entities = {}
        self.use_user_defined_entities_file = self.ENTITIES_FILE_YES
        self.read_entities_file(self.user_defined_entities_file_name)

        self.__make_language_selection_menu()
        self.__make_entities_file_dialog()
        self.__make_regexp_filter_dialog()
        self.__make_document_viewer()


    def __make_language_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="Language",
            value="language",
            items=constants.SUPPORTED_LANGUAGES,
            sendSelectedValue=True,
            currentIndex=1,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)
        self.select_language_combo.setEnabled(True)


    def __make_entities_file_selection_menu(self):
        self.select_language_combo = gui.comboBox(
            widget=self.controlArea,
            master=self,
            label="",
            value="use_user_defined_entities_file",
            items=[self.ENTITIES_FILE_YES, self.ENTITIES_FILE_NO],
            sendSelectedValue=True,
            currentIndex=0,
            callback=self.__process_use_user_defined_entities_file_change,
            sizePolicy=QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        self.controlArea.layout().addWidget(self.select_language_combo)
        self.select_language_combo.setEnabled(True)


    def __make_entities_file_dialog(self):
        # code copied from Corpus widget
        fbox = gui.widgetBox(self.controlArea, "Entities file", orientation=0)
        self.file_widget = widgets.FileWidget(
            recent_files=self.recent_files, icon_size=(16, 16),
            on_open=self.read_entities_file, dialog_format=self.dlgFormats,
            dialog_title='Open entities file',
            reload_label='Reload', browse_label='Browse',
            allow_empty=False, minimal_width=150,
        )
        fbox.layout().addWidget(self.file_widget)
        self.__make_entities_file_selection_menu()


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
        self.doc_webview.setHtml("")
        self.mainArea.layout().addWidget(self.splitter)


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


    def read_entities_file(self, user_defined_entities_file_name):
        self.user_defined_entities_file_name = user_defined_entities_file_name
        self.user_defined_entities = {}
        if self.use_user_defined_entities_file == self.ENTITIES_FILE_YES:
            user_defined_entities_lines = pathlib.Path(user_defined_entities_file_name).read_text(encoding="utf-8").strip().split("\n")
            for line in user_defined_entities_lines:
                try:
                    entity_class, entity_token = line.strip().split(",")
                    self.user_defined_entities[entity_token] = entity_class
                except:
                    pass
        if self.story_elements:
            self.reset_story_elements(self.story_elements)


    def __process_use_user_defined_entities_file_change(self):
        self.read_entities_file(self.user_defined_entities_file_name)


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
            self.__action_analyze_setting_wrapper()

            self.doc_list_model.setup_data(self.make_document_names(self.text_tuples), [text for text, _, _ in self.text_tuples])


    def make_document_names(self, text_tuples):
        document_names = []
        for _, text_id, _ in text_tuples:
            document_names.append("Document " + str(int(text_id) + 1))
        return document_names


    def make_text_tuples(self, story_elements):
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        current_story_id = ""
        current_story_text = ""
        current_story_sentences = []
        last_sentence_id = -1
        text_tuples = []
        for index, row in story_elements_df.iterrows():
            story_id = row["storyid"]
            if story_id != current_story_id:
                if current_story_id != "":
                    text_tuples.append((current_story_text, current_story_id, current_story_sentences))
                current_story_id = story_id
                current_story_text = ""
                current_story_sentences = []
                last_sentence = ""
            if row["sentence_id"] != last_sentence_id:
                current_story_text += row["sentence"] if current_story_text == "" else " " + row["sentence"]
                current_story_sentences.append((row["storyid"], row["sentence_id"], row["segment_id"], row["sentence"]))
                last_sentence_id = row["sentence_id"]
        if current_story_text != "":
            text_tuples.append((current_story_text, current_story_id, current_story_sentences))
        return text_tuples


    def reset_widget(self):
        self.corpus = None
        self.Warning.clear()


    def __action_analyze_setting_wrapper(self):
        self.start(self.action_analyze_setting)


    def action_analyze_setting(self, state: TaskState):
        def move_progress_bar(progress):
            if state.is_interruption_requested():
                raise InterruptedError
            state.set_progress_value(progress)

        self.analyzer = SettingAnalyzer(
             language=self.language,
             n_segments=int(self.n_segments),
             text_tuples=self.text_tuples,
             story_elements=self.story_elements,
             user_defined_entities=self.user_defined_entities,
             callback=move_progress_bar
        )


    def on_done(self, result) -> None:
        self.refresh_search()
        self.Outputs.dataset_level_data.send(table_from_frame(self.analyzer.settings_analysis))


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
        for index, row in self.analyzer.settings_analysis.loc[
                             self.analyzer.settings_analysis["text_id"] == int(story_id)].iloc[::-1].iterrows():
           start = int(row["character_id"])
           end = start + len(row["text"])
           story_text = self.__insert_entity_color_in_story_text(story_text,
                                                                 start,
                                                                 end,
                                                                 row["label"])
        return story_text


    def __add_paragraphs_to_story_text(self, story_text):
        return "<p>" + re.sub("\n\n", "<p>", story_text)


    def __visualize_text_data(self):
        html_text = "<html><body>"
        html_text += self.__make_entity_bar_for_html()
        for story_text, story_id, _ in self.text_tuples:
            if len(self.stories_selected) == 0 or int(story_id) in self.stories_selected:
                story_text = self.__add_entity_colors_to_story_text(story_text, story_id)
                html_text += "<hr>" + self.__add_paragraphs_to_story_text(story_text)
        html_text += "</body></html>"
        self.doc_webview.setHtml(html_text)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    corpus_ = Corpus.from_file("tests/storynavigator-testdata.tab")
    previewer = WidgetPreview(OWSNSettingAnalysis)

