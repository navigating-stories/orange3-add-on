"""Modules required for SettingAnalysis widget in the Orange Story Navigator add-on.
"""

import json
import os
import re
import requests
import time
import pandas as pd
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from collections import Counter
from spacy.matcher import Matcher
from pathlib import Path


class SettingAnalyzer:
    """Class to analyze the setting in textual texts
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        language (str): ISO string of the language of the input text
        n_segments (int): Number of segments to split each text into
        text_tuples (list): ternary tuple: text (str), storyid, sentences (list)
        story_elements (list of lists): tokens with their Spacy analysis
        callback: function in widget to show the progress of this process
    """

    DATE_LABELS = ["DATE", "TIME"]
    EVENT_LABELS = ["EVENT"]
    LOCATION_LABELS = ["LOC", "FAC", "GPE"]
    ENTITY_GROUPS = [DATE_LABELS, EVENT_LABELS, LOCATION_LABELS]
    ENTITY_LABELS = DATE_LABELS + EVENT_LABELS + LOCATION_LABELS
    ENTITY_CACHE_FILE_NAME = "orange_story_navigator_wikidata_entity_cache.json"


    def __init__(self, language, n_segments, text_tuples, story_elements, user_defined_entities, callback=None):
        self.text_tuples = text_tuples
        self.n_segments = n_segments
        self.user_defined_entities = user_defined_entities
        self.callback = callback

        self.__setup_required_nlp_resources(language)
        self.nlp = util.load_spacy_pipeline(self.model)

        entities = self.extract_entities_from_table(story_elements)
        self.text_analysis = self.__process_texts(self.nlp, self.text_tuples, entities, self.callback)
        self.settings_analysis = self.__select_best_entities(self.text_analysis)


    def extract_entities_from_table(self, story_elements):
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        last_story_id = -1
        last_sentence = ""
        last_sentence_id = -1
        char_offset = 0
        entities = []
        for index, row in story_elements_df.iterrows():
            story_id = row["storyid"]
            if story_id != last_story_id:
                entities.append({})
                last_entity_class = "O"
                last_entity_start_id = -1
                last_sentence = ""
                char_offset = 0
            sentence = row["sentence"]
            sentence_id = row["sentence_id"]
            if sentence_id != last_sentence_id or story_id != last_story_id:
                if len(last_sentence) > 0:
                    char_offset += 1 + len(last_sentence)
                last_sentence = sentence
                last_sentence_id = sentence_id
            if row["spacy_ne"] == "O":
                last_entity_class = "O"
                last_entity_start_id = -1
            else:
                entity_class = re.sub("^.-", "", row["spacy_ne"])
                entity_iob = re.sub("-.*$", "", row["spacy_ne"])
                entity_start_id = int(row["token_start_idx"]) + char_offset
                if entity_class != last_entity_class or entity_iob == "B" or story_id != last_story_id:
                    entities[-1][entity_start_id] = {"text": row["token_text"], "label_": entity_class, "sentence_id": row["sentence_id"], "segment_id": row["segment_id"]}
                    last_entity_start_id = entity_start_id
                    last_entity_class = entity_class
                else:
                    entities[-1][last_entity_start_id]["text"] += " " + row["token_text"]
            last_story_id = story_id
        return entities


    def __setup_required_nlp_resources(self, language):
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            language (string): the ISO code for the language of the input texts (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if language == constants.NL:
            self.model = constants.NL_SPACY_MODEL
            self.time_words = constants.NL_TIME_WORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        elif language == constants.EN:
            self.model = constants.EN_SPACY_MODEL
            self.time_words = constants.EN_TIME_WORDS_FILE.read_text(encoding="utf-8").strip().split(os.linesep)
        else:
            raise ValueError(f"settingsanalysis.py: unknown language {language}")

        # self.entity_list = [line.split(",") for line in self.entity_list]

    def __sort_and_filter_results(self, results):
        results = [(x[0], x[1], int(x[2]), int(x[3]), int(x[4]), x[5], x[6]) for x in results]
        results_df = pd.DataFrame(results, columns=["text", "label", "text_id", "sentence_id", "segment_id", "character_id", "location_type"]).sort_values(by=["text_id", "character_id"])
        return results_df[["text", "label", "text_id", "sentence_id", "segment_id", "character_id", "location_type"]].reset_index(drop=True)


    def __process_texts(self, nlp, text_tuples, entities, callback=None):
        results = []
        index = 0
        for entities_per_text in entities:
            results.extend(self.__process_text(text_tuples[index][1], text_tuples[index][2], nlp, entities_per_text))
            index += 1
            if callback:
                callback(100*index/len(entities))
        return self.__sort_and_filter_results(results)


    def __analyze_text_with_list(self, sentences, nlp, user_defined_entities):
        matcher = Matcher(nlp.vocab)
        for entity_group in self.ENTITY_GROUPS:
            patterns = [[{"lower": entity_token} for entity_token in entity_text.lower().split()]
                for entity_text, entity_label in list(user_defined_entities.items())
                    if entity_label in entity_group]
            matcher.add(entity_group[0], patterns)
        results = dict()
        last_story_id = -1
        character_id = 0
        for story_id, sentence_id, segment_id, sentence_text in sentences:
            if story_id != last_story_id:
                last_story_id = story_id
                character_id = 0
            tokens = nlp(sentence_text)
            for m in matcher(tokens):
                results[character_id + tokens[m[1]].idx] = {
                    "text": " ".join([tokens[token_id].text for token_id in range(m[1], m[2])]),
                    "label_": nlp.vocab.strings[m[0]],
                    "sentence_id": sentence_id,
                    "segment_id": segment_id
                }
            character_id += len(sentence_text) + 1
        return results

#        return {tokens[m[1]].idx: {
#                    "text": " ".join([tokens[token_id].text for token_id in range(m[1], m[2])]),
#                    "label_": nlp.vocab.strings[m[0]],
#                    "sentence_id": list(tokens.sents).index(tokens[m[1]:m[2]].sent), # might fail with duplicate sentences
#                    "segment_id": 99 # temporary filler
#                } for m in matcher(tokens)}


    def __combine_analyses(self, spacy_analysis, list_analysis):
        combined_analysis = { entity_start_id:spacy_analysis[entity_start_id] 
                              for entity_start_id in spacy_analysis.keys()
                              if spacy_analysis[entity_start_id]["label_"] in self.ENTITY_LABELS }
        for start in list_analysis:
            combined_analysis[start] = list_analysis[start]
        return combined_analysis


    def __expand_locations(self, combined_analysis):
        for start in combined_analysis:
            combined_analysis[start]["location_type"] = ""
        entities_to_add = {}
        for start in combined_analysis.keys():
            if combined_analysis[start]["label_"] in self.LOCATION_LABELS:
                wikidata_info = self.__get_wikidata_info(combined_analysis[start]["text"])
                if len(wikidata_info) > 0 and "description" in wikidata_info[0]:
                    combined_analysis[start]["location_type"] = re.sub("^.* ", "", wikidata_info[0]["description"])
        for start in entities_to_add:
            combined_analysis[start] = entities_to_add[start]
        return combined_analysis


    def __filter_dates(self, combined_analysis):
        to_be_deleted = []
        for start in combined_analysis.keys():
            if combined_analysis[start]["label_"] in self.DATE_LABELS:
                words = combined_analysis[start]["text"].lower().split()
                if len(words) > 0 and words[-1] in self.time_words:
                    to_be_deleted.append(start)
        for start in to_be_deleted:
            del(combined_analysis[start])
        return combined_analysis


    def __process_text(self, text_id, sentences, nlp, spacy_analysis):
        first_key = list(spacy_analysis.keys())[0]
        list_analysis = self.__analyze_text_with_list(sentences, nlp, self.user_defined_entities) # will not be called if spacy found no entities
        combined_analysis = self.__combine_analyses(spacy_analysis, list_analysis)
        combined_analysis = self.__expand_locations(combined_analysis)
        combined_analysis = self.__filter_dates(combined_analysis)
        return [(combined_analysis[start]["text"],
                 combined_analysis[start]["label_"],
                 text_id,
                 combined_analysis[start]["sentence_id"],
                 combined_analysis[start]["segment_id"],
                 start,
                 combined_analysis[start]["location_type"]) for start in combined_analysis]


    def __normalize_entities(self, entity_data):
        entity_data_copy = entity_data.copy(deep=True)
        for index, row in entity_data_copy.iterrows():
            for entity_list in self.ENTITY_GROUPS:
                if len(entity_list) > 1 and row["label"] in entity_list[1:]:
                    entity_data_copy.loc[index, "label"] = entity_list[0]
                    break
        return entity_data_copy


    def __select_earliest_entities(self, entity_data):
        counts_series = entity_data[["text", "label", "text_id", "location_type"]].value_counts()
        counts_df = counts_series.reset_index(name="count").set_index(["text", "label", "text_id"])
        selected_indices = {}
        for index, row in entity_data.iterrows():
            key = " ".join([str(row["text_id"]), row["label"]])
            if (key not in selected_indices.keys() and
                (row["label"] not in self.LOCATION_LABELS or 
                 (re.search("^[A-Z]", row["text"]) and
                  re.search("^[A-Z]", row["location_type"])))):
                selected_indices[key] = [row["text"],
                                         row["label"],
                                         row["text_id"],
                                         row["location_type"],
                                         counts_df.loc[row["text"], row["label"], row["text_id"]]["count"]]
        return [list(x)[:4] for x in selected_indices.values()]


    def __lookup_selected_values(self, entity_data, selected_values):
        selected_column = len(entity_data) * [""]
        for entity_data_index, row in entity_data.iterrows():
            try:
                selected_values_index = selected_values.index([row["text"],
                                                               row["label"],
                                                               row["text_id"],
                                                               row["location_type"]])
                selected_column[entity_data_index] = "selected"
                selected_values.pop(selected_values_index)
            except:
                pass
        return selected_column


    def __select_best_entities(self, entity_data):
        entity_data_copy = self.__normalize_entities(entity_data)
        selected_values = self.__select_earliest_entities(entity_data_copy)
        selected_column = self.__lookup_selected_values(entity_data_copy,
                                                        selected_values)
        entity_data["selected"] = selected_column
        return entity_data


    def __get_wikidata_info(self, entity_name, find_property=False):
        if self.ENTITY_CACHE_FILE_NAME and os.path.isfile(self.ENTITY_CACHE_FILE_NAME):
            with open(self.ENTITY_CACHE_FILE_NAME, "r") as cache_file:
                cache = json.load(cache_file)
            cache_file.close()
        else:
            cache = {}
        if entity_name in cache and len(cache[entity_name]) > 0:
            return cache[entity_name]
        print("__get_wikidata_info: looking up", entity_name)
        url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "nl",
            "limit": 10,
            "uselang": "nl",
            "search": entity_name
        }
        if find_property:
            params["type"] = "property"
        try:
            response = requests.get(url, params=params)
            time.sleep(1)
            data = response.json()
        except:
            print("lookup failure: no network connection?")
            data = {}
        if 'search' in data.keys() and len(data["search"]) > 0:
            cache[entity_name] = data["search"]
        else:
            cache[entity_name] = [{"not found": True}]
        if self.ENTITY_CACHE_FILE_NAME:
            try:
                with open(self.ENTITY_CACHE_FILE_NAME, "w") as cache_file:
                    json.dump(cache, cache_file)
                cache_file.close()
            except:
                pass
        return cache[entity_name]
