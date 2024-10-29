"""Modules required for MeansAnalysis widget in the Orange Story Navigator add-on.
"""

import pandas as pd
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util


class MeansAnalyzer:
    """Class for extracting means from texts
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.11/

    Args:
        language (str): ISO string of the language of the input text
        story_elements (list of lists): tokens with their Spacy analysis
        verb_frames: verb frames indicating means
        means_strategy: strategy to identify means
        callback: function in widget to show the progress of this process
    """


    def __init__(self, language, story_elements, verb_frames, means_strategy, callback=None) -> None:
        self.language = language
        self.verb_frames = verb_frames
        self.means_strategy = means_strategy
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        self.__convert_str_columns_to_ints(story_elements_df)

        entities = self.__process_texts(story_elements_df, callback=callback)
        sentence_offsets = self.__compute_sentence_offsets(story_elements_df)
        entities_from_onsets = self.__convert_entities(entities, sentence_offsets)
        self.means_analysis = self.__sort_and_filter_results(entities_from_onsets)


    def __convert_str_columns_to_ints(self, story_elements_df) -> None:
        story_elements_df["storyid"] = story_elements_df["storyid"].apply(lambda x: int(x))
        story_elements_df["sentence_id"] = story_elements_df["sentence_id"].apply(lambda x: int(x))
        story_elements_df["token_start_idx"] = story_elements_df["token_start_idx"].apply(lambda x: int(x))
        story_elements_df["spacy_head_idx"] = story_elements_df["spacy_head_idx"].apply(lambda x: int(x))


    def __compute_sentence_offsets(self, story_elements_df) -> pd.DataFrame:
        sentences_df = story_elements_df.groupby(["storyid", "sentence_id"]).first().reset_index()[["storyid", "sentence_id", "sentence"]]
        char_offsets = []
        last_sentence = ""
        for index,row in sentences_df.iterrows():
            if row["sentence_id"] == 0:
                char_offset = 0
            else:
                char_offset += len(last_sentence) + 1
            char_offsets.append(char_offset)
            last_sentence = row["sentence"]
        sentences_df["char_offset"] = char_offsets
        return sentences_df[["storyid", "sentence_id", "char_offset"]].set_index(["storyid", "sentence_id"])


    def __convert_entities(self, entities, sentences_offsets) -> dict:
        entities_from_onsets = {}
        for storyid, sentence_id, sentence_data in entities:
            if storyid not in entities_from_onsets:
                entities_from_onsets[storyid] = {}
            for token_start_id in sentence_data:
                char_offset_sentence = sentences_offsets.loc[(storyid, sentence_id)]["char_offset"]
                entities_from_onsets[storyid][token_start_id + char_offset_sentence] = sentence_data[token_start_id]
        return entities_from_onsets


    def __convert_stories_to_sentences(self, story_elements_df) -> pd.DataFrame:
        # return story_elements_df.groupby(["storyid", "sentence_id"]).agg(lambda x: list(x)).reset_index()
        return { index: group.to_dict(orient="index") for index, group in story_elements_df.groupby(["storyid", "sentence_id"])}


    def __process_texts(self, story_elements_df, callback=None) -> list:
        sentences_dict = self.__convert_stories_to_sentences(story_elements_df)
        entities = []
        for sentence_dict_index, row_sentence_dict in sentences_dict.items():
            row_sentence_dict = { token["token_start_idx"]: token
                                 for token_idx, token in row_sentence_dict.items() }
            sentence_entities = self.__process_sentence(row_sentence_dict)
            if sentence_entities:
                entities.append([
                    sentence_dict_index[0],
                    sentence_dict_index[1],
                    sentence_entities])
        return entities

    def __matching_dependencies(self, sentence_df, entity_start_id, head_start_id, head_of_head_start_id) -> bool:
        if sentence_df[head_of_head_start_id]["spacy_tag"] not in ["VERB", "AUX"]:
            return False
        verb_frame_prepositions = [x[1] for x in self.verb_frames]
        return ((self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAMES and
                 [sentence_df[head_of_head_start_id]["spacy_lemma"],
                  sentence_df[entity_start_id]["spacy_lemma"]] in self.verb_frames) or
                (self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAME_PREPS and
                 sentence_df[entity_start_id]["spacy_lemma"] in verb_frame_prepositions) or
                (self.means_strategy == constants.MEANS_STRATEGY_SPACY_PREPS and
                 sentence_df[entity_start_id]["spacy_tag"] == "ADP"))

    def __expand_means_phrase(self, sentence_df, sentence_entities, char_offset, entity_start_id, head_start_id) -> None:
        child_entity_ids = self.__get_head_dependencies(sentence_df, char_offset, entity_start_id, head_start_id)
        processed_ids = []
        self.__prepend_tokens_to_means_phrase(sentence_df, sentence_entities, char_offset, head_start_id, child_entity_ids, processed_ids)
        self.__append_tokens_to_means_phrase(sentence_df, sentence_entities, char_offset, head_start_id, child_entity_ids, processed_ids)
        for child_entity_id in list(child_entity_ids) - processed_ids:
            print(sentence_df[entity_start_id]["token_text"], sentence_df[head_start_id]["token_text"],
                  "skipping means word", sentence_df[child_entity_id]["sentence"])

    def __prepend_tokens_to_means_phrase(self, sentence_df, sentence_entities, char_offset, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids, reverse=True):
            child_entity_text = sentence_df[child_entity_id]["token_text"]
            entity_gap_size = head_start_id - len(child_entity_text) - child_entity_id
            if child_entity_id not in processed_ids and entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[child_entity_id + char_offset] = {
                   "text": sentence_df[child_entity_id]["token_text"] + in_between_text + sentence_entities[head_start_id + char_offset]["text"],
                   "label_": "MEANS" }
                del(sentence_entities[head_start_id + char_offset])
                head_start_id = child_entity_id
                processed_ids.append(child_entity_id)

    def __extend_tokens_to_means_phrase(self, sentence_df, sentence_entities, char_offset, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids):
            entity_gap_size = child_entity_id - head_start_id - len(sentence_entities[head_start_id + char_offset]["text"])
            if child_entity_id not in processed_ids and entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[head_start_id + char_offset]["text"] += in_between_text + sentence_df[child_entity_id]["token_text"]
                processed_ids.append(child_entity_id)

    def __process_sentence(self, sentence_dict, char_offset=0) -> dict:
        sentence_entities = {}
        for entity_start_id, token_data in sorted(sentence_dict.items()):
            try:
                head_start_id = token_data["spacy_head_idx"]
                head_of_head_start_id = sentence_dict[head_start_id]["spacy_head_idx"]
                # nl head relations: PREP -> MEANS -> VERB
                # en head relations: MEANS -> PREP -> VERB
                if self.language == constants.EN:
                    entity_start_id, head_start_id = head_start_id, entity_start_id
                if self.__matching_dependencies(sentence_dict, entity_start_id, head_start_id, head_of_head_start_id):
                    try:
                        self.__add_sentence_entity(sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id, char_offset)
                    except Exception as e:
                        self.__log_key_error(e, token_data)
            except Exception as e:
                self.__log_key_error(e, token_data)
        return sentence_entities

    def __add_sentence_entity(self, sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id, char_offset) -> None:
        sentence_entities[entity_start_id + char_offset] = {
            "label_": "PREP", "text": sentence_dict[entity_start_id]["token_text"]}
        sentence_entities[head_start_id + char_offset] = {
            "label_": "MEANS", "text": sentence_dict[head_start_id]["token_text"]}
        sentence_entities[head_of_head_start_id + char_offset] = {
            "label_": "VERB", "text": sentence_dict[head_of_head_start_id]["token_text"]}
        self.__expand_means_phrase(sentence_dict, sentence_entities, char_offset, entity_start_id, head_start_id)

    def __log_key_error(self, e, token_data) -> None:
        print(f"key error: missing {str(e)} in", token_data["storyid"], token_data["token_text"], token_data["sentence"])

    def __get_head_dependencies(self, sentence_df, char_offset, entity_start_id, head_start_id) -> list:
        entity_ids = []
        for start_id, token in sorted(sentence_df.items()):
            if token["spacy_head_idx"] == head_start_id and start_id not in (entity_start_id, head_start_id):
                child_entity_ids = self.__get_head_dependencies(sentence_df, char_offset, entity_start_id, start_id)
                entity_ids.append(start_id)
                entity_ids.extend(child_entity_ids)
        return entity_ids

    def __sort_and_filter_results(self, entities) -> pd.DataFrame:
        results = [
            (story_entity["text"], story_entity["label_"], storyid, character_id)
            for storyid, story_entities in entities.items()
            for character_id, story_entity in story_entities.items()]
        results_df = pd.DataFrame(results, columns=["text", "label", "storyid", "character_id"])
        results_df = results_df.sort_values(by=["storyid", "character_id"])
        results_df["text_id"] = results_df["storyid"].apply(lambda x: "ST" + str(x))
        return results_df[["text", "label", "text_id", "character_id"]].reset_index(drop=True)