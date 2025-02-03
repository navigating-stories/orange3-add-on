import pandas as pd
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util


class MeansAnalyzer:
    """Class for extracting means from texts

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
        columns_to_convert = ["storyid", "segment_id", "sentence_id", "token_start_idx", "spacy_head_idx"]
        story_elements_df[columns_to_convert] = story_elements_df[columns_to_convert].astype(int)


    def __compute_sentence_offsets(self, story_elements_df) -> pd.DataFrame:
        sentences_df = story_elements_df.groupby(["storyid", "sentence_id"]).first().reset_index()[["storyid", "segment_id", "sentence_id", "sentence"]]
        char_offsets = []
        last_sentence = ""
        for sentence_id, sentence in zip(sentences_df["sentence_id"],
                                         sentences_df["sentence"]):
            if sentence_id == sentences_df.iloc[0]["sentence_id"]:
                char_offset = 0
            else:
                char_offset += len(last_sentence) + 1
            char_offsets.append(char_offset)
            last_sentence = sentence
        sentences_df["char_offset"] = char_offsets
        return sentences_df[["storyid", "segment_id", "sentence_id", "char_offset"]].set_index(["storyid", "sentence_id"])


    def __convert_entities(self, entities, sentence_offsets) -> dict:
        entities_from_onsets = {}
        for storyid, sentence_id, sentence_data in entities:
            story_entities = entities_from_onsets.setdefault(storyid, {})
            char_offset_sentence = sentence_offsets.loc[(storyid, sentence_id)]["char_offset"]
            for token_start_id, token_data in sentence_data.items():
                story_entities[token_start_id + char_offset_sentence] = token_data
        return entities_from_onsets


    def __convert_stories_to_sentences(self, story_elements_df) -> pd.DataFrame:
        return { index: group.to_dict(orient="index") for index, group in story_elements_df.groupby(["storyid", "sentence_id"])}


    def __process_texts(self, story_elements_df, callback=None) -> list:
        sentence_dict = self.__convert_stories_to_sentences(story_elements_df)
        entities = []
        for index, (sentence_dict_index, row_sentence_dict) in enumerate(sentence_dict.items()):
            row_sentence_dict = { token["token_start_idx"]: token
                                 for token_idx, token in row_sentence_dict.items() }
            sentence_entities = self.__process_sentence(row_sentence_dict)
            if sentence_entities:
                entities.append([
                    sentence_dict_index[0],
                    sentence_dict_index[1],
                    sentence_entities])
            if callback:
                callback((100*(index + 1))/len(sentence_dict))
        return entities


    def __find_matching_dependencies(self, sentence_df, entity_start_id, head_start_id, head_of_head_start_id) -> bool:
        if sentence_df[head_of_head_start_id]["spacy_tag"] not in {"VERB", "AUX"}:
            return False
        verb_frame_prepositions = [x[1] for x in self.verb_frames]
        entity = sentence_df[entity_start_id]
        head_of_head = sentence_df[head_of_head_start_id]
        return ((self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAMES and
                 [head_of_head["spacy_lemma"], entity["spacy_lemma"]] in self.verb_frames) or
                (self.means_strategy == constants.MEANS_STRATEGY_VERB_FRAME_PREPS and
                 entity["spacy_lemma"] in verb_frame_prepositions) or
                (self.means_strategy == constants.MEANS_STRATEGY_SPACY_PREPS and
                 entity["spacy_tag"] == "ADP"))


    def __expand_means_phrase(self, sentence_df, sentence_entities, entity_start_id, head_start_id) -> None:
        child_entity_ids = self.__get_head_dependencies(sentence_df, entity_start_id, head_start_id)
        processed_ids = set()
        head_start_id = self.__prepend_tokens_to_means_phrase(sentence_df, sentence_entities, head_start_id, child_entity_ids, processed_ids)
        self.__append_tokens_to_means_phrase(sentence_df, sentence_entities, head_start_id, child_entity_ids, processed_ids)
        for child_entity_id in set(child_entity_ids) - processed_ids:
            print(sentence_df[entity_start_id]["token_text"], sentence_df[head_start_id]["token_text"],
                  "skipping means word", sentence_df[child_entity_id]["sentence"])


    def __prepend_tokens_to_means_phrase(self, sentence_df, sentence_entities, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids, reverse=True):
            if child_entity_id in processed_ids:
                continue
            child_entity_text = sentence_df[child_entity_id]["token_text"]
            entity_gap_size = head_start_id - len(child_entity_text) - child_entity_id
            if entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[child_entity_id] = {
                    "text": child_entity_text + in_between_text + sentence_entities[head_start_id]["text"],
                    "segment_id": sentence_df[child_entity_id]["segment_id"],
                    "sentence_id": sentence_df[child_entity_id]["sentence_id"],
                    "label_": "MEANS" }
                del sentence_entities[head_start_id]
                head_start_id = child_entity_id
                processed_ids.add(child_entity_id)
        return head_start_id


    def __append_tokens_to_means_phrase(self, sentence_df, sentence_entities, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids):
            if child_entity_id in processed_ids:
                continue
            entity_gap_size = child_entity_id - head_start_id - len(sentence_entities[head_start_id]["text"])
            if entity_gap_size in [1, 2]:
                in_between_text = " " if entity_gap_size == 1 else ", "
                sentence_entities[head_start_id]["text"] += in_between_text + sentence_df[child_entity_id]["token_text"]
                processed_ids.add(child_entity_id)


    def __process_sentence(self, sentence_dict) -> dict:
        sentence_entities = {}
        for entity_start_id, token_data in sorted(sentence_dict.items()):
            try:
                head_start_id = token_data.get("spacy_head_idx")
                head_of_head_start_id = sentence_dict.get(head_start_id, {}).get("spacy_head_idx")
                # nl head relations: PREP -> MEANS -> VERB
                # en head relations: MEANS -> PREP -> VERB
                if self.language == constants.EN:
                    entity_start_id, head_start_id = head_start_id, entity_start_id
                if self.__find_matching_dependencies(sentence_dict, entity_start_id, head_start_id, head_of_head_start_id):
                    self.__add_sentence_entity(sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id)
            except AttributeError as e:
                self.__log_key_error(e, token_data)
            except KeyError as e:
                self.__log_key_error(e, token_data)
        return sentence_entities


    def __add_sentence_entity(self, sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id) -> None:
        entity = sentence_dict[entity_start_id]
        segment_id = entity["segment_id"]
        sentence_id = entity["sentence_id"]
        sentence_entities[entity_start_id] = {
            "label_": "PREP", 
            "segment_id": segment_id,
            "sentence_id": sentence_id,
            "text": entity["token_text"]}
        sentence_entities[head_start_id] = {
            "label_": "MEANS",
            "segment_id": segment_id,
            "sentence_id": sentence_id,
            "text": sentence_dict[head_start_id]["token_text"]}
        sentence_entities[head_of_head_start_id] = {
            "label_": "VERB",
            "segment_id": segment_id,
            "sentence_id": sentence_id,
            "text": sentence_dict[head_of_head_start_id]["token_text"]}
        self.__expand_means_phrase(sentence_dict, sentence_entities, entity_start_id, head_start_id)


    def __log_key_error(self, e, token_data) -> None:
        print(f"key error: missing {e} in {token_data['storyid']} {token_data['token_text']} {token_data['sentence']}")


    def __get_head_dependencies(self, sentence_df, entity_start_id, head_start_id) -> list:
        entity_ids = []
        for start_id, token in sorted(sentence_df.items()):
            if token["spacy_head_idx"] == head_start_id and start_id not in {entity_start_id, head_start_id}:
                entity_ids.append(start_id)
                entity_ids.extend(self.__get_head_dependencies(sentence_df, entity_start_id, start_id))
        return entity_ids


    def __sort_and_filter_results(self, entities) -> pd.DataFrame:
        results = [(entity["text"], entity["label_"], storyid, entity["segment_id"], entity["sentence_id"], char_id)
                   for storyid, story_entities in entities.items()
                   for char_id, entity in story_entities.items()]
        results_df = pd.DataFrame(results, columns=["text", "label", "storyid", "segment_id", "sentence_id", "character_id"])
        results_df.sort_values(by=["storyid", "character_id"], inplace=True)
        results_df["text_id"] = results_df["storyid"].astype(int)
        return results_df[["text", "label", "text_id", "segment_id", "sentence_id", "character_id"]].reset_index(drop=True)
