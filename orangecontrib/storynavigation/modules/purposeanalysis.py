import pandas as pd
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
import sys

class PurposeAnalyzer:
    """Class for extracting purpose from texts

    Args:
        language (str): ISO string of the language of the input text
        story_elements (list of lists): tokens with their Spacy analysis
        verb_frames: verb frames indicating purpose
        purpose_strategy: strategy to identify purpose
        callback: function in widget to show the progress of this process
    """


    PURPOSE_LABELS = ['PURPOSE', 'ADVERB', 'CONTEXT']


    def __init__(self, language, story_elements, verb_frames, purpose_strategy, callback=None) -> None:
        self.language = language
        self.verb_frames = verb_frames
        self.purpose_strategy = purpose_strategy
        if self.language == constants.NL:
            self.first_person_words = constants.NL_FIRST_PERSON_WORDS
        else:
            self.first_person_words = constants.EN_FIRST_PERSON_WORDS
        story_elements_df = util.convert_orangetable_to_dataframe(story_elements)
        self.__convert_str_columns_to_ints(story_elements_df)
        entities = self.__process_texts(story_elements_df, callback=callback)
        sentence_offsets = self.__compute_sentence_offsets(story_elements_df)
        entities_from_onsets = self.__convert_entities(entities, sentence_offsets)
        entities_from_onsets = self.__add_missing_relation_parts(story_elements_df, entities_from_onsets, sentence_offsets)
        self.purpose_analysis = self.__sort_and_filter_results(entities_from_onsets)


    def __convert_str_columns_to_ints(self, story_elements_df) -> None:
        columns_to_convert = ["storyid", "sentence_id", "segment_id", "token_start_idx", "spacy_head_idx"]
        story_elements_df[columns_to_convert] = story_elements_df[columns_to_convert].astype(int)


    def __compute_sentence_offsets(self, story_elements_df) -> pd.DataFrame:
        sentences_df = story_elements_df.groupby(["storyid", "sentence_id"]).first().reset_index()[["storyid", "sentence_id", "segment_id", "sentence"]]
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
        return sentences_df[["storyid", "sentence_id", "segment_id", "char_offset"]].set_index(["storyid", "sentence_id"])


    def __get_missing_label(self, entities_from_onsets, storyid, sentence_id) -> list:
        labels_found = [entity['label_'] for entity in entities_from_onsets[storyid].values() if entity['sentence_id'] == sentence_id]
        return [x for x in self.PURPOSE_LABELS if x not in labels_found]


    def __add_missing_relation_part(self, entities_from_onsets, sentence_offsets, storyid, sentence_id, segment_id, previous_sentence) -> None:
        missing_labels = self.__get_missing_label(entities_from_onsets, storyid, sentence_id)
        if len(missing_labels) == 1:
            char_id_start = sentence_offsets.loc[(storyid, sentence_id - 1)]["char_offset"]
            char_id_end = sentence_offsets.loc[(storyid, sentence_id)]["char_offset"] - 1
            entities_from_onsets[storyid][char_id_start] = {
                'label_': missing_labels[0],
                'sentence_id': sentence_id,
                'segment_id': segment_id,
                'text': previous_sentence
            }


    def __add_missing_relation_parts(self, story_elements_df, entities_from_onsets, sentence_offsets) -> dict:
        sentences_df = story_elements_df.groupby(['storyid', 'sentence_id'])['sentence'].first()
        for storyid in entities_from_onsets:
            sentence_ids = {}
            for char_id in entities_from_onsets[storyid]:
                sentence_id = entities_from_onsets[storyid][char_id]['sentence_id']
                segment_id = entities_from_onsets[storyid][char_id]['segment_id']
                label = entities_from_onsets[storyid][char_id]['label_']
                if sentence_id in sentence_ids:
                    sentence_ids[sentence_id].append(label)
                else:
                    sentence_ids[sentence_id] = [label]
            for sentence_id in sentence_ids:
                if len(sentence_ids[sentence_id]) == 2 and sentence_id > 0:
                    self.__add_missing_relation_part(entities_from_onsets, 
                                                     sentence_offsets, 
                                                     storyid, 
                                                     sentence_id,
                                                     segment_id,
                                                     sentences_df.loc[storyid, sentence_id - 1])
        return entities_from_onsets


    def __convert_entities(self, entities, sentence_offsets) -> dict:
        entities_from_onsets = {}
        for storyid, sentence_id, segment_id, sentence_data in entities:
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
                    sentence_entities[list(sentence_entities.keys())[0]]["segment_id"],
                    sentence_entities])
            if callback:
                callback((100*(index + 1))/len(sentence_dict))
        return entities


    def __find_matching_dependencies(self, sentence_dict, entity_start_id, head_start_id, head_of_head_start_id) -> bool:
        try:
            if sentence_dict[head_of_head_start_id]["spacy_tag"] not in {"VERB", "AUX"}:
                return False
        except:
            return False
        verb_frame_adverbs = [x[1] for x in self.verb_frames]
        verb_frame_verbs = [x[0] for x in self.verb_frames]
        entity = sentence_dict[entity_start_id]
        head = sentence_dict[head_start_id]
        return ((self.purpose_strategy == constants.PURPOSE_STRATEGY_ADVERBS and
                 entity['spacy_lemma'].lower() in verb_frame_adverbs) or
                (self.purpose_strategy == constants.PURPOSE_STRATEGY_VERBS and
                 entity['spacy_lemma'].lower() in self.first_person_words and
                 head['spacy_lemma'].lower() in verb_frame_verbs))


    def __expand_phrase(self, sentence_dict, sentence_entities, entity_start_id, head_start_id, processed_ids) -> None:
        child_entity_ids = self.__get_head_dependencies(sentence_dict, entity_start_id, head_start_id)
        head_start_id = self.__prepend_tokens_to_purpose_phrase(sentence_dict, sentence_entities, head_start_id, child_entity_ids, processed_ids)
        self.__append_tokens_to_purpose_phrase(sentence_dict, sentence_entities, head_start_id, child_entity_ids, processed_ids)
        for child_entity_id in sorted(set(child_entity_ids).difference(processed_ids)):
            print(sentence_dict[entity_start_id]["token_text"], sentence_dict[head_start_id]["token_text"],
                  "skipping purpose word", sentence_dict[child_entity_id]["spacy_lemma"], sentence_dict[child_entity_id]["sentence"])


    def __prepend_tokens_to_purpose_phrase(self, sentence_dict, sentence_entities, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids, reverse=True):
            if child_entity_id in processed_ids:
                continue
            child_entity_text = sentence_dict[child_entity_id]["token_text"]
            entity_gap_size = head_start_id - len(child_entity_text) - child_entity_id
            if entity_gap_size in [1, 2, 3]:
                if entity_gap_size == 1:
                   in_between_text = " "
                elif entity_gap_size == 2:
                   in_between_text = ", "
                elif entity_gap_size == 3:
                   in_between_text = " , "
                sentence_entities[child_entity_id] = {
                    "text": child_entity_text + in_between_text + sentence_entities[head_start_id]["text"],
                    "sentence_id": sentence_dict[child_entity_id]["sentence_id"],
                    "segment_id": sentence_dict[child_entity_id]["segment_id"],
                    "label_": sentence_entities[head_start_id]['label_'] }
                del sentence_entities[head_start_id]
                head_start_id = child_entity_id
                processed_ids.add(child_entity_id)
        return head_start_id


    def __append_tokens_to_purpose_phrase(self, sentence_dict, sentence_entities, head_start_id, child_entity_ids, processed_ids) -> None:
        for child_entity_id in sorted(child_entity_ids):
            if child_entity_id in processed_ids:
                continue
            entity_gap_size = child_entity_id - head_start_id - len(sentence_entities[head_start_id]["text"])
            if entity_gap_size in [1, 2, 3]:
                if entity_gap_size == 1:
                   in_between_text = " "
                elif entity_gap_size == 2:
                   in_between_text = ", "
                elif entity_gap_size == 3:
                   in_between_text = " , "
                sentence_entities[head_start_id]["text"] += in_between_text + sentence_dict[child_entity_id]["token_text"]
                processed_ids.add(child_entity_id)


    def __process_sentence(self, sentence_dict) -> dict:
        sentence_entities = {}
        for entity_start_id, token_data in sorted(sentence_dict.items()):
            try:
                head_start_id = token_data.get("spacy_head_idx")
                head_of_head_start_id = sentence_dict.get(head_start_id, {}).get("spacy_head_idx")
                if self.language == constants.EN:
                    entity_start_id, head_start_id = head_start_id, entity_start_id
                if self.__find_matching_dependencies(sentence_dict, entity_start_id, head_start_id, head_of_head_start_id):
                    if self.purpose_strategy == constants.PURPOSE_STRATEGY_VERBS:
                        self.__add_sentence_entity_verb(sentence_dict, sentence_entities, head_start_id)
                    elif self.purpose_strategy == constants.PURPOSE_STRATEGY_ADVERBS:
                        if head_start_id == head_of_head_start_id:
                            print("overlapping relation parts!", sentence_dict[head_start_id]["token_text"])
                        self.__add_sentence_entity_adverb(sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id)
            except AttributeError as e:
                self.__log_error("attribute error", e, token_data)
            except KeyError as e:
                self.__log_error("key error", e, token_data)
        return sentence_entities


    def __log_error(self, error_phrase, e, token_data) -> None:
        print(f"{error_phrase}: missing {e} in {token_data['storyid']} {token_data['token_text']} {token_data['sentence']}")


    def __add_sentence_entity_adverb(self, sentence_dict, sentence_entities, entity_start_id, head_start_id, head_of_head_start_id) -> None:
        entity = sentence_dict[entity_start_id]
        sentence_id = entity["sentence_id"]
        segment_id = entity["segment_id"]
        reversed_order = ([x[2] for x in self.verb_frames if x[1] == entity["token_text"].lower()] == ['yes'])
        head_label, head_of_head_label = ['PURPOSE', 'CONTEXT'] if reversed_order else ['CONTEXT', 'PURPOSE']
        sentence_entities[entity_start_id] = {
            "label_": "ADVERB", 
            "sentence_id": sentence_id,
            "segment_id": segment_id,
            "text": entity["token_text"]}
        sentence_entities[head_start_id] = {
            "label_": head_label,
            "sentence_id": sentence_id,
            "segment_id": segment_id,
            "text": sentence_dict[head_start_id]["token_text"]}
        processed_ids = {entity_start_id, head_start_id, head_of_head_start_id}
        self.__expand_phrase(sentence_dict, 
                                     sentence_entities, 
                                     entity_start_id, 
                                     head_start_id, 
                                     processed_ids=processed_ids)
        if head_of_head_start_id != head_start_id:
            sentence_entities[head_of_head_start_id] = {
                "label_": head_of_head_label,
                "sentence_id": sentence_id,
                "segment_id": segment_id,
                "text": sentence_dict[head_of_head_start_id]["token_text"]}
            self.__expand_phrase(sentence_dict, 
                                         sentence_entities, 
                                         entity_start_id, 
                                         head_of_head_start_id, 
                                         processed_ids=processed_ids)


    def __add_sentence_entity_verb(self, sentence_dict, sentence_entities, entity_start_id) -> None:
        entity = sentence_dict[entity_start_id]
        sentence_id = entity["sentence_id"]
        segment_id = entity["segment_id"]
        sentence_entities[entity_start_id] = {
            "label_": "PURPOSE", 
            "sentence_id": sentence_id,
            "segment_id": segment_id,
            "text": entity["token_text"]}
        self.__expand_phrase(sentence_dict, sentence_entities, entity_start_id, entity_start_id, processed_ids=set())


    def __get_head_dependencies(self, sentence_dict, entity_start_id, head_start_id) -> list:
        entity_ids = []
        for start_id, token in sorted(sentence_dict.items()):
            if token["spacy_head_idx"] == head_start_id and start_id not in {entity_start_id, head_start_id}:
                entity_ids.append(start_id)
                entity_ids.extend(self.__get_head_dependencies(sentence_dict, entity_start_id, start_id))
        return entity_ids


    def __sort_and_filter_results(self, entities) -> pd.DataFrame:
        results = [(entity["text"], entity["label_"], storyid, entity["segment_id"], entity["sentence_id"], char_id)
                   for storyid, story_entities in entities.items()
                   for char_id, entity in story_entities.items()]
        results_df = pd.DataFrame(results, columns=["text", "label", "storyid", "segment_id", "sentence_id", "character_id"])
        results_df.sort_values(by=["storyid", "character_id"], inplace=True)
        results_df["text_id"] = results_df["storyid"].astype(int)
        return results_df[["text", "label", "text_id", "segment_id", "sentence_id", "character_id"]].reset_index(drop=True)
