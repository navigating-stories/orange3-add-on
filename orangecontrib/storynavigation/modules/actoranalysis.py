"""Modules required for Actor Analysis widget in Story Navigator.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
import storynavigation.modules.constants as constants
import storynavigation.modules.util as util
from spacy import displacy

class ActorTagger:
    """Class to perform NLP analysis of actors in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    def __init__(self, lang):
        self.lang = lang
        self.stopwords = None
        self.__setup_required_nlp_resources(lang)
        self.html_result = ""
        self.num_sents_in_stories = {}
        self.entity_prominence_scores = {}
        self.prominence_score_max = 0.0
        self.prominence_score_min = 0.0
        self.tagging_cache = {}

    def __filter_custom_word_matches(self, story_elements_df, selected_stories, cust_tag_cols):
        cols = []
        story_elements_df[cust_tag_cols] = story_elements_df[cust_tag_cols].astype(str)
        words_tagged_with_current_cust_tags_frame = story_elements_df

        if selected_stories is not None:
            cols = ['segment_id']
            words_tagged_with_current_cust_tags_frame = story_elements_df[story_elements_df['storyid'].isin(selected_stories)]
        else:
            cols = ['storyid', 'segment_id']
        
        combined_df = pd.DataFrame()
        for cust_tag_col in cust_tag_cols:
            current_frame = words_tagged_with_current_cust_tags_frame.groupby(cols+[cust_tag_col])[cust_tag_col].agg('count').to_frame("freq").reset_index()
            c_col = [cust_tag_col] * len(current_frame)
            current_frame['classification'] = c_col
            current_frame.rename(columns={cust_tag_col: 'category'}, inplace=True)
            combined_df = pd.concat([combined_df, current_frame], axis=0)
                
        combined_df = combined_df[~combined_df['category'].isin(['?', 'nan'])]

        return combined_df.reset_index(drop=True)

    def __filter_rows(self, story_elements_df, pos_tags):
        story_elements_df = story_elements_df.copy()
        story_elements_df['story_navigator_tag'] = story_elements_df['story_navigator_tag'].astype(str)
        story_elements_df['spacy_tag'] = story_elements_df['spacy_tag'].astype(str) 
        matched_df = story_elements_df[story_elements_df['story_navigator_tag'].isin(pos_tags) | story_elements_df['spacy_tag'].isin(pos_tags)]
        
        matched_df = matched_df.copy()
        
        matched_df['merged_tags'] = np.where(matched_df['story_navigator_tag'] == '-', matched_df['spacy_tag'], matched_df['story_navigator_tag'])
        matched_df['token_start_idx'] = matched_df['token_start_idx'].astype(str)
        matched_df['token_end_idx'] = matched_df['token_end_idx'].astype(str)

        return matched_df
    
    def __do_tagging(self, df):
        ents = []
        if len(df) > 0:
            displacy_tags_list = df['displacy_tag_strings'].tolist()
            for displacy_tag in displacy_tags_list:
                dtag = displacy_tag.split(' | ')
                ents.append({"start": int(float(dtag[0])), "end": int(float(dtag[1])), "label": dtag[2]})

            ents = util.remove_duplicate_tagged_entities(ents) 

        return ents
    
    def __print_html_no_highlighted_tokens(self, sentences):
        html = ''
        for sentence in sentences:
            doc = {"text": sentence, "ents": []}
            options = {"ents": [], "colors": {}}
            html += displacy.render(doc, style="ent", options=options, manual=True)
        return html
    
    def __filter_and_sort_matched_dataframe_by_sentence(self, df, sent, sents):
        order_mapping = {value: index for index, value in enumerate(sents)}
        matched_sent_df = df[df['sentence'] == sent]
        matched_sent_df = matched_sent_df.copy()
        matched_sent_df.loc[:, 'sorting_key'] = matched_sent_df['sentence'].map(lambda value: order_mapping.get(value, len(sents)))
        matched_sent_df_sorted = matched_sent_df.sort_values(by='sorting_key').drop('sorting_key', axis=1)        
        return matched_sent_df_sorted
        
    
    def __do_custom_tagging(self, df, cust_tag_cols):
        df = df.copy()
        
        df_filtered = df.dropna(subset=cust_tag_cols, how='all')
        multi_custom_tags = []
        ents = []

        for index, row in df_filtered.iterrows():
            current_row_ents = []
            for col in cust_tag_cols:
                if not pd.isna(row[col]) and (row[col] not in ['?', 'nan']):
                    current_row_ents.append({"start": int(row['token_start_idx']), "end": int(row['token_end_idx']), "label": row[col]})                    

            # Convert each dictionary to a tuple of (key, value) pairs
            tuple_list = [tuple(sorted(d.items())) for d in current_row_ents]
            # Use a set to remove duplicates
            unique_tuples = set(tuple_list)
            # Convert back to dictionaries
            current_row_ents = [dict(t) for t in unique_tuples]

            if len(current_row_ents) > 1:
                concat_labels = '|'.join([d['label'] for d in current_row_ents if 'label' in d and isinstance(d['label'], str)])
                multi_custom_tags.append(concat_labels)
                ents.append({"start": int(current_row_ents[0]['start']), "end": int(current_row_ents[0]['end']), "label": concat_labels})
            elif len(current_row_ents) == 1:
                ents.extend(current_row_ents)
        return ents, multi_custom_tags

    def __postag_sents(
            self, sentences, nouns, subjs, custom, selected_prominence_metric, prominence_score_min, story_elements_df
    ):
        html = ""
        
        if story_elements_df is None:
            return self.__print_html_no_highlighted_tokens(sentences)

        pos_tags = []
        custom_tag_columns = []
        custom_tags = []
        if nouns:
            pos_tags.append("NSP")
            pos_tags.append("NSNP")
        if subjs:
            pos_tags.append("SP")
            pos_tags.append("SNP")
        if custom:
            custom_tag_columns, custom_tags = util.get_custom_tags_list_and_columns(story_elements_df)

        if len(pos_tags) == 0 and len(custom_tags) == 0:
            return self.__print_html_no_highlighted_tokens(sentences)
        
        matched_df = None
        
        for sentence in sentences:
            ents = []
            if custom:
                nents = []
                cents = []
                new_color_map = constants.COLOR_MAP
                if nouns or subjs:
                    matched_df = self.__filter_rows(story_elements_df, pos_tags)                    
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(matched_df, sentence, sentences)
                    matched_sent_df_sorted['displacy_tag_strings'] = matched_sent_df_sorted['token_start_idx'] + ' | ' + matched_sent_df_sorted['token_end_idx'] + ' | ' + matched_sent_df_sorted['merged_tags']
                    nents = self.__do_tagging(matched_sent_df_sorted)
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(story_elements_df, sentence, sentences)
                    cents, multi_tags = self.__do_custom_tagging(matched_sent_df_sorted, custom_tag_columns)
                    for custom_pos_tag in custom_tags + multi_tags:
                        new_color_map[custom_pos_tag] = constants.CUSTOMTAG_HIGHLIGHT_COLOR
                    options = {"ents": pos_tags + custom_tags + multi_tags, "colors": new_color_map}
                else:
                    matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(story_elements_df, sentence, sentences)
                    cents, multi_tags = self.__do_custom_tagging(matched_sent_df_sorted, custom_tag_columns)
                    for custom_pos_tag in custom_tags + multi_tags:
                        new_color_map[custom_pos_tag] = constants.CUSTOMTAG_HIGHLIGHT_COLOR
                    options = {"ents": custom_tags + multi_tags, "colors": new_color_map}
                ents = nents + cents
            else:
                matched_df = self.__filter_rows(story_elements_df, pos_tags)
                matched_sent_df_sorted = self.__filter_and_sort_matched_dataframe_by_sentence(matched_df, sentence, sentences)
                
                matched_sent_df_sorted['displacy_tag_strings'] = matched_sent_df_sorted['token_start_idx'] + ' | ' + matched_sent_df_sorted['token_end_idx'] + ' | ' + matched_sent_df_sorted['merged_tags']
                ents = self.__do_tagging(matched_sent_df_sorted)
                options = {"ents": pos_tags, "colors": constants.COLOR_MAP}

            # Convert each dictionary to a tuple of (key, value) pairs
            tuple_list = [tuple(sorted(d.items())) for d in ents]
            # Use a set to remove duplicates
            unique_tuples = set(tuple_list)
            # Convert back to dictionaries
            ents = [dict(t) for t in unique_tuples]

            doc = {"text": sentence, "ents": ents}    
            print('ACTORS.PY', doc)
            html += displacy.render(doc, style="ent", options=options, manual=True)
            
        

        if custom:
            self.html_result = util.remove_span_tags_except_custom(html)
            return self.html_result
        else:
            self.html_result = util.remove_span_tags(html)
            return self.html_result

    def postag_text(
        self, text, nouns: bool, subjs: bool, custom: bool, selected_prominence_metric: float, prominence_score_min: int, story_elements_df: pd.DataFrame | None
    ):
        """POS-tags story text and returns HTML string which encodes the the tagged text, ready for rendering in the UI

        Args:
            text (string): Story text
            nouns (boolean): whether noun tokens should be tagged
            subjs (boolean): whether subject tokens should be tagged
            custom (boolean): whether custom tags should be highlighted or not
            selected_prominence_metric (float): the selected metric by which to calculate the word prominence score
            prominence_score_min (int): the minimum prominence score for an entity which qualifies it to be tagged
            story_elements_df (pandas.DataFrame): a dataframe with all required nlp tagging information

        Returns:
            string: HTML string representation of POS tagged text
        """

        if story_elements_df is None or (len(story_elements_df) == 0):
            sentences = util.preprocess_text(text)
            return self.__print_html_no_highlighted_tokens(sentences)

        story_elements_df['sentence_id'] = pd.to_numeric(story_elements_df['sentence_id'])
        story_elements_df['storyid'] = pd.to_numeric(story_elements_df['storyid'])
        sorted_df = story_elements_df.sort_values(by=['storyid', 'sentence_id'], ascending=True)
        sentences_df = sorted_df[['sentence','storyid','sentence_id']].drop_duplicates()
        sentences = sentences_df['sentence'].tolist()      
        
        selected_storyid = story_elements_df['storyid'].unique().tolist()[0]
        specific_tag_choice_html = (str(int(nouns)) + str(int(subjs)) + str(int(custom)))
        if selected_storyid in self.tagging_cache:
            if specific_tag_choice_html in self.tagging_cache[str(selected_storyid)]: # tagging info already generated, just lookup cached results
                return self.tagging_cache[str(selected_storyid)][specific_tag_choice_html]
            else:
                self.tagging_cache[str(selected_storyid)][specific_tag_choice_html] = self.__postag_sents(sentences, nouns, subjs, custom, selected_prominence_metric, prominence_score_min, story_elements_df)
        else:
            self.tagging_cache[str(selected_storyid)] = {}
            self.tagging_cache[str(selected_storyid)][specific_tag_choice_html] = self.__postag_sents(sentences, nouns, subjs, custom, selected_prominence_metric, prominence_score_min, story_elements_df)

        return self.tagging_cache[str(selected_storyid)][specific_tag_choice_html]
    
    def calculate_customfreq_table(self, df, selected_stories=None):
        """Prepares data table for piping to Output variable of widget: frequencies of custom tokens by user

        Args:
            df (pandas dataframe): the dataframe of all categories of custom words by the user

        Returns:
            data table (pandas dataframe)
        """
        
        cust_tag_cols, cust_tag_names = util.get_custom_tags_list_and_columns(df)
        df['token_text'] = df['token_text'].astype(str)
        df['token_text_lowercase'] = df['token_text'].str.lower()

        if df is None:
            return pd.DataFrame([], columns=constants.CUSTOMFREQ_TABLE_HEADER)

        return self.__filter_custom_word_matches(df, selected_stories=selected_stories, cust_tag_cols=cust_tag_cols)

    def __prepare_story_elements_frame_for_filtering(self, story_elements_df):
        story_elements_df = story_elements_df.copy()
        strcols = ['token_text', 'sentence', 'associated_action']

        for col in strcols:
            story_elements_df[col] = story_elements_df[col].astype(str)

        try:
            story_elements_df['is_sentence_subject_boolean'] = story_elements_df['is_sentence_subject_boolean'].astype(int)
        except ValueError:
            pass

        try:
            story_elements_df['active_voice_subject_boolean'] = story_elements_df['active_voice_subject_boolean'].astype(int)
        except ValueError:
            pass

        story_elements_df['storyid'] = 'ST' + story_elements_df['storyid'].astype(str)
        story_elements_df['segment_id'] = 'SE' + story_elements_df['segment_id'].astype(str)

        return story_elements_df
    
    # Custom aggregation function
    def __custom_agg_agency(self, row):
        return row['agency'] / self.num_sents_in_stories[(row['storyid'].replace('ST',''))]
    
    def __custom_agg_prominence(self, row):
        return row['prominence_sf'] / self.num_sents_in_stories[(row['storyid'].replace('ST',''))]
    
    # def __generate_tagging_cache(self, story_elements_df, callback=None):
    #     result = {}
    #     c = 1
    #     for storyid in story_elements_df['storyid'].unique().tolist():
    #         result[storyid] = {}
    #         sents_df = story_elements_df[story_elements_df['storyid'] == storyid]
    #         sorted_df = sents_df.sort_values(by=['sentence_id'], ascending=True)
    #         sents = sorted_df['sentence'].unique().tolist()
    #         result[storyid]['000'] = self.__postag_sents(sents, 0, 0, 0, None, None, story_elements_df)
    #         result[storyid]['001'] = self.__postag_sents(sents, 0, 0, 1, None, None, story_elements_df)
    #         result[storyid]['010'] = self.__postag_sents(sents, 0, 1, 0, None, None, story_elements_df)
    #         result[storyid]['011'] = self.__postag_sents(sents, 0, 1, 1, None, None, story_elements_df)
    #         result[storyid]['100'] = self.__postag_sents(sents, 1, 0, 0, None, None, story_elements_df)
    #         result[storyid]['101'] = self.__postag_sents(sents, 1, 0, 1, None, None, story_elements_df)
    #         result[storyid]['110'] = self.__postag_sents(sents, 1, 1, 0, None, None, story_elements_df)
    #         result[storyid]['111'] = self.__postag_sents(sents, 1, 1, 1, None, None, story_elements_df)
    #         c+=1
    #         if callback:
    #             increment = ((c/len(story_elements_df['storyid'].unique().tolist()))*80)
    #             callback(increment)
    #     return result

    def generate_actor_analysis_results(self, story_elements_df, callback=None):
        story_elements_df = story_elements_df.copy()
        # self.tagging_cache = self.__generate_tagging_cache(story_elements_df, callback)
        self.num_sents_in_stories = story_elements_df.groupby('storyid')['sentence'].nunique().to_dict()
        story_elements_df = self.__prepare_story_elements_frame_for_filtering(story_elements_df)
        result_df = pd.DataFrame()
        
        word_col = next((word for word in story_elements_df.columns if word.startswith('custom_')), None)
        if word_col is None:
            word_col = 'token_text_lowercase'
            
        c = 1
        rel_rows = story_elements_df[story_elements_df['story_navigator_tag'].isin(['SP', 'SNP'])][word_col].unique().tolist()

        for word in rel_rows: # loop through unique words that are subjects of sentences in a story
            df_word = story_elements_df[story_elements_df[word_col] == str(word)]
            raw_freq_df = df_word.groupby(['storyid', 'segment_id', word_col])[word_col].agg('count').to_frame("raw_freq").reset_index()
            
            subj_freq_df = None
            if df_word['is_sentence_subject_boolean'].dtype == int:
                subj_freq_df = df_word.groupby(['storyid', 'segment_id', word_col])['is_sentence_subject_boolean'].agg('sum').to_frame("subj_freq").reset_index()
            else:
                df_word = df_word.copy()
                df_word['is_sentence_subject_boolean'] = df_word['is_sentence_subject_boolean'].astype(bool)
                
                rel_df = df_word[df_word['is_sentence_subject_boolean']]
                subj_freq_df = rel_df.groupby(['storyid', 'segment_id', word_col])['is_sentence_subject_boolean'].agg('nunique').to_frame("subj_freq").reset_index()

            agency_df = None
            if df_word['active_voice_subject_boolean'].dtype == int:
                agency_df = df_word.groupby(['storyid', 'segment_id', word_col])['active_voice_subject_boolean'].agg('sum').to_frame("agency").reset_index()
            else:
                df_word['active_voice_subject_boolean'] = df_word['active_voice_subject_boolean'].astype(bool)
                rel_df = df_word[df_word['active_voice_subject_boolean']]
                agency_df = rel_df.groupby(['storyid', 'segment_id', word_col])['active_voice_subject_boolean'].agg('nunique').to_frame("agency").reset_index()

            agency_df['agency'] = agency_df.apply(lambda row: self.__custom_agg_agency(row), axis=1)
            prominence_df = df_word.groupby(['storyid', 'segment_id', word_col])['is_sentence_subject_boolean'].agg('sum').to_frame("prominence_sf").reset_index()
            prominence_df['prominence_sf'] = prominence_df.apply(lambda row: self.__custom_agg_prominence(row), axis=1)
            combined_df = pd.merge(raw_freq_df, pd.merge(subj_freq_df, pd.merge(agency_df, prominence_df, on=['storyid', 'segment_id', word_col], how='outer')), on=['storyid', 'segment_id', word_col], how='outer')
            result_df = pd.concat([result_df, combined_df], axis=0, ignore_index=True)

            c+=1
            if callback:
                callback((c / len(rel_rows) * 100))

        
        for word in result_df[word_col].unique().tolist():
            self.entity_prominence_scores[word] = result_df[result_df[word_col] == word]['prominence_sf'].tolist()[0]

        self.prominence_score_max = result_df['prominence_sf'].max()
        return result_df
    
    def __setup_required_nlp_resources(self, lang):
        """Loads and initialises all language and nlp resources required by the tagger based on the given language

        Args:
            lang (string): the ISO code for the language of the input stories (e.g. 'nl' or 'en'). Currently only 'nl' and 'en' are supported
        """
        if lang == constants.NL:
            self.stopwords = constants.NL_STOPWORDS_FILE.read_text(encoding="utf-8").split("\n")
        else:
            self.stopwords = constants.EN_STOPWORDS_FILE.read_text(encoding="utf-8").split("\n")

        self.stopwords = [item for item in self.stopwords if len(item) > 0]
