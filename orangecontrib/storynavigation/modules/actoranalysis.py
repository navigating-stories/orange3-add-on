"""Modules required for Actor Analysis widget in Story Navigator.
"""

import sys
import os
import pandas as pd
from operator import itemgetter
import spacy
import storynavigation.modules.constants as constants
import re
from spacy import displacy
import string
from nltk.tokenize import RegexpTokenizer
from thefuzz import fuzz
from statistics import median
from bs4 import BeautifulSoup

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    import importlib.resources as importlib_resources


class ActorTagger:
    """Class to perform NLP analysis of actors in textual stories
    For the storynavigator Orange3 add-on:
    https://pypi.org/project/storynavigator/0.0.7/
    """

    PKG = importlib_resources.files(constants.MAIN_PACKAGE)
    NL_STOPWORDS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_STOPWORDS_FILENAME
    )
    NL_PRONOUNS_FILE = (
        PKG / constants.RESOURCES_SUBPACKAGE / constants.NL_PRONOUNS_FILENAME
    )

    def __init__(self, model):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        pr = self.NL_PRONOUNS_FILE.read_text(encoding="utf-8")
        self.pronouns = pr
        self.stopwords = s
        self.html_result = ""

        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.active_agency_scores = {}
        self.passive_agency_scores = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        self.nlp = self.__load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}
        self.sentence_nlp_models = []

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0

    @classmethod
    def __load_spacy_pipeline(self, name):
        """Check if the spacy language pipeline was downloaded and load it.
        Downloads the language pipeline if not available.

        Args:
            name (string): Name of the spacy language.

        Returns:
            spacy.language.Language: The spacy language pipeline
        """
        if spacy.util.is_package(name):
            nlp = spacy.load(name)
        else:
            os.system(f"spacy download {name}")
            nlp = spacy.load(name)
            nlp.add_pipe("merge_noun_chunks")
            nlp.add_pipe("merge_entities")
            nlp.add_pipe("sentencizer")
        return nlp

    def __preprocess_text(self, text):
        """Preprocesses story text. A lot of stories in the Corona in de stad dataset
        have sentences with no period at the end followed immediately by newline characters.
        This function processes these and other issues to make the resulting text suitable for
        further NLP analysis (e.g. postagging and ner).

        Args:
            text (string): Story text

        Returns:
            list: List of processed string sentences in story text
        """
        # find all regex matches for a newline character immediately followed by uppercase letter
        match_indices = []
        for i in re.finditer("\n[A-Z]", text):
            startindex = i.start()
            match_indices.append(startindex + 1)
        match_indices.append(None)
        # split the text into clauses (based on regex matches) - clauses can be single or multiple sentences
        clauses = [
            text[match_indices[i] : match_indices[i + 1]]
            for i in range(0, len(match_indices) - 1)
        ]
        # clean clauses: remove newlines in the middle of clauses and tokenize them into individual sentences
        cleaned_sentences = []
        for clause in clauses:
            cleaned_clause = clause.replace("\n", " ")
            # tokenize clause into sentences
            sentences = cleaned_clause.split(".")
            for sent in sentences:
                sent_tmp = sent.strip()
                if len(sent_tmp) > 1:
                    if sent_tmp[len(sent_tmp) - 1] != ".":
                        sent_tmp += "."  # add a period to end of sentence (if there is not one already)
                    cleaned_sentences.append(sent_tmp)

        return cleaned_sentences

    def nertag_text(self, text, per, loc, product, date, nlp):
        """Preprocesses story text. A lot of stories in the Corona in de stad dataset
        have sentences with no period at the end followed immediately by newline characters.
        This function processes these and other issues to make the resulting text suitable for
        further NLP analysis (e.g. postagging and ner).

        Args:
            text (string): Story text
            per (boolean): whether person tokens should be tagged
            loc (boolean): whether location tokens should be tagged
            product (boolean): whether product tokens should be tagged
            date (boolean): whether date tokens should be tagged
            nlp (spacy.language.Language): Spacy Language object for NLP parsing

        Returns:
            string: HTML string representation of NER tagged text
        """

        sentences = self.__preprocess_text(text)
        html = ""

        ner_tags = []
        if per:
            ner_tags.append("PERSON")
        if loc:
            ner_tags.append("LOC")
            ner_tags.append("GPE")
            ner_tags.append("NORP")
            ner_tags.append("FAC")
            ner_tags.append("ORG")
            ner_tags.append("EVENT")
        if product:
            ner_tags.append("PRODUCT")
            ner_tags.append("WORK_OF_ART")
        if date:
            ner_tags.append("DATE")
            ner_tags.append("TIME")

        options = {"ents": ner_tags, "colors": {}}

        # NLP tag each sentence and render the result using displacy and build up the HTML output
        for sent in sentences:
            tagged_sentence = nlp(sent)
            html += displacy.render(tagged_sentence, style="ent", options=options)

        return html

    def __update_postagging_metrics(
        self, tagtext, selected_prominence_metric, prominence_score_min, token
    ):
        """After pos-tagging a particular token, this method is executed to calculate the word prominence score
        for the given token and to check whether this score is above the threshold of the user-specified
        minimum word prominence score to display.

        Args:
            tagtext (string): the string representation of the input token from the story text
            selected_prominence_metric: the selected metric by which to calculate the word prominence score

        Returns:
            boolean: True if the word prominence score of the input token is greater or equal to the
            current minimum threshold for the word prominence score specified by the user. False otherwise.
        """

        # This needs to move to Action Analysis module
        vb = self.__find_verb_ancestor(token)
        if vb is not None:
            if tagtext in self.noun_action_dict:
                self.noun_action_dict[tagtext].append(vb.text)
            else:
                self.noun_action_dict[tagtext] = []
        # -----------------------------------------------#

        p_score = self.__calculate_prominence_score(tagtext, selected_prominence_metric)
        self.word_prominence_scores[tagtext] = p_score

        if p_score >= prominence_score_min:
            return True
        else:
            return False

    def __calculate_pretagging_metrics(self, sentences):
        """Before pos-tagging commences, this method is executed to calculate some basic story metrics
        including word count (with and without stopwords) and sentence count.

        Args:
            sentences (list): list of string sentences from the story
        """

        self.sentence_count = len(sentences)
        for sentence in sentences:
            words = sentence.split()
            tokens = []
            for word in words:
                if len(word) > 1:
                    if word[len(word) - 1] in string.punctuation:
                        tokens.append(word[: len(word) - 1].lower().strip())
                    else:
                        tokens.append(word.lower().strip())

            self.word_count += len(tokens)

            if len(self.stopwords) > 0:
                for token in tokens:
                    if token not in self.stopwords:
                        self.word_count_nostops += 1
            else:
                self.word_count_nostops = self.word_count

    def __is_subject(self, tag):
        """Checks whether a given pos-tagged token is a subject of its sentence or not

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            (boolean, string): (True if the given token is a subject of its sentence - False otherwise, the POS tag type of the token)
        """

        if tag[3].lower() in ["nsubj", "nsubj:pass", "csubj"] and tag[1] in [
            "PRON",
            "NOUN",
            "PROPN",
        ]:
            if tag[3].lower() in ["nsubj", "csubj"] and self.__find_verb_ancestor(tag[4]) is not None:
                print('active-noun: ', tag[0].lower(), ' active-verb: ', self.__find_verb_ancestor(tag[4]))
                if tag[0].lower() in self.active_agency_scores:
                    self.active_agency_scores[tag[0].lower()] += 1
                else:
                    self.active_agency_scores[tag[0].lower()] = 1
                if tag[0].lower() not in self.passive_agency_scores:
                    self.passive_agency_scores[tag[0].lower()] = 0
            else:
                print('pass-noun: ', tag[0].lower(), ' pass-verb: ', self.__find_verb_ancestor(tag[4]))
                if tag[0].lower() in self.passive_agency_scores:
                    self.passive_agency_scores[tag[0].lower()] += 1
                else:
                    self.passive_agency_scores[tag[0].lower()] = 1
                if tag[0].lower() not in self.active_agency_scores:
                    self.active_agency_scores[tag[0].lower()] = 0

            if tag[1] == "PRON":
                return True, "PRON"
            elif tag[1] == "NOUN":
                return True, "NOUN"
            else:
                return True, "PROPN"
        else:
            print('pass-noun: ', tag[0].lower(), ' pass-verb: ', self.__find_verb_ancestor(tag[4]))
            if tag[0].lower() in self.passive_agency_scores:
                self.passive_agency_scores[tag[0].lower()] += 1
            else:
                self.passive_agency_scores[tag[0].lower()] = 1
            if tag[0].lower() not in self.active_agency_scores:
                self.active_agency_scores[tag[0].lower()] = 0

        return False, ""

    def __is_pronoun(self, tag):
        """Checks whether a given pos-tagged token is a pronoun or not

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is a pronoun - False otherwise
        """

        if tag[0].lower().strip() == "ik":
            return True
        if tag[0].lower().strip() not in self.stopwords:
            if tag[1] == "PRON":
                if "|" in tag[2]:
                    tmp_tags = tag[2].split("|")
                    if (tmp_tags[1] == "pers" and tmp_tags[2] == "pron") or (
                        tag[0].lower().strip() == "ik"
                    ):
                        return True
        return False

    def __is_noun_but_not_pronoun(self, tag):
        """Checks whether a given pos-tagged token is a non-pronoun noun (or not)

        Args:
            tag (tuple): a tuple with 4 components:
                        1) text: the text of the given token
                        2) pos_: the coarse-grained POS tag of token (string)
                        3) tag_: the fine-grained POS tag of token (string)
                        4) dep_: the syntactic linguistic dependency relation of the token (string)

        Returns:
            boolean: True if the given token is a non-pronoun noun - False otherwise
        """

        if (not self.__is_pronoun(tag)) and (tag[1] in ["NOUN", "PROPN"]):
            return True
        else:
            return False

    def postag_text(
        self, text, nouns, subjs, selected_prominence_metric, prominence_score_min
    ):
        """POS-tags story text and returns HTML string which encodes the the tagged text, ready for rendering in the UI

        Args:
            text (string): Story text
            nouns (boolean): whether noun tokens should be tagged
            subjs (boolean): whether subject tokens should be tagged
            selected_prominence_metric: the selected metric by which to calculate the word prominence score

        Returns:
            string: HTML string representation of POS tagged text
        """
        sentences = self.__preprocess_text(text)
        self.__calculate_pretagging_metrics(sentences)

        # pos tags that the user wants to highlight
        pos_tags = []

        # add pos tags to highlight according to whether the user has selected them or not
        # if vbz:
        #     pos_tags.append("VERB")
        # if adj:
        #     pos_tags.append("ADJ")
        #     pos_tags.append("ADV")
        if nouns:
            pos_tags.append("NOUN")
            pos_tags.append("PRON")
            pos_tags.append("PROPN")
            pos_tags.append("NSP")
            pos_tags.append("NSNP")
        if subjs:
            pos_tags.append("SUBJ")
            pos_tags.append("SP")
            pos_tags.append("SNP")

        # output of this function
        html = ""

        # generate and store nlp tagged models for each sentence
        if self.sentence_nlp_models is None or len(self.sentence_nlp_models) == 0:
            print("got here!")
            # sentence_nlp_models = []
            for sentence in sentences:
                tagged_sentence = self.nlp(sentence)
                self.sentence_nlp_models.append(tagged_sentence)

            self.__calculate_word_type_count(sentences, self.sentence_nlp_models)

        # loop through model to filter out those words that need to be tagged (based on user selection and prominence score)
        for sentence, tagged_sentence in zip(sentences, self.sentence_nlp_models):
            first_word_in_sent = sentence.split()[0].lower().strip()
            tags = []
            tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")
            spans = list(tokenizer.span_tokenize(sentence))

            for token in tagged_sentence:
                tags.append((token.text, token.pos_, token.tag_, token.dep_, token))

            ents = []
            for tag, span in zip(tags, spans):
                normalised_token, is_valid_token = self.__is_valid_token(tag)
                if is_valid_token:
                    is_subj, subj_type = self.__is_subject(tag)
                    if is_subj:
                        p_score_greater_than_min = self.__update_postagging_metrics(
                            tag[0].lower().strip(),
                            selected_prominence_metric,
                            prominence_score_min,
                            token,
                        )
                        if p_score_greater_than_min:
                            if self.__is_pronoun(tag):
                                ents.append(
                                    {"start": span[0], "end": span[1], "label": "SP"}
                                )
                            else:
                                ents.append(
                                    {"start": span[0], "end": span[1], "label": "SNP"}
                                )
                    else:
                        if self.__is_pronoun(tag):
                            ents.append(
                                {"start": span[0], "end": span[1], "label": "NSP"}
                            )
                        elif self.__is_noun_but_not_pronoun(tag):
                            ents.append(
                                {"start": span[0], "end": span[1], "label": "NSNP"}
                            )

            if first_word_in_sent in self.pronouns:
                p_score_greater_than_min = self.__update_postagging_metrics(
                    first_word_in_sent,
                    selected_prominence_metric,
                    prominence_score_min,
                    token
                )

                if p_score_greater_than_min:
                    ents.append(
                        {"start": 0, "end": len(first_word_in_sent), "label": "SP"}
                    )

                print('pass-noun: ', first_word_in_sent, ' pass-verb: None')
                if first_word_in_sent in self.passive_agency_scores:
                    self.passive_agency_scores[first_word_in_sent] += 1
                else:
                    self.passive_agency_scores[first_word_in_sent] = 1
                if first_word_in_sent not in self.active_agency_scores:
                    self.active_agency_scores[first_word_in_sent] = 0

            # specify sentences and filtered entities to tag / highlight
            doc = {"text": sentence, "ents": ents}

            # specify colors for highlighting each entity type
            colors = {}
            if nouns:
                colors["NSP"] = constants.NONSUBJECT_PRONOUN_HIGHLIGHT_COLOR
                colors["NSNP"] = constants.NONSUBJECT_NONPRONOUN_HIGHLIGHT_COLOR
            if subjs:
                colors["SP"] = constants.SUBJECT_PRONOUN_HIGHLIGHT_COLOR
                colors["SNP"] = constants.SUBJECT_NONPRONOUN_HIGHLIGHT_COLOR

            self.agent_prominence_score_max = self.__get_max_prominence_score()
            # collect the above config params together
            options = {"ents": pos_tags, "colors": colors}
            # give all the params to displacy to generate HTML code of the text with highlighted tags
            html += displacy.render(doc, style="ent", options=options, manual=True)

        self.html_result = html
        # return html
        return self.__remove_span_tags(html)

    def __get_normalized_token(self, token):
        """cleans punctuation from token and verifies length is more than one character

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string: cleaned token text
        """

        if type(token) == spacy.tokens.token.Token:
            normalised_token = token.text.lower().strip()
        else:
            normalised_token = token[0].lower().strip()
        if len(normalised_token) > 1:
            if normalised_token[len(normalised_token) - 1] in string.punctuation:
                normalised_token = normalised_token[: len(normalised_token) - 1].strip()
            if normalised_token[0] in string.punctuation:
                normalised_token = normalised_token[1:].strip()

        return normalised_token

    def __is_valid_token(self, token):
        """Verifies if token is valid word

        Args:
            token (spacy.tokens.token.Token): tagged Token | tuple : 4 components - (text, tag, fine-grained tag, dependency)

        Returns:
            string, boolean : cleaned token text, True if the input token is a valid word, False otherwise
        """

        word = self.__get_normalized_token(token)
        return word, (word not in self.stopwords) and len(word) > 1

    def __calculate_word_type_count(self, sents, sent_models):
        """Calculates the frequency of mentions for each word in the story:
            - Number of times word appears as a subject of a sentence
            - Number of times the word appears period 

        Args:
            sents (list): list of all sentences (strings) from the input story
            sent_models (list): list of (spacy.tokens.doc.Doc) objects - one for each element of 'sents'
        """

        for sent_model in sent_models:
            for token in sent_model:
                normalised_token, is_valid_token = self.__is_valid_token(token)
                tag = (token.text, token.pos_, token.tag_, token.dep_, token)
                if is_valid_token:
                    is_subj, subj_type = self.__is_subject(tag)
                    if is_subj:
                        print('test: ', token.text.lower().strip())
                        if token.text.lower().strip() in self.num_occurences_as_subject:
                            self.num_occurences_as_subject[
                                token.text.lower().strip()
                            ] += 1
                        else:
                            self.num_occurences_as_subject[
                                token.text.lower().strip()
                            ] = 1
                    else:
                        if self.__is_pronoun(tag) or self.__is_noun_but_not_pronoun(
                            tag
                        ):
                            if token.text.lower().strip() in self.num_occurences:
                                self.num_occurences[token.text.lower().strip()] += 1
                            else:
                                self.num_occurences[token.text.lower().strip()] = 1

        for sent in sents:
            word = sent.split()[0].lower().strip()
            if word in self.pronouns:
                if word in self.num_occurences_as_subject:
                    self.num_occurences_as_subject[word] += 1
                else:
                    self.num_occurences_as_subject[word] = 1

        print()
        print("subj dict now: ", self.num_occurences_as_subject)
        print()

        print()
        print("active agency dict: ", self.active_agency_scores)
        print()

        print()
        print("passive agency dict: ", self.passive_agency_scores)
        print()

    def __find_closest_match(self, word, dictionary):
        """Uses fuzzy string matching to find the closest match in a given dictionary (dict) for an input string

        Args:
            word (string): input word
            dictionary (dict): keys are words, values are numbers (mention frequency)

        Returns:
            word, boolean: string of the best match, True if a match is found above the threshold, False otherwise
        """
        highest_score = -10
        word_with_highest_score = word
        for item in dictionary:
            similarity_score = fuzz.ratio(item, word)
            if similarity_score > highest_score:
                highest_score = similarity_score
                word_with_highest_score = item

        if highest_score > 80:
            return word_with_highest_score, True
        else:
            return word, False

    def __calculate_prominence_score(self, word, selected_prominence_metric):
        """Calculates the promience score for a given word in the story, uses two simple metrics (work in progress and more to follow):
        - Subject frequency : number of times the word appears as a subject of a sentence in the story divided by the number of words in the story
        - Subject frequency (normalized) : number of times the word appears as a subject of a sentence in the story divided by the median subject frequency of a word in the story

        Args:
            word (string): input word
            selected_prominence_metric (string): name of the metric to use

        Returns:
            score: the prominence score of the input word within the story using the specified metric
        """
        score = 0
        # match spacy-tagged token text to the existing dictionary of words in num_occurrences_as_subject
        closest_match_word, successful_match = self.__find_closest_match(
            word, self.num_occurences_as_subject
        )

        if selected_prominence_metric == "Subject frequency (normalized)":
            score = (
                self.num_occurences_as_subject[closest_match_word] / median(list(self.num_occurences_as_subject.values()))
            )
        elif selected_prominence_metric == "Subject frequency":
            score = self.num_occurences_as_subject[closest_match_word] / self.word_count_nostops

        return score

    # Function to recursively traverse ancestors
    def __find_verb_ancestor(self, token):
        """Finds the main verb associated with a token (mostly nouns) in a sentence

        Args:
            token (spacy.tokens.token.Token): input token

        Returns:
            verb: the verb text if any, otherwise None
        """
        # Check if the token is a verb
        if token.pos_ == "VERB":
            return token

        # Traverse the token's ancestors recursively
        for ancestor in token.ancestors:
            # Recursive call to find the verb ancestor
            verb_ancestor = self.__find_verb_ancestor(ancestor)
            if verb_ancestor:
                return verb_ancestor

        # If no verb ancestor found, return None
        return None

    def __get_max_prominence_score(self):
        """Finds the word in the story with the highest prominence score and returns this score

        Returns:
            highest_score: the score of the word with highest prominence score in the story
        """

        highest_score = 0
        for item in self.word_prominence_scores:
            if self.word_prominence_scores[item] > highest_score:
                highest_score = self.word_prominence_scores[item]
        return highest_score

    def __calculate_agency(self, word):
        """Calculates the agency of a given word (noun) in the story using a custom metric

        Args:
            word (string): input word

        Returns:
            agency_score: the agency score for the input word in the given story
        """
        active_freq = 0
        passive_freq = 0
        for item in self.active_agency_scores:
            active_freq += self.active_agency_scores[item]
        for item in self.passive_agency_scores:
            passive_freq += self.passive_agency_scores[item]

        if active_freq > 0 and passive_freq > 0:
            return (self.active_agency_scores[word]/active_freq) - (self.passive_agency_scores[word]/passive_freq)
        elif active_freq == 0 and passive_freq > 0:
            return 0 - (self.passive_agency_scores[word]/passive_freq)
        elif active_freq > 0 and passive_freq == 0:
            return (self.active_agency_scores[word]/active_freq)
        else:
            return 0
        
    def calculate_metrics_freq_table(self):
        """Prepares data table for piping to Output variable of widget: frequency of words in story

        Returns:
            data table (pandas dataframe)
        """

        rows = []
        n = 10
        res = dict(
            sorted(
                self.num_occurences.items(), key=itemgetter(1), reverse=True
            )
        )

        words = list(res.keys())

        for word in words:
            rows.append(
                    [
                        word,
                        self.num_occurences[word]
                    ]
                )
            
        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.FREQ_TABLE_HEADER)
    
    def calculate_metrics_subjfreq_table(self):
        """Prepares data table for piping to Output variable of widget: frequencies as subjects of words in story

        Returns:
            data table (pandas dataframe)
        """
        rows = []
        n = 10
        res = dict(
            sorted(
                self.num_occurences_as_subject.items(), key=itemgetter(1), reverse=True
            )
        )

        words = list(res.keys())

        for word in words:
            rows.append(
                    [
                        word,
                        self.num_occurences_as_subject[word]
                    ]
                )
    
        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.SUBFREQ_TABLE_HEADER)


    def calculate_metrics_agency_table(self):
        """Prepares data table for piping to Output variable of widget: agency scores of words in story

        Returns:
            data table (pandas dataframe)
        """
        rows = []
        n = 10
        words = set()
        for item in list(self.num_occurences_as_subject.keys()):
            words.add(item)
        for item2 in list(self.num_occurences.keys()):
            words.add(item2)

        words = list(words)

        for word in words:
            agency = self.__calculate_agency(word)
            rows.append(
                    [
                        word,
                        agency
                    ]
                )

        rows.sort(key=lambda x: x[1])

        return pd.DataFrame(rows[-n:], columns=constants.AGENCY_TABLE_HEADER)


    def generate_noun_action_table(self):
        """Prepares data table for piping to Output variable of widget: 
        - list of actors in story (1st column), 
        - comma-separated list of verbs that each actor is involved in (2nd column)

        Returns:
            data table (pandas dataframe)
        """
        n = 10
        res = dict(
            sorted(
                self.word_prominence_scores.items(), key=itemgetter(1), reverse=True
            )[:n]
        )

        names = list(res.keys())

        rows = []
        for item in self.noun_action_dict:
            if len(self.noun_action_dict[item]) > 0 and (item in names):
                curr_row = []
                curr_row.append(item)
                curr_row.append(", ".join(list(set(self.noun_action_dict[item]))))
                rows.append(curr_row)

        return pd.DataFrame(rows, columns=["actor", "actions"])

    @staticmethod
    def sort_tuple(tup):
        """Sorts a tuple of numerical items in ascending order

        Args:
            tup (Tuple) : n-dimensional tuple

        Returns:
            tuple with each element sorted in ascending order
        """
        lst = len(tup)
        for i in range(0, lst):
            for j in range(0, lst - i - 1):
                if tup[j][1] > tup[j + 1][1]:
                    temp = tup[j]
                    tup[j] = tup[j + 1]
                    tup[j + 1] = temp
        return tup
    
    def __remove_span_tags(self, html_string):
        soup = BeautifulSoup(html_string, 'html.parser')
        
        # Remove all <span> tags
        for span_tag in soup.find_all('span'):
            span_tag.decompose()

        return str(soup)


class ActorMetricCalculator:
    """Unused class / code so far...
    """
    def __init__(self, text, listofwords):
        s = self.NL_STOPWORDS_FILE.read_text(encoding="utf-8")
        self.stopwords = s
        self.html_result = ""

        # Other counts initialisation
        self.word_count = 0
        self.word_count_nostops = 0
        self.sentence_count = 0
        self.sentence_count_per_word = {}
        self.num_occurences = {}
        self.num_occurences_as_subject = {}
        self.noun_action_dict = {}

        # self.nlp = self.__load_spacy_pipeline(model)

        # Scoring related to agent prominence score
        self.agent_prominence_score_max = 0.0
        self.agent_prominence_score_min = 0.0

        # Index of word prominence scores for each word in story
        self.word_prominence_scores = {}

        # POS counts initialisation
        self.noun_count = 0
        self.verb_count = 0
        self.adjective_count = 0
