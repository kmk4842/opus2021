''' At the beginning, we import all the necessary libraries used in the rest of the script. '''

from typing import List, Set, Dict, Tuple, Optional, Any  # this is for type hints in fuctions
from jinja2.compiler import generate
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath
import pickle
import numpy as np
import pandas as pd
import os
from collections import Counter
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename="log.txt")
from gensim.models import TfidfModel

'''Paths'''
gdict_eng_path = str('D:\WordNetPython\Makary - Python\gdict_eng.dct')  # gensim dictionary for English corpus
gcorpus_eng_path = str('D:\WordNetPython\Makary - Python\corpus_eng_mm.mm')  # gensim English corpus mm
gdict_fr_path = str('D:\WordNetPython\Makary - Python\gdict_fr.dct')  # gensim dictionary for French corpus
gcorpus_fr_path = str('D:\WordNetPython\Makary - Python\corpus_fr_mm.mm')  # gensim French corpus mm

'''Analysis options'''
corpus_type_TFIDF = 1  # set to 1 for TFIDF frequencies, or 0 for simple counts
generate_french_lists = 1  # if you need to re-run list generation, change set to 1, if you want to load it set to 0

''' Sections ON/OFF - be careful here, best to run it as is, without loading, there is some problem at unpickling'''
section_dictionary = 1  # core analysis to create wordnet objects
section_saving_results = 1  # saving analysis results
section_EnglishCorpus_analysis = 1  # comparing original vs. wordnet lists
statistics_old_vs_new = 1  # produce classification statistics between original and wordnet lexicons in English
section_pickling_save = 1  # saving English corpus analysis after first run
section_pickling_load = 0  # NOTE: The pickling_save process must have taken place at least once for this to work.
section_FrenchCorpus_analysis = 1  # Translation to French and comparative statistics

print("Running the WordNet translation script using the following switches: Dictionary %d, Saving %d, "
      "Pickle Save %d and Load %d, Analysis for English %d and French %d."
      % (section_dictionary, section_saving_results, section_pickling_save, section_pickling_load,
         section_EnglishCorpus_analysis, section_FrenchCorpus_analysis))

if corpus_type_TFIDF:
    print("Analysis results calculated for TFIDF frequencies.")
else:
    print("Analysis results calculated for counts (not TFIDF).")

''' Necessary conditions '''
if section_saving_results == 1:
    section_dictionary = 1
if section_pickling_save == 1:
    section_EnglishCorpus_analysis = 1
if section_EnglishCorpus_analysis == 1:
    statistics_old_vs_new = 1


class WLN:
    # General class for mapping wordlists to synsets and scoring them.
    # once initiated it generates a WLN object with a range of sets and dictionaries
    # NOTE: it requires a gensim dictionary to work (specified in the first line of __init__
    def __init__(self, list_name):
        self.name = list_name
        self.gensim_dict_filename = 'gdict_eng.dct'  # we need a gensim dictionary of lemmas in corpus prepared earlier
        self.gdictionary = Dictionary.load(self.gensim_dict_filename)  # gensim dictionary object

        self.synsets_set = set()  # a set containing all synsets related to words in the given list
        self.no_synsets_set = set()  # set containing all words for which no synsets were found using the "no_synsets_for_this_word" function
        self.LEM_wordnet_set = set()  # set containing all words selected with the "lemmas_from_synsets" function

        self.list_to_LEM = {}  # a dictionary containing all the words from a given list and the corresponding lemmas
        self.LEM_to_synsets = {}  # a dictionary containing all (single) values analogous to the "list_to_LEM" dictionary and all corresponding synsets for different parts of speech
        self.LEM_to_best_synset = {}  # a dictionary containing all (single) values analogous to the "list_to_LEM" dictionary and the corresponding best synsets for each part of speech
        # self.WORD_to_SCORE = {}  # a dictionary containing all words from the main dictionary and the corresponding scores given by the "scoring_script" function

        self.dictionary_creation(
            list_name)  # function creating "list_to_LEM" and "LEM_to_synsets" dictionaries for each of the word lists
        # self.WORD_to_SCORE_creation(
        #   self.gensim_dict_filename)  # function creating "WORD_to_SCORE" dictionary based on "TOKEN_to_SCORE" values and "WORD_to_TOKEN" keys
        self.no_synsets_for_this_word(self.LEM_to_synsets)  # set creation function for set "no_synsets_set"
        self.synset_to_score = self.scoring_script()  # scoring function, counting the number of occurrences of words in the corpus on the basis of data obtained from the main dictionary
        self.LEM_to_best_synset = self.best_synset(
            self.LEM_to_synsets)  # the function creating a dictionary "LEM_to_best_synset" based on the number of occurrences of each synset in the corpus for each part of speech
        self.LEM_wordnet_set = self.lemmas_from_synsets()  # the function that creates a set "LEM_wordnet_set" for each synset from "LEM_to_best_synset" finds the corresponding lemmas
        self.LEM_revised_list = self.revise_list()

    def __getstate__(self):
        state = self.__dict__.copy()  # copy the entire class dictionary
        del state['synsets_set']  # remove items causing pickling problems because they contain synset objects
        del state['LEM_to_synsets']
        del state['LEM_to_best_synset']
        del state['synset_to_score']  # remove this, it's still accessible via function: scoring_script
        synsets_set_text = set()  # convert this set to synset names, not synset objects
        for s in self.synsets_set:
            synsets_set_text.add(s.name())
        state['synsets_set'] = synsets_set_text
        LEM_to_best_synset_text = {}  # convert this dict to list of synset names, not synset objects
        for key, value in self.LEM_to_best_synset.items():
            LEM_to_best_synset_text[key] = [s.name() for s in self.LEM_to_best_synset[key]]
        state['LEM_to_best_synset'] = LEM_to_best_synset_text
        LEM_to_synsets_text = {}  # convert this dict to list of synset names, not synset objects
        for key, value in self.LEM_to_synsets.items():
            LEM_to_synsets_text[key] = [s.name() for s in self.LEM_to_synsets[key]]
        state['LEM_to_synsets'] = LEM_to_synsets_text
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # restore previously deleted attributes (in getstate) if possible/useful, e.g. synsets from text or files from filenames
        # self.whatever = what needs to be restored
        synsets_set = set()  # restore synset objects into synsets_set
        for s in self.synsets_set:
            synsets_set.add(wn.synset(s))
        self.synsets_set = synsets_set
        LEM_to_best_synset_ss = {}  # restore synset objects from names
        for key, value in self.LEM_to_best_synset.items():
            LEM_to_best_synset_ss[key] = [wn.synset(name) for name in self.LEM_to_synsets[key]]
        self.LEM_to_best_synset = LEM_to_best_synset_ss
        LEM_to_synsets_ss = {}  # restore synset objects from names
        for key, value in self.LEM_to_synsets.items():
            LEM_to_synsets_ss[key] = [wn.synset(name) for name in self.LEM_to_synsets[key]]
        self.LEM_to_synsets = LEM_to_synsets_ss


    def dictionary_creation(self,
                            list_name):  # function creating "list_to_LEM" and "LEM_to_synsets" dictionaries for each of the word lists
        self.list_to_LEM_creation(list_name)  # function creating "list_to_LEM" dictionary
        self.LEM_to_synsets_creation(list_name)  # function creating "LEM_to_synsets" dictionary

    def list_to_LEM_creation(self, list_name):  # function creating "list_to_LEM" dictionary
        LEM = WordNetLemmatizer()
        with open(list_name) as f:
            for line in f:  # data imported from a given word list
                line = line.strip()  # trimming ends and unnecessary characters
                line = line.lower()  # small letters in order to standardize words starting with an uppercase letter
                self.list_to_LEM[line.upper()] = LEM.lemmatize(
                    line)  # capital letters for better standardization of the next steps
        return self.list_to_LEM

    def LEM_to_synsets_creation(self, list_name):  # function creating "LEM_to_synsets" dictionary
        LEM = WordNetLemmatizer()
        with open(list_name) as f:  # data imported from a given word list
            for line in f:
                line = line.strip()  # trimming ends and unnecessary characters
                line = line.lower()  # lower-case in order to standardize words starting with an uppercase letter
                #  get a list of synsets for each word in line, apply some manual corrections on the go
                self.LEM_to_synsets[LEM.lemmatize(line)] = self.synsets_correction(line)
                #  create a set of synsets to remove duplicates
                for SYNSET in self.LEM_to_synsets[LEM.lemmatize(line)]:
                    self.synsets_set.add(SYNSET)
        return self.LEM_to_synsets

    def synsets_correction(self, line):
        # manual corrections to synsets during mapping
        # ADD MORE CORRECTIONS HERE AS NEEDED
        # the following synsets are excluded from mapping:
        synsets_set_Blacklist = set(
            [wn.synset('best.n.03'), wn.synset('sue.n.01'), wn.synset('pm.n.03'), wn.synset('le.n.01'),
             wn.synset('low.v.01'), wn.synset('low.n.02'), wn.synset('veto.v.01'), wn.synset('unaccented.s.02'),
             wn.synset('humble.s.01'), wn.synset('future.s.03'), wn.synset('beating.n.02'),
             wn.synset('wildness.n.01'), wn.synset('ambitious.s.02'), wn.synset('commodity.n.01'),
             wn.synset('meter.n.03'), wn.synset('tone.v.05'), wn.synset('bettor.n.01'), wn.synset('originate.v.01'),
             wn.synset('unsealed.a.01'), wn.synset('bolster.n.01'), wn.synset('inhalation.n.01'),
             wn.synset('stable.n.01'), wn.synset('business.n.01'), wn.synset('conclusion.n.08'), wn.synset('base.s.04'),
             wn.synset('argument.n.01'), wn.synset('breed.n.01'), wn.synset('gloomy.s.02'),
             wn.synset('exploitation.n.01'), wn.synset('conclude.v.04'), wn.synset('dwell.v.02'),
             wn.synset('underperform.v.02'), wn.synset('damage.v.02'), wn.synset('lag.v.04')])  # wn.synset('')
        syns_in = wn.synsets(line)
        syns_out = []
        for x in syns_in:
            if "_" in x.name():
                continue  # skip two-word synsets, we only take single lemmas
            if x in synsets_set_Blacklist:
                continue  # skip blacklisted synsets
            syns_out.append(x)
        return syns_out

    '''
    # function removed to save on processing time. Use the word_to_score function isntead
    def WORD_to_SCORE_creation(self,
                               gensim_dict_filename):  # function creating "WORD_to_SCORE" dictionary based on "TOKEN_to_SCORE" values and "WORD_to_TOKEN" keys
        ANG_DICT = Dictionary.load(
            gensim_dict_filename)  # main dictionary containing words obtained from the corpus of financial statements
        WORD_to_TOKEN = ANG_DICT.token2id
        # TOKEN_to_SCORE_dfs = ANG_DICT.dfs
        TOKEN_to_SCORE = ANG_DICT.cfs
        for key, value in WORD_to_TOKEN.items():  # keys taken from the "WORD_to_TOKEN" dictionary and values from the "TOKEN_to_SCORE" dictionary
            for key2, value2 in TOKEN_to_SCORE.items():
                if str(value) == str(key2):
                    self.WORD_to_SCORE[key] = value2
        return (self.WORD_to_SCORE)
    '''

    def word_to_score(self, lemma: str):
        score: int = 0
        try:
            score = self.gdictionary.cfs[self.gdictionary.token2id[lemma]]
        except:
            score = 0  # if lemma not in dictionary
        return score

    def no_synsets_for_this_word(self, LEM_to_synsets):  # set creation function for set "no_synsets_set"
        for key, value in LEM_to_synsets.items():
            if value == []:
                if wn.morphy(key) is not None:  # trying to find another synset
                    print("FOUND A NEW SYNSET:", wn.morphy(key))
                self.no_synsets_set.add(key)
        return (self.no_synsets_set)

    def scoring_script(
            self):  # scoring function, counting the number of occurrences of words in the corpus on the basis of data obtained from the main dictionary
        synset_to_hits = {}  # local variable that counts the occurrences
        for s in self.synsets_set:
            hit = 0  # reset
            for lemma in s.lemma_names():
                try:
                    hit = hit + self.word_to_score(lemma)
                except:
                    continue
            synset_to_hits[s] = hit  # dictionary record - number of occurrences for each synset
        return (synset_to_hits)

    def best_synset(self,
                    LEM_to_synsets):  # the function creating a dictionary "LEM_to_best_synset" based on the number of occurrences of each synset in the corpus for each part of speech
        t = tuple()  # local empty tuple
        min_hits = 10  # a local variable that specifies the minimum number of occurrences in the corpus
        for key, values in LEM_to_synsets.items():  # creating empty lists for each of the Wordnet parts of speech
            list_n = []  # empty noun list
            list_v = []  # empty verb list
            list_a = []  # empty adjective list
            list_s = []  # empty adjective sattelite list
            list_r = []  # empty adverb list
            list_best = []  # empty list of best synsets
            for i in values:  # loop sorting synsets by parts of speech and adding them to the corresponding list
                t = (self.synset_to_score.get(i), i)  # loading scoring information into an empty tuple
                li = []
                if i.pos() == 'v':
                    li = list_v
                    li.append(t)
                elif i.pos() == 'n':
                    li = list_n
                    li.append(t)
                elif i.pos() == 'a':
                    li = list_a
                    li.append(t)
                elif i.pos() == 's':
                    li = list_s
                    li.append(t)
                elif i.pos() == 'r':
                    li = list_r
                    li.append(t)
                else:
                    print(i, " - ERROR")  # error in case of incorrect data downloaded from WORDNET
                li.sort(reverse=True)  # sorting the list by score
            (hits,
             s) = t  # checking if the number of occurrences meets the minimum specified earlier by means of the "min_hits" variable
            if list_v:  # for each list separately ofc.
                if hits >= min_hits:  # synsets meeting the condition are placed on the "list_best" list for the corresponding part of speech
                    list_best.append(list_v[
                                         0])  # from each list for a given part of speech, the synset with the highest score is taken
            if list_n:  # for this reason, prior sorting was introduced
                if hits >= min_hits:
                    list_best.append(list_n[0])
            if list_a:
                if hits >= min_hits:
                    list_best.append(list_a[0])
            if list_s:
                if hits >= min_hits:
                    list_best.append(list_s[0])
            if list_r:
                if hits >= min_hits:
                    list_best.append(list_r[0])
            if list_best:
                hb, sb = zip(*list_best)
                # print("sb : " + str(sb))
                self.LEM_to_best_synset[key] = list(sb)
            else:
                self.LEM_to_best_synset[key] = []
            # print("LIST BEST:", key, " - ", list_best)
        # print('\n', "LEM_to_best_synset", self.LEM_to_best_synset)
        return (self.LEM_to_best_synset)

    def lemmas_from_synsets(self):
        # for each synset from "LEM_to_best_synset" finds the corresponding lemmas
        # remove duplicates and create an output list that can later be revised manually
        for key, value in self.LEM_to_best_synset.items():  # for each synset (if any) for each part of WORDNET speech
            for SYNSET in value:  # we are looking for the lemma from which it was created
                for name in [lemma.name() for lemma in SYNSET.lemmas()]:  # (probably)
                    if not "_" in name:
                        self.LEM_wordnet_set.add(name)
        return (self.LEM_wordnet_set)

    def revise_list(self):
        # Function revises lemmas_from_synsets based on manual verification results coded here
        # it produces a list of lemmas based on the set generated earlier via lemmas_from_synsets()
        LEM_revised_list: List[Any] = []
        blacklist = []  # includes lemmas to be excluded
        whitelist = []  # includes lemmas that need to be included

        # define blacklists for specific word-lists and a general one
        if self.name == "HenryPos2008.csv":
            blacklist = ["acquire", "acquisition", "addition", "adult", "advance", "amend", "arise", "book", "capital",
                         "chance", "climb", "commodity", "follow", "get", "heavy", "increment", "leave", "measure",
                         "meter", "metre", "near", "nearly", "originate", "pass", "pound", "produce", "raise", "read",
                         "register", "render", "repay", "result", "return", "right", "show", "skill", "tone",
                         "virtually", "outstanding"]
        elif self.name == "HenryNeg2008.csv":
            blacklist = \
                ["adventure", "blue", "break", "chance", "correct", "cut", "dampen", "dip", "expend", "go", "hard",
                 "hazard", "land", "little", "minor", "pass", "reduction", "refuse", "reject", "return", "slack",
                 "small", "soften", "speculative", "spend", "going", "set", "take", "use", "short"]
        elif self.name == "positive.csv":  # LM wordlist
            blacklist = ["acquisition", "addition", "advanced", "ahead", "bond", "capital", "chance", "charge", "check",
                         "commodity", "declaration", "democratic", "design", "expert", "fabricate", "fast", "fill",
                         "find", "firm", "follow", "forward", "foundation", "free", "fulfill", "further", "gift",
                         "golden", "hike", "increase", "inhalation", "initiation", "institution", "insure", "intake",
                         "introduction", "leave", "light", "loose", "mad", "make", "manufacture", "model", "note",
                         "observe", "one", "origination", "outstanding", "pad", "particular", "partner", "pet",
                         "preferred", "progression", "ranking", "repercussion", "restoration", "result", "retrieve",
                         "return", "right", "see", "slow", "slowly", "solely", "striking", "substantial", "taking",
                         "technical", "technique", "tone", "transport", "turmoil", "typical", "up", "upheaval", "use",
                         "welfare", "work", "acquire", "invest", "control"]
            # the following lemmas need to be added, because no synsets were found for them
            whitelist = ['proactively', 'exclusivity']
        elif self.name == "negative.csv":  # LM wordlist
            blacklist = ["action", "adjust", "adventure", "aim", "anticipate", "assume", "assumed", "attach", "audit",
                         "back", "bar", "base", "bias", "black", "block", "blue", "bond", "breed", "business", "buy",
                         "call", "care", "careful", "carry", "cast", "chance", "check", "clear", "close",
                         "charge", "commission", "commit", "competition", "compression", "conclude", "conclusion",
                         "conduct", "consider", "consist", "consume", "contract", "core", "corner", "correct",
                         "corrosion", "counter", "deal", "debate", "decrease", "deletion", "depart", "departure",
                         "depreciate", "development", "difference", "discharge", "discount", "discover", "discovery",
                         "down", "draw", "drive", "ease", "easy", "eat", "effect", "effort", "elimination", "end",
                         "ending", "engage", "essence", "essential", "evidence", "exact", "except", "exclude",
                         "exhaust", "expiration", "explosive", "extend", "extended", "extension", "fabrication",
                         "facilitate", "find", "finish", "fleece", "fluid", "focus", "founder", "fracture", "free",
                         "game", "gap", "give", "go", "gross", "hard", "head", "heavy", "hedge", "hold", "important",
                         "indemnification", "indicate", "intensify", "intent", "interest", "interested", "interim",
                         "interview", "job", "keep", "lately", "later", "lay", "lengthy", "lift", "limit", "load",
                         "low", "lower", "manage", "mar", "master", "match", "meantime", "measured", "natural", "niche",
                         "occupy", "off", "offset", "open", "order", "outcome", "oversight", "pain", "pass", "pause",
                         "period", "pit", "place", "point", "pose", "position", "prevent", "price", "privacy",
                         "process", "projection", "propose", "prove", "pull", "purpose", "pursue", "put", "qualify",
                         "ram", "rank", "reach", "receiver", "recent", "recently", "recreational", "red", "reduce",
                         "reduction", "refine", "reiterate", "release", "relief", "remain", "remit", "repeal", "repeat",
                         "research", "result", "resultant", "return", "revaluation", "reveal", "reverse", "review",
                         "right", "rigorous", "risk", "round", "say", "sentence", "set", "settle", "settlement",
                         "shift", "short", "show", "slip", "small", "pan", "speculative", "spend", "spirit", "stake",
                         "stand", "statement", "stay", "stock", "stop", "strive", "subject", "submit", "support",
                         "surcharge", "table", "take", "tell", "terms", "tight", "tighten", "tilt", "track", "transfer",
                         "trip", "try", "unrealized", "use", "venture", "visit", "void", "wait", "waive", "want",
                         "waste", "work", "completion", "line", "form"]
            # the following lemmas need to be added, because no synsets were found for them
            whitelist = ['arrearages', 'unfavorability', 'mischaracterization', 'mislabeling', 'unstabilized',
                         'disallowance', 'misjudgment', 'detract', 'mispricings', 'underperformance', 'divestment',
                         'nonrenewal', 'cyberattacks', 'writedowns', 'aversely', 'unreimbursed', 'underinsured',
                         'irrecoverably', 'underreporting', 'deadweights', 'markdown', 'redefault', 'unconscionably',
                         'misclassification', 'redefaulted', 'adversarial', 'unrecovered', 'protestor', 'detracting',
                         'illiquidity', 'aberrational', 'abusiveness', 'disaffiliation', 'overbuild', 'overbuilds',
                         'egregiously', 'cyberattack', 'spamming', 'uncollectibility', 'protestors', 'cyberbullying',
                         'uncollectable', 'oversupplied', 'overbuilding', 'irreconcilably', 'undercapitalized',
                         'nonproducing', 'undeliverable', 'unsellable', 'markdowns', 'deadweight', 'illiquid',
                         'indefeasibly', 'mispricing', 'deadlocking', 'unreasonableness', 'redefaults', 'renegotiation',
                         'mislabel', 'renegotiations', 'misjudgments', 'misclassifications', 'miscommunication',
                         'cybercriminal',
                         'overbuilt', 'unforseen', 'against', 'disallowances', 'underutilized', 'unaccounted',
                         'undelivered',
                         'nonattainment', 'writeoffs', 'unliquidated', 'contentiously', 'misprice', 'divestments',
                         'cybercriminals', 'underfunded', 'untrusted', 'misclassify', 'mislabels', 'overcapacity',
                         'anticompetitive', 'frustratingly', 'opportunistically', 'underutilization', 'uneconomically',
                         'arrearage', 'writeoff', 'writedown', 'unapproved', 'delinquently', 'unsustainable',
                         'nondisclosure', 'fine'
                         'nonperforming', 'uncollectibles', 'nonrecoverable', 'difficultly', 'chargeoffs', 'bailout',
                         'detracted', 'overcapacities', 'misclassified', 'noncomplying', 'mislabelled', 'restructuring']
        else:
            blacklist = []
            whitelist = []

        for lemma in self.LEM_wordnet_set:
            # check if the lemma appears in the corpus
            try:
                rareword = self.word_to_score(lemma) < 3
            except:
                rareword = True

            if lemma in blacklist:  # drop words from final_list that are on the blacklist
                continue
            elif rareword:  # drop words that appear fewer than 3 times in training corpus
                continue
            elif "_" in lemma:  # drop phrasal and other groups of words
                continue
            else:
                LEM_revised_list.append(lemma)  # append to output if neither blacklist nor < 3 times in data

        LEM_revised_list = LEM_revised_list + whitelist
        LEM_revised_list.sort()
        return LEM_revised_list


### VARIOUS UTILITY FUNCTIONS ###

def get_gensim_ids(list_of_lemmas, gensim_dictionary):
    """
    Generates a set of ids from gensim.dictionary.token2id.
    :param list_of_lemmas: any list of strings that may appear in the dictionary
    :param gensim_dictionary: a gensim.dictionary object
    :return: list of gensim ids
    """
    ang_dict_keys = [k for k in gensim_dictionary.token2id.keys()]
    lemmatizer = WordNetLemmatizer()
    ang_dict_keys_lemmatized = [lemmatizer.lemmatize(k) for k in ang_dict_keys]
    ang_dict_keys_lemmatized_count = {k: ang_dict_keys_lemmatized.count(k) for k in ang_dict_keys_lemmatized}
    # df_testing = pd.DataFrame({'original': ang_dict_keys, 'lemmatized': ang_dict_keys_lemmatized})
    # df_testing.to_csv("./df_testing.csv")
    ids = set()
    for lemma in list_of_lemmas:
        try:
            count = ang_dict_keys_lemmatized_count[lemma]
        except:
            continue
        if count == 1:
            ids.add(gensim_dictionary.token2id.get(ang_dict_keys[ang_dict_keys_lemmatized.index(lemma)]))
        else:
            templist = ang_dict_keys_lemmatized
            baseindex = 0
            for x in range(0, count):
                index = templist.index(lemma)
                baseindex = baseindex + index  # index used for accessing ids in gensim dictionary
                ids.add(gensim_dictionary.token2id.get(ang_dict_keys[baseindex]))
                templist = templist[index+1:len(templist)-1]  # slicing list to search the rest
                baseindex += 1  # this takes care of changing index when list is sliced
    return ids

def max_hits_function(list_of_lemmas):
    # function makes a list of the 30 most common lemmas in corpus for a nice graph to publish
    # WARNING: Function relies on external object WLN_DICT
    max_list = {}
    for lemma in list_of_lemmas:
        max_list[lemma] = WLN_DICT[l].word_to_score(lemma)
    c = Counter(max_list)
    max_hits = c.most_common(30)
    return (max_hits)


def list_diagnostics(list_name, wln_object):
    """
    Function prints diagnostics for original vs. new word-list processed via WordNet
    :param list_name: name of list used for printing
    :param wln_object: the WLN class object representing that list
    :return: the final, new list with counts per lemma
    """
    # function prints diagnostic information of a word list
    wln = wln_object
    old_set = set([lemma.lower() for lemma in wln.list_to_LEM.values()])  # take values, ie. lemmatized original list
    raw_set = wln.LEM_wordnet_set
    revised_set = set(wln.LEM_revised_list)
    print()
    print('***REPORT LIST DIAGNOSTICS for list %s .***' % list_name)
    print('Number of lemmas in new list = %d and original list (lemmatized) = %d .' % (
        len(wln.LEM_revised_list), len(old_set)))
    print('Number of missing synsets: %d.' % len(wln.no_synsets_set))
    print('Missing synsets for the following lemmas: ')
    print('\t' + str(wln.no_synsets_set))
    print('Lemmas common between new and old list counting %d. Before manual revision it was %d.'
          % (len(revised_set.intersection(old_set)), len(raw_set.intersection(old_set))))
    print('\t' + str(sorted(revised_set.intersection(old_set))))
    print('Lemmas different in the new list than old counting %d. Before manual revision it was %d.'
          % (len(revised_set.difference(old_set)), len(raw_set.difference(old_set))))
    print('\t' + str(sorted(revised_set.difference(old_set))))
    print('Lemmas from old list that are not included in new list, counting %d. Before manual revision it was %d.'
          % (len(old_set.difference(revised_set)), len(old_set.difference(revised_set))))
    print('\t' + str(sorted(old_set.difference(revised_set))))
    print('Counts in base corpus for omitted lemmas from original list (only non-zero cases):')
    zero_counter = 0
    zero_list = []
    missing_lemmas = list(sorted(old_set.difference(revised_set)))
    rows_to_append = []
    for lemma in missing_lemmas:
        count = wln.word_to_score(lemma)
        if count > 0:
            dit_row = {"lemma" : lemma, "count" : count}
            rows_to_append.append(dit_row)
        else:
            zero_counter += 1
            zero_list.append(lemma)
    df_missing_lemmas = pd.DataFrame(rows_to_append)
    print(df_missing_lemmas.sort_values(by=["count"], ascending=False).to_string(max_rows=100))
    print('\t' + "Zero values for %d lemmas:" % zero_counter)
    print(zero_list)
    print('Counts for each lemma on the final list are returned by this function.')
    LEM_final_to_score = {}
    for lemma in wln.LEM_revised_list:
        LEM_final_to_score[lemma] = wln.word_to_score(lemma)
    return LEM_final_to_score


def classificationStatistics(title, new_quality_list, old_quality_list, **kwargs):
    """
    Function to print statistics for categories in two arrays. Use keyword t_type = "posneg" or "counts" to transform.
    Otherwise generic transformation will kick in.
    Based on:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    :param title: String for recognizing in output.
    :param new_quality_list: Array-like object with single vector.
    :param old_quality_list: Array-like object with single vector.
    :return: tuple of confusion matrix and kappa score.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import classification_report
    print("***Classification statistics for " + title + ".***")
    t_type = str(kwargs.get('t_type', None))
    trueval = []
    predictval = []
    transform = False
    labels = None
    if t_type == "counts":
        # transform counts or TFIDF frequencies to 0 or 1 for classification
        xa = np.array(old_quality_list)
        out = np.divide(xa, xa)
        out[xa == 0] = 0
        trueval = out.astype(int)
        xb = np.array(new_quality_list)
        out = np.divide(xb, xb)
        out[xb == 0] = 0
        predictval = out.astype(int)
        labels = ["0", "1"]
    elif t_type == "posneg":
        # it would be better to avoid looping in some revision,
        # use array comparisons eg. trueval[old_quality_list == 0] = 0
        transform = True
        for x in range(len(old_quality_list)):
            if np.isnan(old_quality_list[x]) or np.isnan(new_quality_list[x]):
                continue
            if old_quality_list[x] == 0 or new_quality_list[x] == 0:
                # classification as zero is tricky, it implies pos=neg which is rare
                continue
            else:
                # transforms ratios to three classes by sign.
                # trueval
                if old_quality_list[x] > 0:
                    trueval.append("POS")
                elif old_quality_list[x] < 0:
                    trueval.append("NEG")
                # predictval
                if new_quality_list[x] > 0:
                    predictval.append("POS")
                elif new_quality_list[x] < 0:
                    predictval.append("NEG")
    else:
        # other types generic.
        transform = True
        for x in range(len(old_quality_list)):
            if np.isnan(old_quality_list[x]) or np.isnan(new_quality_list[x]):
                continue
            else:
                trueval.append(old_quality_list[x])
                predictval.append(new_quality_list[x])
    if transform:
        # change classes using sklearn preprocessing
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(trueval)
        trueval_classes = le.classes_
        transformer = le.transform(trueval)
        trueval = transformer
        del le
        le = preprocessing.LabelEncoder()
        le.fit(predictval)
        predictval_classes = le.classes_
        transformer = le.transform(predictval)
        predictval = transformer
        del le
        if set(predictval_classes) == set(trueval_classes):
            print("Classes for two series match: " + str(trueval_classes))
            labels = trueval_classes
        else:
            print("The classes don't match: " + str(trueval_classes) + " vs. " + str(predictval_classes))
            return 1
    cm = confusion_matrix(trueval, predictval)
    print("Confusion matrix for " + title + ":\n%s" % cm)
    from sklearn.metrics import confusion_matrix
    kappa = cohen_kappa_score(trueval, predictval)
    print("Kohen's Kappa: " + str(kappa))
    print(classification_report(trueval, predictval, target_names=labels))
    return cm, kappa


def descriptiveStatistics(title, new_quality_list, old_quality_list, **kwargs):
    """
    Function compares sentiment indexes for two lists using simple statistics and correlations.
    :param title: String to be included in output.
    :param new_quality_list: Array-like object with single vector in interval scale (eg. floats).
    :param old_quality_list: Array-like object with single vector in interval scale (eg. floats).
    :param kwargs: Optional parameters to trigger switches.
    :return: person correlation coefficient and anova results
    """
    import scipy.stats as stats
    print("***Descriptive statistics for " + title + ".***")
    trueval = []
    predictval = []
    # remove NaN rows.
    for x in range(len(old_quality_list)):
        if np.isnan(old_quality_list[x]) or np.isnan(new_quality_list[x]):
            continue
        else:
            trueval.append(old_quality_list[x])
            predictval.append(new_quality_list[x])
    df = pd.DataFrame({"original list": trueval, "new list": predictval})
    print(df.describe())
    r = np.corrcoef(trueval, predictval)
    print("Pearson correlation coefficient:")
    print(r)
    print("ANOVA")
    anova = stats.f_oneway(trueval, predictval)
    print(anova)
    return r, anova


def english_french_lemmas(LEM_to_best_synset):
    """
    Find matching lemmas in English and French for each synset in dictionary.
    :param LEM_to_best_synset: dictionary from WLN Class object containing synsets as values
    :return: dictionary of synset.name: tuple of two lists of lemmas (English and French)
    """
    efl = {}
    for key, value in LEM_to_best_synset.items():
        for s in value:
            efl[s.name()] = (s.lemma_names(), s.lemma_names('fra'))
    return efl


def french_lemmas(efl, wordlist_name):
    """
    Generate a list of lemmas, a word-list, in French for scoring. Relies on a separate french_lemmas_revised function.
    :param efl: result of the english_french_lemmas function
    :param wordlist_name: name of wordlist matching the WLN object being used
    :return: list of lemmas without duplicates
    """
    out_list: List[str] = []
    lemma_set = set()  # use a set to remove duplicates.
    for v in efl.values():
        lemma_set.update(v[1])
    out_list = french_lemmas_revised(list(lemma_set), wordlist_name)
    return out_list


def french_lemmas_revised(lemma_list, wordlist_name):
    """
    Apply a blacklist and other rules to verify the automatically generated list of lemmas in French
    :param lemma_list: list of lemmas
    :param wordlist_name: name of wordlist matching the WLN object being used
    :return: list of verified lemmas
    """
    list_out: List[str] = []
    blacklist = []  # includes lemmas to be excluded
    whitelist = []  # includes lemmas that need to be included

    if wordlist_name == 'positive':
        blacklist = ["acquérir", "admissible", "aide", "ajout", "alliance", "amender", "arriver", "autoriser",
                     "bénéfice", "billet", "cadeau", "causer", "certifier", "collaborateur", "collaboration",
                     "collaborer", "combler", "commercial", "comptable", "conception", "conduire", "connaissance",
                     "conséquence", "constater", "coopérer", "création", "décider", "déclaration", "diriger", "donner",
                     "droit", "également", "élever", "emploi", "employer", "estimer", "expert", "fabriquer",
                     "fondation", "fonder", "fortement", "gain", "gratuit", "inhalation", "institution", "intérêt",
                     "introduire", "investir", "léger", "lent", "lentement", "libre", "lien", "lumière", "magnétique",
                     "maître", "manufacturer", "mettre", "note", "observer", "obtenir", "partenaire", "particulier",
                     "partir", "plomb", "populaire", "pousser", "présent", "présenter", "prix", "profit", "propre",
                     "puits", "qualification", "rapide", "rapidement", "réalisation", "réaliser", "récompenser",
                     "recouvrer", "régler", "rendement", "répercussion", "répondre", "reprendre", "représentatif",
                     "résolution", "restauration", "résulter", "salarier", "seul", "souhait", "technique",
                     "transparence", "transport", "transporter", "trouver", "typique", "uniquement", "unité",
                     "usage", "utilisation", "utiliser", "utilité", "vérifier", "voir", "charge", "interligne",
                     "capitale", "traversin", "augmentation", "augmenter"]
        whitelist = ["accomplisseur", "amelioration", "amélioré", "amendé", "avantageusement", "bienveillant",
                     "charitable", "généreux", "charitable", "digne", "encourageant", "réconfortant", "excellence",
                     "excitant", "favorable", "amical", "gagnant", "grand", "hausse", "haut", "incomparable",
                     "nonpareil", "leader", "magnifiquement", "attrayant", "meilleur", "plaisir", "précieux",
                     "prééminent", "premier", "rallye", "rebond", "renforcement", "raffermissement", "satifaction",
                     "splendide", "excellent", "formidable", "utilement"]
    elif wordlist_name == 'negative':
        blacklist = ["abolition", "aboutir", "abus", "accent", "accomplir", "achever", "adopter", "affirmer", "agir",
                     "aller", "altération", "amateur", "amener", "amortir", "appartenir", "appeler", "appliquer",
                     "arrêt", "arriver", "assumer", "astreindre", "atteindre", "atteinte", "attendre", "attester",
                     "augmenter", "automne", "avoir", "bénéficier", "bleu", "brusquement", "causer", "charge",
                     "chercher", "chose", "clore", "clos", "coin", "commission", "compenser", "ompléter", "comporter",
                     "composer", "comprendre", "compromettre", "concerner", "conclure", "conclusion", "concours",
                     "conduire", "congé", "conséquence", "considérable", "considérer", "consister", "consommer",
                     "constituer", "continuer", "contraire", "côte", "court", "coûteux", "créneau", "croître",
                     "déclaration", "déclarer", "découverte", "découvrir", "demande", "demandeur", "demeurer",
                     "démontrer", "départ", "dépenser", "déposer", "dernièrement", "détenir", "déterminer", "devenir",
                     "dire", "distinction", "distribuer", "divulguer", "domicile", "effet", "éligible", "employer",
                     "encourager", "engager", "englober", "enjeu", "enjoindre", "entreprise", "entretenir", "entrevue",
                     "envisager", "époque", "esprit", "essentiel", "examiner", "expiration", "exploitation",
                     "exploiter", "exposer", "fabrication", "facile", "facilement", "faciliter", "faire", "fardeau",
                     "fermer", "fin", "finir", "fixer", "fluide", "force", "fort", "garder", "honorer", "humide",
                     "impliquer", "important", "indiquer", "intégrer", "intensifier", "intéresser", "intérêt",
                     "intervalle", "intervenir", "introduire", "issue", "jeu", "jeune", "lever", "licenciement",
                     "lien", "lier", "livrer", "long", "maître", "malade", "match", "mettre", "minimiser", "modifier",
                     "montrer", "naturel", "noir", "observer", "occuper", "œuvrer", "ordre", "ouvert", "ouvrage",
                     "ouvrir", "partir", "parvenir", "passer", "ause", "période", "petit", "phrase", "placer", "point",
                     "pont", "porter", "poser", "poursuivre", "pouvoir", "précéder", "prendre", "présenter", "presser",
                     "prêter", "prévoir", "prix", "procurer", "prolongement", "proposer", "prouver", "question",
                     "quitter", "rappeler", "récemment", "récent", "echerche", "rechercher", "remettre", "rendre",
                     "reporter", "repos", "reprendre", "respecter", "rester", "résultat", "révéler", "revenir",
                     "rigoureux", "rouge", "saisir", "secret", "ignalement", "soin", "solvabilité", "sortie",
                     "souligner", "soumettre", "soutenir", "spéculatif", "subir", "sujet", "superviser", "supporter",
                     "supposer", "surcharge", "tâche", "tard", "témoigner", "temps", "tenir", "terme", "terminer",
                     "tête", "tirer", "trancher", "transférer", "transporter", "travailler", "trouver", "usage",
                     "utilisation", "utiliser", "vérifier", "vigueur", "viser", "visiter", "race", "seller"]
        whitelist = ["affaiblissement", "aggraver", "ajournement", "suspension", "clôture", "bas", "cessez", "cession",
                     "confiné", "conflit", "empiètement", "déclassement", "déclasser", "défaut", "non-paiement",
                     "défavorable", "démasquage", "démentir", "détériorer", "dégénèrer", "dégénèrer", "disculpé",
                     "dommages", "douteux", "effondrer", "endomager", "exigeant", "faux", "incorrect", "fermeture",
                     "gravement", "impayé", "impliqué", "incertain", "indu", "invalider", "anuler", "maltreter",
                     "mensonge", "obligation", "prolongé", "interminable", "réclamer", "restructurer", "retraitement",
                     "révoquer", "renier", "soudoyer", "corrompre", "suborner", "supprimer", "perdre", "vertigineuse",
                     "brutal", "volatil", "perte", "pertes"]
    elif wordlist_name == 'HenryPos2008':
        blacklist = ["abandonner", "aboutir", "aboutissement", "accroître", "achèvement", "addition", "adresser",
                     "adulte", "affecter", "aggraver", "aimer", "ajout", "aller", "aptitude", "arriver", "ascenseur",
                     "autant", "avoir", "causer", "céder", "certitude", "chaud", "chef", "comptabiliser", "conduire",
                     "connaissance", "considérablement", "consolider", "constater", "continuer", "contribuer", "côte",
                     "courir", "coûteux", "créer", "creuser", "culture", "déboucher", "délivrer", "déposer", "devenir",
                     "diriger", "donner", "dossier", "dresser", "droit", "également", "élaboration", "élaborer",
                     "émergence", "émettre", "enchanter", "enregistrement", "enregistrer", "entier", "entraîner",
                     "estimer", "être", "évaluer", "évident", "fabriquer", "fin", "finalement", "fournir", "gain",
                     "gonfler", "grandir", "grimper", "gros", "guider", "habile", "hauteur", "indiquer", "inscrire",
                     "intensité", "intérêt", "jouir", "laisser", "lancer", "largement", "lever", "livre", "livrer",
                     "mener", "mettre", "monter", "obtenir", "passer", "pousser", "pouvoir", "pratiquement", "précéder",
                     "prendre", "présent", "présider", "presque", "presser", "principalement", "priorité", "prix",
                     "produire", "profit", "prononcer", "prospérer", "publier", "puits", "qualification", "quasi",
                     "réalisation", "récompenser", "rédiger", "registre", "remplacer", "reprendre", "représenter",
                     "réserver", "résistance", "résulter", "retourner", "revenir", "rond", "rythme", "sain", "salaire",
                     "salarier", "sécurité", "sérieusement", "sérieux", "spécialité", "succéder", "surface", "surgir",
                     "tête", "tique", "ultérieur", "vers"]
        whitelist = ["amélioré", "amendé", "améliorer", "élever", "élever", "hisser", "encourageant", "encourageant",
                     "réconfortant", "excellent", "remarquable", "supérieur", "grand", "haut", "important", "meilleur",
                     "mieux", "renforcement", "raffermissement"]
    elif wordlist_name == 'HenryNeg2008':
        blacklist = ["aboutir", "accroître", "accumuler", "aligner", "amener", "arriver", "automne", "balançoire",
                     "barrière", "bassin", "bombarder", "bombe", "cascade", "cataracte", "cause", "causer", "compter",
                     "conformément", "consommer", "côte", "courir", "crépuscule", "croître", "cuir", "deçà",
                     "décharger", "déclinaison", "dépenser", "déposer", "dernier", "Down", "duvet", "entrer", "épingle",
                     "escarpement", "exposer", "Fear", "finir", "jet", "léger", "lumière", "mouche", "notamment",
                     "parvenir", "perle", "plus", "polir", "porte", "poser", "poudre", "pratiquement", "provenir",
                     "psychiatre", "Psychiatre", "relever", "remplir", "réussir", "revenir", "saut", "sembler",
                     "sortir", "surtout", "terre", "trancher", "trouver", "venir", "voûte", "faire", "partir",
                     "devenir", "aller"]
        whitelist = ["abattre", "affaiblissement", "baisser", "chuter", "chute", "baisse", "défavorable",
                     "désavantageux", "exigeant", "incertain", "instable", "modifiable", "réduite"]
    else:
        blacklist = ["avoir", "pouvoir", "prendre", "faire", "partir", "devenir", "aller"]

    for lemma in lemma_list:
        if lemma in blacklist:
            continue
        elif "_" in lemma:
            continue
        else:
            list_out.append(lemma)

    list_out = list(set(list_out + whitelist))

    return list_out


def parallel_corpus_scoring(efl, eng_dict, fra_dict):
    '''
    Generate a dataframe with synset-level statistics
    :param efl: result of the english_french_lemmas function
    :param eng_dict: gensim dictionary from an English corpus
    :param fra_dict: gensim dictionary from a French corpus
    :return: dataframe with synsets and statistics.
    '''
    synsets = []
    eng = []
    fra = []
    lemmas = []
    for key, value in efl.items():
        synsets.append(key)
        lemmas.append(value)
        (eng_lemmas, fr_lemmas) = value
        eng.append(0)
        fra.append(0)
        for lemma in eng_lemmas:
            try:
                hits = eng_dict.dfs[eng_dict.token2id[lemma]]
                eng[-1] += hits
            except:
                continue
        for lemma in fr_lemmas:
            try:
                hits = fra_dict.dfs[fra_dict.token2id[lemma]]
                fra[-1] += hits
            except:
                continue
    df_out = pd.DataFrame({'Synset': synsets, 'English': eng, 'French': fra, 'Lemmas': lemmas})
    return df_out


def corpus_scoring(word_list, gensim_dict):
    """
    Produces a dataframe with frequencies per lemma from word_list based on a corpus.
    :param word_list: a list of strings, lemmas
    :param gensim_dict: a dictionary produced by gensim from a corpus
    :return: dataframe
    """
    freq = []
    for lemma in word_list:
        score: int = 0
        try:
            score = gensim_dict.cfs[gensim_dict.token2id[lemma]]
        except:
            score = 0  # if lemma not in dictionary
        freq.append(score)
    df_out = pd.DataFrame({'Lemma': word_list, 'Frequency': freq})
    return df_out


''' THIS IS THE PLACE WHERE THE ACTUAL PROGRAM SCRIPT BEGINS '''

# input_word_lists is a list containing the names of word-lists the program uses, adjust if you change lists.

input_word_lists = ['positive', 'negative', 'weak_modal', 'strong_modal', 'constraining', 'uncertainty', 'HenryPos2006',
                    'HenryNeg2006', 'HenryPos2008', 'HenryNeg2008']

input_word_lists = ['positive', 'negative', 'HenryPos2008', 'HenryNeg2008']

counter_lists = 0  # variable responsible for the loop of quality in the tuples
dictionary_of_quality_OLD = {}
dictionary_of_quality_NEW = {}
dictionary_of_quality_tfidf_OLD = {}
dictionary_of_quality_tfidf_NEW = {}
WLN_DICT = {}  # for easier iteration through dictionaries, we create a dictionary of wln objects "WLN_DICT"

if section_dictionary == 1:
    print("Generating WordNet lexicons from input word-lists.")
    for l in input_word_lists:
        # for each list from "input_word_lists" we execute the WLN procedure to obtain dictionaries for each list
        print("Processing list: " + str(l))
        list_name = str(l + '.csv')  # for output, we divide the data into 2 categories: sets and dictionaries
        # run the main procedure and save results to dictionary:
        WLN_DICT[l] = WLN(list_name)
        # run list diagnostics, print to console and store results for saving:
        LEM_final_list_counts = list_diagnostics(list_name, WLN_DICT[l])  # print list diagnostics to console

        #  obtain a list of 30 most common lemmas in training corpus using the original and mapped list (old/new)
        max_hits_new = max_hits_function(WLN_DICT[l].LEM_revised_list)
        max_hits_old = max_hits_function(WLN_DICT[l].LEM_to_synsets.keys())

        #  make holders to save all elements to files
        wln_dicts = {'list_to_LEM': WLN_DICT[l].list_to_LEM, 'LEM_to_synsets': WLN_DICT[l].LEM_to_synsets,
                     'LEM_to_best_synset': WLN_DICT[l].LEM_to_best_synset,
                     'LEM_final_list_counts': LEM_final_list_counts}
        wln_sets = {'synsets_set': WLN_DICT[l].synsets_set, 'no_synsets_set': WLN_DICT[l].no_synsets_set,
                    'LEM_wordnet_set': WLN_DICT[l].LEM_wordnet_set,
                    'LEM_revised_list': WLN_DICT[l].LEM_revised_list}

        # save results
        save_path1 = './results/'  # variables for setting save path
        save_path2 = str(l + '/')
        save_path4 = '.csv'
        if section_saving_results == 1:
            if os.path.isdir("./results/" + l):  # and then we save all data in folders intended for them
                print("saving to existing folder")  # in the absence of a folder, creating it:
            else:
                print("creating folder and saving to new direction")
                os.makedirs("./results/" + l)

            for d in wln_dicts.keys():  # function for saving dictionaries:
                completeName = os.path.join(save_path1, save_path2,
                                            d + save_path4)  # save path of a specific dictionary
                # print(completeName)
                output = open(completeName, 'wt', encoding="utf-8")
                for key, value in wln_dicts[d].items():
                    output.write("\t".join((str(key), str(value))) + '\n')
                output.close()

            for s in wln_sets.keys():  # function for saving sets:
                completeName = os.path.join(save_path1, save_path2, s + save_path4)  # save path of a specific set
                # print(completeName)
                output = open(completeName, 'wt', encoding="utf-8")
                for synset in sorted(wln_sets[s]):
                    output.write(str(synset) + '\n')
                output.close()

            completeName = os.path.join(save_path1, save_path2, 'max_hits_new' + save_path4)  # save path of a max_hits
            # print(completeName)
            output = open(completeName, 'wt', encoding="utf-8")
            for tpl in max_hits_new:
                output.write(str(tpl[0]) + "\t" + str(tpl[1]) + '\n')
            output.close()

            completeName = os.path.join(save_path1, save_path2, 'max_hits_old' + save_path4)  # save path of a max_hits
            # print(completeName)
            output = open(completeName, 'wt', encoding="utf-8")
            for tpl in max_hits_old:
                output.write(str(tpl[0]) + "\t" + str(tpl[1]) + '\n')
            output.close()

    if os.path.isdir("./results/dictionaries"):  # and then we save all data in folders intended for them
        print("saving WLN_DICT to existing folder")  # in the absence of a folder, creating it:
        pickle.dump(WLN_DICT, open("./results/dictionaries/WLN_DICT.p", "wb"))
    else:
        print("creating folder and saving WLN_DICT to new direction")
        os.makedirs("./results/dictionaries")
        pickle.dump(WLN_DICT, open("./results/dictionaries/WLN_DICT.p", "wb"))
else:
    try:
        WLN_DICT = pickle.load(open("./results/dictionaries/WLN_DICT.p", "rb"))
    except:
        print("Failed to load WLN_DICT from pickle. Please create it first.")
        exit()

if section_EnglishCorpus_analysis == 1:
    #  analyze a corpus to evaluate the original vs. wordnet list in English
    #  load test corpus and apply TFIDF weighting in line with LM 2011
    print("Analyzing English corpus to compare original and WordNet lexicons, both counts and TFIDF.")
    corpus = MmCorpus(datapath(gcorpus_eng_path))
    model = TfidfModel(corpus)
    corpus_tfidf = model[corpus]
    ANG_DICT = Dictionary.load(datapath(gdict_eng_path))

    for l in input_word_lists:  # calculate frequencies for original and wordnet lists
        gensim_ids1 = get_gensim_ids(WLN_DICT[l].list_to_LEM.values(), ANG_DICT)  # original list
        gensim_ids2 = get_gensim_ids(WLN_DICT[l].LEM_revised_list, ANG_DICT)  # wordnet list

        tuple_of_quality = ()
        quality_list_OLD = []
        quality_list_NEW = []
        quality_list_tfidf_OLD = []
        quality_list_tfidf_NEW = []

        # simple counts
        for document in corpus:
            hits1 = 0
            hits2 = 0
            for tpl in document:
                if tpl[0] in gensim_ids1:
                    hits1 += tpl[1]
                if tpl[0] in gensim_ids2:
                    hits2 += tpl[1]
            quality_list_OLD.append(hits1)
            quality_list_NEW.append(hits2)
        dictionary_of_quality_OLD[counter_lists] = quality_list_OLD
        dictionary_of_quality_NEW[counter_lists] = quality_list_NEW

        # TFIDF frequencies
        for document in corpus_tfidf:
            hits1 = 0
            hits2 = 0
            for tpl in document:
                if tpl[0] in gensim_ids1:
                    hits1 += tpl[1]
                if tpl[0] in gensim_ids2:
                    hits2 += tpl[1]
            quality_list_tfidf_OLD.append(hits1)
            quality_list_tfidf_NEW.append(hits2)
        dictionary_of_quality_tfidf_OLD[counter_lists] = quality_list_tfidf_OLD
        dictionary_of_quality_tfidf_NEW[counter_lists] = quality_list_tfidf_NEW
        print()
        counter_lists += 1

if section_pickling_save == 1:
    print("Saving English corpus analysis results for later use to ./results/dictionaries.")
    if not os.path.isdir("./results/dictionaries"):  # and then we save all data in folders intended for them
        os.makedirs("./results/dictionaries")  # in the absence of a folder, creating it
    pickle.dump(dictionary_of_quality_OLD, open("./results/dictionaries/dictionary_of_quality_OLD.p", "wb"))
    pickle.dump(dictionary_of_quality_NEW, open("./results/dictionaries/dictionary_of_quality_NEW.p", "wb"))
    pickle.dump(dictionary_of_quality_tfidf_OLD,
                open("./results/dictionaries/dictionary_of_quality_tfidf_OLD.p", "wb"))
    pickle.dump(dictionary_of_quality_tfidf_NEW,
                open("./results/dictionaries/dictionary_of_quality_tfidf_NEW.p", "wb"))
    pickle.dump(WLN_DICT, open("./results/dictionaries/WLN_DICT.p", "wb"))

if section_pickling_load == 1:
    print("Loading English corpus analysis results saved earlier to compare with other languages.")
    dictionary_of_quality_tfidf_OLD = pickle.load(open("./results/dictionaries/dictionary_of_quality_tfidf_OLD.p", "rb"))
    dictionary_of_quality_tfidf_NEW = pickle.load(open("./results/dictionaries/dictionary_of_quality_tfidf_NEW.p", "rb"))
    dictionary_of_quality_OLD = pickle.load(open("./results/dictionaries/dictionary_of_quality_OLD.p", "rb"))
    dictionary_of_quality_NEW = pickle.load(open("./results/dictionaries/dictionary_of_quality_NEW.p", "rb"))

if statistics_old_vs_new == 1:
    print('### CLASSIFICATION STATISTICS for original vs. wordnet lists ###')
    # positive - list_0, negative - list_1
    # x_positive = [0, 6, 8]  # LM, Henry 2006, Henry 2008
    x_positive = [0, 2]  # only LM and Henry 2008
    for x in x_positive:
        arr1 = np.array(dictionary_of_quality_OLD[x])
        arr2 = np.array(dictionary_of_quality_OLD[x + 1])
        arr3 = np.add(arr1, arr2)
        arr4 = np.subtract(arr1, arr2)
        array_of_quality_OLD = np.divide(arr4, arr3)

        arr1 = np.array(dictionary_of_quality_NEW[x])
        arr2 = np.array(dictionary_of_quality_NEW[x + 1])
        arr3 = np.add(arr1, arr2)
        arr4 = np.subtract(arr1, arr2)
        array_of_quality_NEW = np.divide(arr4, arr3)

        if os.path.isdir("./results/arrays"):  # and then we save all data in folders intended for them
            print("saving to existing folder")  # in the absence of a folder, creating it:
        else:
            print("creating folder and saving to new direction")
            os.makedirs("./results/arrays")
        completeName = "./results/arrays/array_of_quality"
        np.savetxt(completeName + "_OLD_" + str(x), array_of_quality_OLD, fmt='%10.2f')
        np.savetxt(completeName + "_NEW_" + str(x), array_of_quality_NEW, fmt='%10.2f')

        # set list names for printing results.
        if x_positive == [0, 6, 8]:
            if x == 0:
                list_title = "LM"
            elif x == 6:
                list_title = "Henry 2006"
            elif x == 8:
                list_title = "Henry 2008"
            else:
                list_title = "what list is this?"
        elif x_positive == [0, 2]:
            if x == 0:
                list_title = "LM"
            elif x == 2:
                list_title = "Henry 2008"
            else:
                list_title = "what list is this?"
        else:
            list_title = "what list is this?"

        classificationStatistics(' using '.join(('Positive tone by frequencies', list_title)),
                                 np.array(dictionary_of_quality_NEW[x]),
                                 np.array(dictionary_of_quality_OLD[x]), t_type="counts")
        classificationStatistics(' using '.join(('Negative tone by frequencies', list_title)),
                                 np.array(dictionary_of_quality_NEW[x + 1]),
                                 np.array(dictionary_of_quality_OLD[x + 1]), t_type="counts")
        classificationStatistics(' using '.join(('Positive/Negative tone by ratio', list_title)), array_of_quality_NEW,
                                 array_of_quality_OLD, t_type="posneg")
        descriptiveStatistics(' using '.join(('Positive/Negative tone by ratio', list_title)), array_of_quality_NEW,
                              array_of_quality_OLD)

if section_FrenchCorpus_analysis == 1:
    print("Analyzing a French parallel corpus to test equivalence of measures between French and English.")
    if not os.path.isdir("./results/french"):
        os.makedirs("./results/french")
    try:  # check if the lists are already available
        french_lists = pickle.load(open("./results/french/french_lists.p", "rb"))
        print('Loading French word-lists prepared in an earlier run.')
    except:  # if lists are not there, create them and dump to pickle
        generate_french_lists = 1
    if generate_french_lists == 1:  # this may be set to 1 already in options section at top
        print('Generating French word-lists using OMW WordNet.')
        french_lists = {}
        print("Saving English corpus analysis results for later use to ./results/french.")
        for l in input_word_lists:
            list_name = str(l)
            # obtain French synsets and save a file with statistics
            efl_dictionary = english_french_lemmas(WLN_DICT[l].LEM_to_best_synset)
            df_parallel = parallel_corpus_scoring(efl_dictionary, Dictionary.load(datapath(gdict_eng_path)),
                                                  Dictionary.load(datapath(gdict_fr_path)))
            df_parallel.to_csv("results/french/" + "_".join(("df_parallel", list_name)) + ".csv")
            # generate word-lists from synsets
            french_lists[l] = french_lemmas(efl_dictionary, list_name)
            corpus_scoring(french_lists[l], Dictionary.load(datapath(gdict_fr_path))
                           ).to_csv("results/french/" + "_".join(("df_fr_score", list_name)) + ".csv")
        print('Generate and save new word-lists in French.')
        pickle.dump(french_lists, open("./results/french/french_lists.p", "wb"))

    # Apply the French lists to the corpus - it's basically a copy of EnglishCorpusAnalysis
    print('Analyzing the French corpus.')
    dictionary_of_quality_FR = {}
    dictionary_of_quality_tfidf_FR = {}
    corpus = MmCorpus(datapath(gcorpus_fr_path))
    model = TfidfModel(corpus)
    corpus_tfidf = model[corpus]
    FR_DICT = Dictionary.load(datapath(gdict_fr_path))
    counter_lists = 0
    for l in input_word_lists:
        gensim_ids = get_gensim_ids(french_lists[l], FR_DICT)
        tuple_of_quality = ()
        quality_list_FR = []
        quality_list_tfidf_FR = []

        for document in corpus:
            hits = 0
            for tpl in document:
                if tpl[0] in gensim_ids:
                    hits += tpl[1]
            quality_list_FR.append(hits)
        dictionary_of_quality_FR[counter_lists] = quality_list_FR

        for document in corpus_tfidf:
            hits = 0
            for tpl in document:
                if tpl[0] in gensim_ids:
                    hits += tpl[1]
            quality_list_tfidf_FR.append(hits)
        dictionary_of_quality_tfidf_FR[counter_lists] = quality_list_tfidf_FR
        counter_lists += 1


# Calculate classification statistics for specific lists:
# positive - list_0, negative - list_1
# x_positive = [0, 6, 8]
# x_positive = [8]
x_positive = [0, 2]
posneg_FR = {}
for x in x_positive:

    # calculate posneg ratios for the French data
    if corpus_type_TFIDF == 1:
        arr1 = np.array(dictionary_of_quality_tfidf_FR[x])
        arr2 = np.array(dictionary_of_quality_tfidf_FR[x + 1])
    else:
        arr1 = np.array(dictionary_of_quality_FR[x])
        arr2 = np.array(dictionary_of_quality_FR[x + 1])
    arr3 = np.add(arr1, arr2)
    arr4 = np.subtract(arr1, arr2)
    array_of_quality_FR = np.divide(arr4, arr3)
    del arr1, arr2, arr3, arr4

    # calculate posneg ratios for the English data
    if corpus_type_TFIDF == 1:
        arr1 = np.array(dictionary_of_quality_tfidf_NEW[x])
        arr2 = np.array(dictionary_of_quality_tfidf_NEW[x + 1])
    else:
        arr1 = np.array(dictionary_of_quality_NEW[x])
        arr2 = np.array(dictionary_of_quality_NEW[x + 1])
    arr3 = np.add(arr1, arr2)
    arr4 = np.subtract(arr1, arr2)
    array_of_quality_NEW = np.divide(arr4, arr3)
    del arr1, arr2, arr3, arr4

    if x_positive == [0, 6, 8]:
        if x == 0:
            list_title = "LM"
        elif x == 6:
            list_title = "Henry 2006"
        elif x == 8:
            list_title = "Henry 2008"
        else:
            list_title = "what list is this?"
    elif x_positive == [0, 2]:
        if x == 0:
            list_title = "LM"
        elif x == 2:
            list_title = "Henry 2008"
        else:
            list_title = "what list is this?"
    else:
        list_title = "what list is this?"

    posneg_FR[list_title] = array_of_quality_FR
    classificationStatistics(' using '.join(('French vs. English - Positive tone by frequencies', list_title)),
                             np.array(dictionary_of_quality_NEW[x]),
                             np.array(dictionary_of_quality_FR[x]), t_type="counts")
    classificationStatistics(' using '.join(('French vs. English - Negative tone by frequencies', list_title)),
                             np.array(dictionary_of_quality_NEW[x + 1]),
                             np.array(dictionary_of_quality_FR[x + 1]), t_type="counts")
    classificationStatistics(' using '.join(('French vs. English - Positive/Negative tone by ratio', list_title)),
                             array_of_quality_NEW, array_of_quality_FR, t_type="posneg")
    descriptiveStatistics(' using '.join(('French vs. English - Positive/Negative tone by ratio', list_title)),
                          array_of_quality_NEW, array_of_quality_FR)
