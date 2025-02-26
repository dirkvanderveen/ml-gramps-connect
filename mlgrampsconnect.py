# ==========================================
# File      : MLGrampsConnect.py
# Project   : ML_Genealogy053
# Task      : A019
# Date      : 25/11/2020
# Author    : Dirk J. van der Veen
# ==========================================

import random
import lxml.etree
import re
import Levenshtein
import multiprocessing
from multiprocessing import Pool
import sys
import joblib
import pickle
import csv
import os
from datetime import datetime

_JLN_SDN_OFFSET = 32083
_JLN_DAYS_PER_5_MONTHS = 153
_JLN_DAYS_PER_4_YEARS = 1461

_LEVENSHTEIN_DISTANCE_THRESHOLD = 20

ABS_AGE_DELTA_ONE_GENERATION = 50

COL_EVENT_HANDLE = 0
COL_EVENT_DATEVAL_VAL = 1
COL_EVENT_DATEVAL_TYPE = 2

COL_FAMILY_HANDLE = 0
COL_FAMILY_FATHER = 1
COL_FAMILY_MOTHER = 2
COL_FAMILY_CHILDREF_LIST = 3

COL_PERSON_HANDLE = 0
COL_PERSON_ID = 1
COL_PERSON_NAME_LIST = 2
COL_PERSON_GENDER = 3
COL_PERSON_BIRTH_DATE = 4
COL_PERSON_RESIDENCE = 5
COL_PERSON_OCCUPATION = 6
COL_PERSON_RELATIVES_TUPLE = 7

COL_RELATIVE_FAMILY_HANDLE = 0
COL_RELATIVE_PERSON_HANDLE = 1
COL_RELATIVE_LINKTYPE = 2

COL_CONNECTION_MAININDEX = 0
COL_CONNECTION_LINKINDEX = 1
COL_CONNECTION_LINKTYPE = 2

COL_PERSON_CONNECTION_INDEX_CONNSTARTIDX = 0
COL_PERSON_CONNECTION_INDEX_NKNOWNCONN = 1
COL_PERSON_CONNECTION_INDEX_NRANDCONN = 2

COL_PERSONLINK_MAININDEX = 0
COL_PERSONLINK_LINKINDEX = 1

COL_OCCUPATION_TABLE_SECTOR = 0
COL_OCCUPATION_TABLE_PROFESSION = 1
COL_OCCUPATION_TABLE_OCCUPATION = 2

MAIN_LINK_PERSON_FIELDNAMES = ("Main Person Index", "Link Person Index")
TARGET_FIELDNAME = "Linktype"
PERSON_CONNECTION_INDEX_FIELDNAMES = ("conn_start_idx", "n_known_conn", "n_rand_conn")     

_ARG_PERSONLINK_MLFEATURE_LIST = 0
_ARG_PERSONLINK_NAME_SIMILARITY_MODE = 1
_ARG_PERSONLINK_INCLUDE_NONE_DATES = 2
_ARG_PERSONLINK_MAX_ABS_AGE_DELTA = 3
_ARG_PERSONLINK_MAINPERSON_INDEX = 4
_ARG_PERSONLINK_MAINPERSON = 5
_ARG_PERSONLINK_LINKPERSON_INDEX = 6
_ARG_PERSONLINK_LINKPERSON = 7
_ARG_PERSONLINK_LINKTYPE = 8
_ARG_PERSONLINK_AGE_DELTA = 9
_ARG_PERSONLINK_KWARGS_FEATURES = 10


###################################################################
#
# Dictionary Functions
#
###################################################################

def dict_invert(dict):
    # values are expected to be unique
    return {value: key for key, value in dict.items()}


###################################################################
#
# Mapping Dictionaries and Functions
#
###################################################################

def gender_combinations() -> tuple:
    gender_combinations_tuple = (
        'M-M',
        'F-F',
        'M-F',
        'F-M',
        'M-U',
        'F-U',
        'U-M',
        'U-F',
        'U-U')
    return gender_combinations_tuple

def gender_combination_strtoint_mapping() -> dict:
    gender_combinations_list = gender_combinations()
    return {gender_combinations_list[i]: i for i in range(len(gender_combinations_list))}

def gender_combination_inttostr_mapping() -> dict:
    return dict_invert(gender_combination_strtoint_mapping())

linktype_modes = ('ByGender', 'Neutral')

def linktypes(linktype_mode: str,
              include_unknown: bool) -> tuple:
    linktypes_tuple = tuple()
    mode = linktype_mode.lower()
    if mode == 'bygender':
        linktypes_tuple = (
            'Vader',
            'Moeder',
            'Man',
            'Vrouw',
            'Broer/zus',
            'Kind')
    elif mode == 'neutral':
        linktypes_tuple = (
            'Ouder',
            'Echtgeno(o)t(e)',
            'Broer/zus',
            'Kind')
    else:
        linktypes_tuple = ()
    if include_unknown:
        linktypes_tuple = linktypes_tuple + ('Onbekend',)
    return linktypes_tuple
    
def linktype_strtoint_mapping(linktype_mode: str,
                              include_unknown: bool) -> dict:
    linktypes_list = linktypes(linktype_mode, include_unknown)
    return {linktypes_list[i]: i for i in range(len(linktypes_list))}

def linktype_inttostr_mapping(linktype_mode: str,
                              include_unknown: bool) -> dict:
    return dict_invert(linktype_strtoint_mapping(
        linktype_mode=linktype_mode, include_unknown=include_unknown))


###################################################################
#
# List Functions
#
###################################################################

def filter_duplicates_and_special_words(filter_lists: tuple,
                                        text_list: list) -> list:
    # delete duplicates
    text_list = list(dict.fromkeys(text_list))
    # filter items from lists
    for filter_list in filter_lists:
        text_list = list(filter(lambda text: text not in filter_list, text_list))
    return text_list

def get_handle_list(elem, xmlns, child_tag: str, attrib_tag: str) -> list:
    handle_list = []
    # get child element(s)
    child_elem_list = elem.findall(xmlns+child_tag)
    for child_elem in child_elem_list:
        # get handle
        handle = child_elem.get(attrib_tag)
        if handle:
            handle_list.append(handle)
    return handle_list
    
def get_listitem_from_list_by_handle(list, handle, col_handle):
    listitem = None
    listindex = None
    for i in range(len(list)):
        if list[i][col_handle] == handle:
            listitem = list[i]
            listindex = i
            break
    return (listitem, listindex)

def get_random_handle_from_list(list, col_handle):
    number_of_listitems = len(list)
    random_listitem_number = random.randint(0, number_of_listitems - 1)
    random_handle = list[random_listitem_number][col_handle]
    return random_handle
    
def get_value_from_list_by_dict(list, where_clause_dict, col_value):
    # find listitem where ALL dict elem are EQ
    # dict elem: key = column_number, value = column_value
    value = None
    for i in range(len(list)):
        found = False
        for key in where_clause_dict:
            if list[i][key] == where_clause_dict.get(key):
                found = True
            if not found:
                break
        if found:
            listitem = list[i]
            if listitem:
                value = listitem[col_value]
            break
    return value

def save_list_as_csv(filename: str, list: list, column_headings: list):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(column_headings)
        writer.writerows(list)

def import_list_from_csv(filename: str, has_heading: bool, delimiter: str = ",") -> (list, tuple):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # imported_list = list(reader)
        imported_list = [tuple(row) for row in reader]

        if has_heading:
            l = len(imported_list)
            if l > 0:
                imported_heading = imported_list[0]
            if l > 1:
                imported_list = imported_list[1:]

    return (imported_list, imported_heading)


###################################################################
#
# Date Functions
#
###################################################################

# For compatabililty reasons the julian-sdn is used from
# the Gramps (genealogy) system is used (gramps-project.org)

def julian_sdn(year, month, day):
    """ Convert a Julian calendar date to a serial date number (SDN).
    """

    if year < 0:
        year += 4801
    else:
        year += 4800

    # Adjust the start of the year
    if month > 2:
        month -= 3
    else:
        month += 9
        year -= 1

    return (year * _JLN_DAYS_PER_4_YEARS) // 4 \
           + (month * _JLN_DAYS_PER_5_MONTHS + 2) // 5 \
           + day - _JLN_SDN_OFFSET

def get_date_sort_value(textdate: str) -> int:
    """ function to convert a date text string to a
        year, month, day format and then to a julian_sdn.
        The gramps internal text format for dates is Y-M-D
    """
    sort_value = 0
    bc = False
    year = 0
    month = 0
    day = 0
    if textdate != '':
        # Inside Gramps dates are converted to strings in the
        # format 'Y-M-D', 'Y-M' or 'Y'. Dates "B.C." start with
        # a minus character like '-Y-M-D', '-Y-M' or '-Y'. This
        # minus sinus may NOT be take into account in the
        # textdate.split('-') function! 
        if textdate[0] == '-':
            textdate = textdate[1:]
            bc = True
        dateparts = textdate.split('-')
        number_of_dateparts = len(dateparts)
        if number_of_dateparts > 0:
            if dateparts[0] != '':
                year = int(dateparts[0])
                if bc:
                    year = -year
            if number_of_dateparts > 1:
                month = int(dateparts[1])
                if number_of_dateparts > 2:
                    day = int(dateparts[2])
            sort_value = julian_sdn(year, month, day)
    return sort_value

def get_age_delta_inyears(firstdate: str, lastdate: str,
                          accept_none_dates: bool,
                          max_abs_age_delta: int) -> (float, bool):
    """ function to calculate the difference between the
        (in this case birth)dates of two persons
    """
    age_delta_inyears = None
    result = False

    if firstdate and lastdate:
        if firstdate:
            firstdays = get_date_sort_value(firstdate)
        else:
            firstdays = 0
        if lastdate:
            lastdays = get_date_sort_value(lastdate)
        else:
            lastdays = 0
        age_delta_indays = lastdays - firstdays
        # round at 2 digits, a day is about 0,00274 year
        age_delta_inyears = round(age_delta_indays / 365.25, 2)
        if max_abs_age_delta < 0:
            # Include all age_delta values
            result = True
        else:
            # Include only if abs is LE then max
            result = abs(age_delta_inyears) <= max_abs_age_delta
    elif accept_none_dates:
        result = True

    return (age_delta_inyears, result)


###################################################################
#
# Text Functions
#
###################################################################

def replace_words(replacement_list: list, text: str) -> str:
    # replace typos, etc. (this is case-sensitive!)
    for replacement in replacement_list:
        if replacement[0] in text:
            text = text.replace(replacement[0], replacement[1])
    # delete characters
    text = text.replace('"', ' ')
    text = text.replace(',', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    text = text.replace('&', ' ')
    text = text.replace("'", ' ')
    text = text.replace('?', ' ')
    text = text.replace('/', ' ')
    # delete digits
    for digit in range(10):
        text = text.replace(str(digit), ' ')
    # return result
    return text


###################################################################
#
# Value Date List/Item Functions
#
###################################################################

class ValueDateList:
    def __init__(self, valueasstring):
        self.valueasstring = valueasstring
        self.valuedatelist = self.get_valueaslist(self.valueasstring)

    def get_valueaslist(self, valueasstring):
        # Syntax: [value segment] [date segment],
        #         [value segment] [date segment, ...]
        # [value segment] = value[, value][, ...]
        # [date segment] = (date[[, date], ...])
        valuedatelist = []
        remainderstr = valueasstring
        str_not_empty = len(remainderstr) > 0
        while str_not_empty:
            valstr = ""
            datstr = ""
            remainderstr = remainderstr.strip(",").strip()
            pos_comma = remainderstr.find(",")
            pos_parentheses_begin = remainderstr.find("(")

            if pos_parentheses_begin == -1:
                # Geen date segment (meer)
                if pos_comma == -1:
                    # 1 value segment (over)
                    valstr = remainderstr
                    remainderstr = ""
                else:
                    # Meer dan 1 value segment (over)
                    valstr = remainderstr[:pos_comma].strip()
                    remainderstr = remainderstr[pos_comma+1:]
            elif pos_parentheses_begin == 0:
                # Wel date segment (meer), maar zonder als eerste
                # een value segment
                pos_parentheses_end = remainderstr.find(")")
                if pos_parentheses_end == -1:
                    # Ten onrechte geen eindhaak meer, dan remainderstr
                    # geheel opnemen als datstr
                    datstr = remainderstr[1:].strip()
                    remainderstr = ""
                else:
                    # Alles tot de eindhaak, dus inclusief door komma's
                    # gescheiden datum delen, nemen als datstr
                    datstr = remainderstr[1:pos_parentheses_end].strip()
                    remainderstr = remainderstr[pos_parentheses_end+1:].strip()
            else:
                # Minimaal 1 date segment (meer), inclusief een of meer
                # voorafgaande value segmenten
                if (pos_comma != -1) and (pos_comma < pos_parentheses_begin):
                    # meer dan 1 value segment voor minimaal 1 aanwezige
                    # date segment, dus dat(die) eerste(n) value segement(en)
                    # heeft(hebben) zelf geen date segment
                    valstr = remainderstr[:pos_comma].strip()
                    remainderstr = remainderstr[pos_comma+1:]
                else:
                    # 1 value segment voor minimaal 1 aanwezige date segment
                    pos_parentheses_end = remainderstr.find(")")
                    if pos_parentheses_end == -1:
                        # Ten onrechte geen eindhaak meer, dan remainderstr
                        # geheel opnemen als datstr
                        valstr = remainderstr[:pos_parentheses_begin].strip()
                        datstr = remainderstr[pos_parentheses_begin+1:].strip()
                        remainderstr = ""
                    else:
                        # 1 value segment voor minimaal 1 aanwezige date segment
                        valstr = remainderstr[:pos_parentheses_begin].strip()
                        datstr = remainderstr[
                            pos_parentheses_begin+1:pos_parentheses_end].strip()
                        remainderstr = remainderstr[pos_parentheses_end+1:]

            valdatitem = ValueDateItem(valstr, datstr)
            valuedatelist.append(valdatitem)
            str_not_empty = len(remainderstr) > 0

        return valuedatelist

    def HasValueIn(self, valuedatelist_obj):
        found = False
        for valdatitem1 in self.valuedatelist:
            for valdatitem2 in valuedatelist_obj.valuedatelist:
                if valdatitem1.valuestr == valdatitem2.valuestr:
                    found = True
                    break
            if found:
                break
        return found

class ValueDateItem:
    def __init__(self, valuestr, datestr):
        self.valuestr = valuestr
        self.datestr = datestr

def get_valuedatelist_correspondence(valuedatelist1, valuedatelist2):
    correspondence = 0.0
    if valuedatelist1 and valuedatelist2:
        if (valuedatelist1 != "") or (valuedatelist2 != ""):
            valuedatelist1_obj = ValueDateList(valuedatelist1)
            valuedatelist2_obj = ValueDateList(valuedatelist2)
            if valuedatelist2_obj.HasValueIn(valuedatelist1_obj):
                correspondence = 1.0    
    return correspondence


###################################################################
#
# Person Support Functions
#
###################################################################

def create_personlink(args: tuple) -> tuple:
    # This person suppert function is defined outside the
    # MGGrampsConnect object because otherwise it cam't be
    # called in the multiprocess pool/map construction

    mlfeature_list = args[_ARG_PERSONLINK_MLFEATURE_LIST]
    name_similarity_mode = args[_ARG_PERSONLINK_NAME_SIMILARITY_MODE]
    include_none_dates = args[_ARG_PERSONLINK_INCLUDE_NONE_DATES]
    max_abs_age_delta = args[_ARG_PERSONLINK_MAX_ABS_AGE_DELTA]

    mainperson_index = args[_ARG_PERSONLINK_MAINPERSON_INDEX]
    mainperson  = args[_ARG_PERSONLINK_MAINPERSON]
    linkperson_index = args[_ARG_PERSONLINK_LINKPERSON_INDEX]
    linkperson = args[_ARG_PERSONLINK_LINKPERSON]
    linktype = args[_ARG_PERSONLINK_LINKTYPE]
    age_delta = args[_ARG_PERSONLINK_AGE_DELTA]
    
    kwargs_features = args[_ARG_PERSONLINK_KWARGS_FEATURES]
    
    # create and add personlink
    personlink = (mainperson_index, linkperson_index, linktype)
    # calc all feature values for this specific connection_list
    result = False
    for mlfeature in mlfeature_list:
        if mlfeature.get_name().lower() == "agedelta":
            featurevalue = age_delta
            result = True
        else:
            featurevalue, result = mlfeature.get_value(mainperson, linkperson, linktype,
                name_similarity_mode=name_similarity_mode,
                include_none_dates=include_none_dates,
                max_abs_age_delta=max_abs_age_delta,
                **kwargs_features)
        if result:
            personlink = personlink + (featurevalue,)
        else:
            # Stop the loop and dan't add the connection
            personlink = None
            break

    return personlink


###################################################################
#
# MLFeature Class
#
###################################################################

class MLFeature:
    def __init__(self):
        pass

    def get_name(self):
        return None

    def get_title(self):
        return None

    def get_value(self):
        return None

class MLFeatureAgeDelta(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "AgeDelta"
        
    def get_title(self):
        return "Age Delta"
        
    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_occupation_correespondence):
        # TODO For now, to train the model only connections with known birth_dates
        # will be used. This can later be changed to also include
        # persons with unknown birthdates to reflect better real life data

        age_delta = None
        result = False

        mainperson_birth_date = mainperson[COL_PERSON_BIRTH_DATE]
        linkperson_birth_date = linkperson[COL_PERSON_BIRTH_DATE]

        mp_birth_date = None
        if mainperson_birth_date:
            mp_birth_date = mainperson_birth_date[0]
        lp_birth_date = None
        if linkperson_birth_date:
            lp_birth_date = linkperson_birth_date[0]
        age_delta, result = get_age_delta_inyears(lp_birth_date, mp_birth_date,
            accept_none_dates=include_none_dates,
            max_abs_age_delta=max_abs_age_delta)

        return (age_delta, result)


class MLFeatureGenderCombination(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "GenderCombination"
        
    def get_title(self):
        return "Gender Combination"
        
    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_occupation_corresponence):
        """ Get the combination of the gender of two persons
        """
        gender_combination = None
        result = True

        if mainperson and linkperson:
            mainperson_gender = mainperson[COL_PERSON_GENDER]
            linkperson_gender = linkperson[COL_PERSON_GENDER]
            gender_combination = "{}-{}".format(mainperson_gender, linkperson_gender)

        return (gender_combination, result)


class MLFeatureKnownLinktype(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "KnownLinktype"
        
    def get_title(self):
        return "Known Linktype"
        
    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_features):
        """ Set to 1 if the linktype is known, otherwise set to 0
        """
        result = True

        if linktype in ("Vader", "Moeder", "Ouder",
                        "Man", "Vrouw", "Echtgeno(o)t(e)",
                        'Broer/zus', 'Kind'):
            known_linktype = 1.0
        else:
            known_linktype = 0.0

        return (known_linktype, result)


class MLFeatureNSiblingsEquality(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "NSiblingsEquality"
        
    def get_title(self):
        return "Number of Siblings Equality"
        
    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_features):
        """ Get the equality, depending of linktype, between the number of nsiblings
        """
        def get_nsiblings(relatives: tuple, linktype: str) -> int:
            nsiblings = 0
            for relative in relatives:
                if relative[COL_RELATIVE_LINKTYPE] == linktype:
                    nsiblings += 1
            return nsiblings

        nsiblings_equality = 0.0
        result = True

        if mainperson and linkperson:
            mainperson_relatives = mainperson[COL_PERSON_RELATIVES_TUPLE]
            linkperson_relatives = linkperson[COL_PERSON_RELATIVES_TUPLE]

            # Perhaps as a principle select and count only childs within the same
            # family group (marriage)! BUT... For a case with more family groups
            # it isn't clear which one should be taken, so as second best option:
            # just compare the total relatives of the given type.
            if linktype in ("Vader", "Moeder", "Ouder"):
                if get_nsiblings(mainperson_relatives, "Broer/zus") + 1 == \
                   get_nsiblings(linkperson_relatives, "Kind"):
                    nsiblings_equality = 1.0
            elif linktype in ("Man", "Vrouw", "Echtgeno(o)t(e)"):
                if get_nsiblings(mainperson_relatives, "Kind") == \
                   get_nsiblings(linkperson_relatives, "Kind"):
                    nsiblings_equality = 1.0
            elif linktype == 'Broer/zus':
                if get_nsiblings(mainperson_relatives, "Broer/zus") == \
                   get_nsiblings(linkperson_relatives, "Broer/zus"):
                    nsiblings_equality = 1.0
            elif linktype == 'Kind':
                if get_nsiblings(mainperson_relatives, "Kind") == \
                   get_nsiblings(linkperson_relatives, "Broer/zus") + 1:
                    nsiblings_equality = 1.0
            elif linktype == 'Onbekend':
                nsiblings_equality = 0.0

        return (nsiblings_equality, result)


class MLFeatureOccupationCorrespondence(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "OccupationCorrespondence"

    def get_title(self):
        return "Occupation Correspondence"

    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_features):

        def get_occupation_list(occupation: str) -> list:
            occupation_replacement_table = kwargs_features['occupation_replacement_table']
            stopword_words_list = kwargs_features['stopword_words_list']
            place_words_list = kwargs_features['place_words_list']
            occupation_exclude_words_list = kwargs_features['occupation_exclude_words_list']

            occupation_list = []
            occupation = replace_words(occupation_replacement_table, occupation)
            # split occupation string in words
            for occupation_word in occupation.split():
                # occupation_list.append((occupation_word.lower(),))
                occupation_list.append(occupation_word.lower())

            # Filter duplicates and words from (in this order) stop_words_list,
            # place_words_list and occupation_exclude_words_list     
            occupation_list = filter_duplicates_and_special_words(
                (stopword_words_list, place_words_list, occupation_exclude_words_list),
                occupation_list)
            return occupation_list

        # set parameter use_occupation_table
        use_occupation_table = kwargs_features['use_occupation_table']
        # set parameter occupation_table
        occupation_table = kwargs_features['occupation_table']
        
        # seet a defaukt if the parameter was not found (or is None)
        if use_occupation_table is None:
            use_occupation_table = False
        
        result = True

        mainperson_occupation = mainperson[COL_PERSON_OCCUPATION]
        linkperson_occupation = linkperson[COL_PERSON_OCCUPATION]

        if use_occupation_table:
            occupation_correspondence = 0.0
            if mainperson_occupation and linkperson_occupation:
                if (mainperson_occupation != "") or (linkperson_occupation != ""):
                    mainperson_occupation_list = get_occupation_list(mainperson_occupation)
                    linkperson_occupation_list = get_occupation_list(linkperson_occupation)

                    for main_occupation in mainperson_occupation_list:
                        occupation_itm, occupation_idx = get_listitem_from_list_by_handle(
                            occupation_table, main_occupation, COL_OCCUPATION_TABLE_OCCUPATION)
                        if not occupation_itm is None:
                            main_profession = occupation_itm[COL_OCCUPATION_TABLE_PROFESSION]
                            main_sector = occupation_itm[COL_OCCUPATION_TABLE_SECTOR]
                        else:
                            main_profession = None
                            main_sector = None
                            
                        for link_occupation in linkperson_occupation_list:
                            occupation_itm, occupation_idx = get_listitem_from_list_by_handle(
                                occupation_table, link_occupation, COL_OCCUPATION_TABLE_OCCUPATION)
                            if not occupation_itm is None:
                                link_profession = occupation_itm[COL_OCCUPATION_TABLE_PROFESSION]
                                link_sector = occupation_itm[COL_OCCUPATION_TABLE_SECTOR]
                            else:
                                link_profession = None
                                link_sector = None

                            correspondence = 0.0
                            if main_occupation and (main_occupation == link_occupation):
                                # correspondence += 0.34
                                correspondence += 0.10
                            if main_profession and (main_profession == link_profession):
                                # correspondence += 0.33
                                correspondence += 0.10
                            if main_sector and (main_sector == link_sector):
                                # correspondence += 0.33
                                correspondence += 0.80

                            if correspondence > occupation_correspondence:
                                occupation_correspondence = correspondence
                            
                            # if correspondence == 0.34:
                            #     print("{} - {} - {} = {} - {} - {} = {} - {}".format(main_occupation, main_profession, main_sector,
                            #                                                             link_occupation, link_profession, link_sector,
                            #                                                             correspondence, occupation_correspondence))

        else:
            occupation_correspondence = get_valuedatelist_correspondence(mainperson_occupation, linkperson_occupation)
        
        return (occupation_correspondence, result)


class MLFeatureResidenceCorrespondence(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "ResidenceCorrespondence"

    def get_title(self):
        return "Residence Correspondence"

    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: str,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_features):
        result = True

        mainperson_residence = mainperson[COL_PERSON_RESIDENCE]
        linkperson_residence = linkperson[COL_PERSON_RESIDENCE]

        residence_correspondence = get_valuedatelist_correspondence(mainperson_residence, linkperson_residence)
        
        return (residence_correspondence, result)


class MLFeatureSurnameSimilarity(MLFeature):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "SurnameSimilarity"

    def get_title(self):
        return "Surname Similarity"

    def get_value(self, mainperson, linkperson, linktype,
                  name_similarity_mode: tuple,
                  include_none_dates: bool,
                  max_abs_age_delta: int,
                  **kwargs_features):
        """ Get the similarity between two names (which could consist of several names
            parts) based on the Levenmshtein distance

            name_similarity_mode: ('LevenshteinDistance', threshold)
            name_similarity_mode: ('LevenshteinDistanceBool', threshold)
            name_similarity_mode: ('LevenshteinDistanceRelative', threshold)
        """
        surname_similarity = 0.0
        result = True
        
        mainperson_name_list = mainperson[COL_PERSON_NAME_LIST]
        linkperson_name_list = linkperson[COL_PERSON_NAME_LIST]

        # The default method is None, which means no check at all and the outcome is 0.0
        # For all methods the default threshold is 0 which means only exacts matches are count for 1.0
        similarity_method = name_similarity_mode[0]
        threshold = name_similarity_mode[1]

        if similarity_method == 'LevenshteinDistance':
            min_distance = threshold + 1
            for mainfullsurname in mainperson_name_list:
                for linkfullsurname in linkperson_name_list:
                    distance = Levenshtein._levenshtein.distance(mainfullsurname, linkfullsurname)
                    if distance < min_distance:
                        min_distance = distance
            if min_distance <= threshold:
                if threshold != 0:
                    surname_similarity = round(1 - (min_distance / threshold), 2)
                else:
                    surname_similarity = 1.0
        elif similarity_method == 'LevenshteinDistanceBool':
            min_distance = threshold + 1
            for mainfullsurname in mainperson_name_list:
                for linkfullsurname in linkperson_name_list:
                    distance = Levenshtein._levenshtein.distance(mainfullsurname, linkfullsurname)
                    if distance < min_distance:
                        min_distance = distance
            if min_distance <= threshold:
                surname_similarity = 1.0                
        elif similarity_method == 'LevenshteinDistanceRelative':
            # max_similarity_score = 0
            # for mainfullsurname in mainperson_name_list:
            #     lenghtofmainname = len(mainfullsurname)
            #     max_distance = round(threshold * lenghtofmainname)
            #     if max_distance == 0:
            #         max_distance = 1
            #     for linkfullsurname in linkperson_name_list:
            #         similarity_score = round(1 - (Levenshtein._levenshtein.distance(mainfullsurname, linkfullsurname) /
            #                                       (threshold * lenghtofmainname)), 2)
            #         if similarity_score < 0:
            #             similarity_score = 0
            #         if similarity_score > max_similarity_score:
            #             max_similarity_score = similarity_score
            # surname_similarity = max_similarity_score
            
            for mainfullsurname in mainperson_name_list:
                similarity = 0.0
                # rel_threshold = round(threshold * (len(mainfullsurname) / 6))
                rel_threshold = len(mainfullsurname)
                min_distance = rel_threshold + 1
                for linkfullsurname in linkperson_name_list:
                    distance = Levenshtein._levenshtein.distance(mainfullsurname, linkfullsurname)
                    if distance < min_distance:
                        min_distance = distance
                if min_distance <= rel_threshold:
                    if rel_threshold != 0:
                        similarity = round(1 - (min_distance / rel_threshold), 2)
                    else:
                        similarity = 1.0
                if similarity > surname_similarity:
                    surname_similarity = similarity           
        else:
            surname_similarity = 0.0
            result = False

        return (surname_similarity, result)


###################################################################
#
# MLGrampsConnect Class
#
###################################################################

class MLGrampsConnect:
    def __init__(self):
        self.filename = None
        self.familytree = None
        self.familytree_root = None
        self.xmlns = None
        self._combined_list = []

    def load(self, xml_filename):
        # load tree
        familytree = lxml.etree.parse(xml_filename)
        if familytree:
            self.filename = xml_filename
            self.familytree = familytree
            # get root
            self.familytree_root = self.familytree.getroot()
            # get namespace
            self.xmlns = ''
            m = re.search('{.*}', self.familytree_root.tag)
            if m:
                self.xmlns = m.group(0)

    def _get_birth_event_list(self):
        birth_event_list = []
        for birth_event_elem in self.familytree_root.findall(self.xmlns+"events"+"/"+self.xmlns+"event"):
            # get event handle
            event_handle = birth_event_elem.get('handle')
            # get type of event
            event_type = None
            event_type_elem_list = birth_event_elem.findall(self.xmlns+"type")
            if event_type_elem_list:
                event_type = event_type_elem_list[0].text
            if event_type == 'Birth':
                # get dateval
                dateval_elem_list = birth_event_elem.findall(self.xmlns+"dateval")
                if dateval_elem_list:
                    dateval_val = dateval_elem_list[0].get('val')
                    dateval_type = dateval_elem_list[0].get('type')
                    birth_event_list.append([event_handle, dateval_val, dateval_type])
        return birth_event_list

    def get_family_list(self):
        family_list = []
        for family_elem in self.familytree_root.findall(self.xmlns+"families"+"/"+self.xmlns+"family"):
            # get family handle
            family_handle = family_elem.get('handle')
            # get father
            father = None
            father_elem_list = family_elem.findall(self.xmlns+"father")
            if father_elem_list:
                father = father_elem_list[0].get("hlink")
            # get mother
            mother = None
            mother_elem_list = family_elem.findall(self.xmlns+"mother")
            if mother_elem_list:
                mother = mother_elem_list[0].get("hlink")
            # get children
            childrefs_tuple = tuple()
            childref_elem_list = family_elem.findall(self.xmlns+"childref")
            for childref_elem in childref_elem_list:
                childref = childref_elem.get("hlink")
                childrefs_tuple = childrefs_tuple + (childref,)
            family_list.append((family_handle, father, mother, childrefs_tuple))
        return family_list

    def get_mlfeature_list(self, features):
        mlfeature_list = []
        for feature in features:
            feature_lower = feature.lower()
            # create the feature object
            mlfeature = None
            if feature_lower == "gendercombination":
                mlfeature = MLFeatureGenderCombination()
            elif feature_lower == "agedelta":
                mlfeature = MLFeatureAgeDelta()
            elif feature_lower == "surnamesimilarity":
                mlfeature = MLFeatureSurnameSimilarity()
            elif feature_lower == "occupationcorrespondence":
                mlfeature = MLFeatureOccupationCorrespondence()
            elif feature_lower == "residencecorrespondence":
                mlfeature = MLFeatureResidenceCorrespondence()
            elif feature_lower == "knownlinktype":
                mlfeature = MLFeatureKnownLinktype()
            elif feature_lower == "numberofsiblingsequality":
                mlfeature = MLFeatureNSiblingsEquality()
            # add the feature object to a list
            if mlfeature:
                mlfeature_list.append(mlfeature)
        return mlfeature_list

    def get_occupation_list(self, occupation_replacement_table: list,
                                  stopword_words_list,
                                  place_words_list,
                                  occupation_exclude_words_list,
                                  sort: str = None):
        _OCCUPATION_FIELDNAMES = ('Name',)

        # # Load occupation replacements
        # occupation_replacement_table_csv = cur_dir_path + "/" + "occupation_replacement_table.csv"
        # occupation_replacement_table, occupation_replacement_headings = import_list_from_csv(occupation_replacement_table_csv, has_heading=True)
        
        # # Load stopwords
        # stopword_table_csv = cur_dir_path + "/" + "stopword_table.csv"
        # stopword_table, stopword_headings = import_list_from_csv(stopword_table_csv, has_heading=True)
        # stopword_words_list = [stopword[0].lower() for stopword in stopword_table]

        # # Load places
        # place_table_csv = cur_dir_path + "/" + "place_table.csv"
        # place_table, place_headings = import_list_from_csv(place_table_csv, has_heading=True)
        # place_words_list = [place[0].lower() for place in place_table]

        # # Load occupation excludes
        # occupation_exclude_table_csv = cur_dir_path + "/" + "occupation_exclude_table.csv"
        # occupation_exclude_table, occupation_exclude_headings = import_list_from_csv(occupation_exclude_table_csv, has_heading=True)
        # occupation_exclude_words_list = [occupation_exclude[0].lower() for occupation_exclude in occupation_exclude_table]

        occupation_list = []
        # get attributes Occupation
        for attribute_elem in self.familytree_root.findall(self.xmlns+"people"+"/"+self.xmlns+"person"+"/"+self.xmlns+"attribute"):
            attribute_type = attribute_elem.get('type')
            attribute_value = attribute_elem.get('value')
            if attribute_type == 'Beroep':
                occupation = attribute_value
                occupation = replace_words(occupation_replacement_table, occupation)
                # split occupation string in words
                for occupation_word in occupation.split():
                    occupation_list.append(occupation_word.lower())

        # Filter duplicates and words from (in this order) stop_words_list,
        # place_words_list and occupation_exclude_words_list    
        occupation_list = filter_duplicates_and_special_words(
            (stopword_words_list, place_words_list, occupation_exclude_words_list),
            occupation_list)

        # sort if necessary
        if sort == 'Asc':
            occupation_list.sort()
        elif sort == 'Desc':
            occupation_list.sort(reverse=True)
        elif sort is None:
            pass    

        return (occupation_list, _OCCUPATION_FIELDNAMES)

    def get_person_list(self, include_none_dates: bool = False,
                              sort_by_birthdate: bool = False,
                              include_empty_residence: bool = False,
                              include_empty_occupation: bool = False) -> (list, list):
        _PERSON_FIELDNAMES = ("Person Handle", "Person ID", "Names List",
                              "Gender", "Birth Date", "Occupation",
                              "Residence", "Relatives List")

        def get_key_birth_date_sort_value(person_tuple):
            birth_date = person_tuple[COL_PERSON_BIRTH_DATE]
            if birth_date:
                birth_date_str = birth_date[0]
            else:
                # if birth_date is None set the birth_date_str to an empty string
                # otherwise the person_list cannot be sorted while comparing None with a string
                birth_date_str = '' 
            # return get_date_sort_value(birth_date_str)
            birth_date_sort_value = get_date_sort_value(birth_date_str)
            return birth_date_sort_value

        # TODO Decide whether get birth_event_list and/or family_list
        # could be done once. For instance in load function instead of
        # every time a person_list has to be delivered

        # get birth_event_list
        birth_event_list = self._get_birth_event_list()
        # get family_list
        family_list = self.get_family_list()

        person_list = []
        for person_elem in self.familytree_root.findall(self.xmlns+"people"+"/"+self.xmlns+"person"):
            # get handle
            handle = person_elem.get('handle')
            # get id
            id = person_elem.get('id')
            # get gender
            gender = None
            gender_elem_list = person_elem.findall(self.xmlns+"gender")
            if gender_elem_list:
                gender = gender_elem_list[0].text
            # get name_list
            name_dict = tuple()
            name_elem_list = person_elem.findall(self.xmlns+"name")
            for name_elem in name_elem_list:
                name = None
                # get prefix(es) and surname(s)
                prefix = None
                surname = None
                surname_elem_list = name_elem.findall(self.xmlns+"surname")
                for surname_elem in surname_elem_list:
                    # prefix = surname_elem_list[0].get('prefix')
                    # surname = surname_elem_list[0].text
                    prefix = surname_elem.get('prefix')
                    surname = surname_elem.text
                    # set name (= first + [prefix + surname])
                    if prefix:
                        if name:
                            name = name + " " + prefix
                        else:
                            name = prefix
                    if surname:
                        if name:
                            name = name + " " + surname
                        else:
                            name = surname
                if name:
                    name_dict = name_dict + (name,)

            # get attributes Occupation and Residence
            occupation = None
            residence = None
            attribute_elem_list = person_elem.findall(self.xmlns+"attribute")
            for attribute_elem in attribute_elem_list:
                # attributes of any type, so also the Occupation and Residence
                # types, could technically occur more than once. But these
                # are, in principle, imput only once. If not (arbitrarely)
                # took the last. 
                attribute_type = attribute_elem.get('type')
                attribute_value = attribute_elem.get('value')
                if attribute_type == 'Beroep':
                    occupation = attribute_value
                elif attribute_type == 'Woonplaats':
                    residence = attribute_value

            # Depending on the parameters include_empty_occupation and include_empty_residence 
            # and the occupation/residence value in- or exclude this person
            if (include_empty_occupation or (not include_empty_occupation and occupation)) and \
               (include_empty_residence or (not include_empty_residence and residence)):
                # get birth_date
                birth_date = None
                # get eventref_list
                eventref_list = get_handle_list(person_elem, self.xmlns, "eventref", "hlink")
                birth_event = None
                for eventref in eventref_list:
                    birth_event, idx = get_listitem_from_list_by_handle(birth_event_list, eventref, COL_EVENT_HANDLE)
                    if birth_event:
                        break
                if birth_event:
                    birth_date = (birth_event[COL_EVENT_DATEVAL_VAL], birth_event[COL_EVENT_DATEVAL_TYPE])

                # Depending on the parameter include_none_dates and the birth_date in- or exclude this person
                if include_none_dates or (not include_none_dates and birth_date):
                    # init relatives which will be filled from the persons childof_list and parentin_list
                    relatives = tuple()

                    # get childof_list
                    childof_family_list = get_handle_list(person_elem, self.xmlns, "childof", "hlink")
                    # get parent en brother/siter relative
                    for childof_family in childof_family_list:
                        family, idx = get_listitem_from_list_by_handle(family_list, childof_family, COL_FAMILY_HANDLE)
                        # Gramps is responsible for keeping the consistency of the
                        # database, so if there is a reference also a fanliy record
                        # SHOULD be found. But as a technical precaution we check
                        # on existence of it
                        if family:
                            father = family[COL_FAMILY_FATHER]
                            mother = family[COL_FAMILY_MOTHER]
                            childref_list = family[COL_FAMILY_CHILDREF_LIST]

                            if father:
                                # relatives.append([father, 'Ouder'])
                                relatives = relatives + ((childof_family, father, 'Vader'),)
                            if mother:
                                # relatives.append([mother, 'Ouder'])
                                relatives = relatives + ((childof_family, mother, 'Moeder'),)
                            for childref in childref_list:
                                # check whether the child is EQ to the person itself
                                if childref != handle:
                                    relatives = relatives + ((childof_family, childref, 'Broer/zus'),)

                    # get parentin_list
                    parentin_family_list = get_handle_list(person_elem, self.xmlns, "parentin", "hlink")
                    # get spouse and child relative
                    for parentin_family in parentin_family_list:
                        family, idx = get_listitem_from_list_by_handle(family_list, parentin_family, COL_FAMILY_HANDLE)
                        # Gramps is resopnisble for keeping the consistency of the
                        # database, so if there is a reference also a fanliy record
                        # SHOULD be found. But as a technical precaution we check
                        # on existence of it
                        if family:
                            father = family[COL_FAMILY_FATHER]
                            mother = family[COL_FAMILY_MOTHER]
                            childref_list = family[COL_FAMILY_CHILDREF_LIST]

                            spouse = None
                            # check whether the person is EQ to family's father
                            if handle == father:
                                if mother:
                                    spouse = mother
                                    # linktype_spouse = 'Echtgeno(o)t(e)'
                                    linktype_spouse = 'Vrouw'
                            # check whether the person is EQ to family's mother
                            if handle == mother:
                                if father:
                                    spouse = father
                                    # linktype_spouse = 'Echtgeno(o)t(e)'
                                    linktype_spouse = 'Man'
                            if spouse:
                                relatives = relatives + ((parentin_family, spouse, linktype_spouse),)
                            for childref in childref_list:
                                relatives = relatives + ((parentin_family, childref, 'Kind'),)

                    # add person data to the person_list
                    person_list.append((handle, id, name_dict, gender, birth_date, residence, occupation, relatives))

        # sort person_list
        if sort_by_birthdate:
            person_list.sort(key=get_key_birth_date_sort_value)
            
        return (person_list, _PERSON_FIELDNAMES)

    def get_connection_list(self, person_list: list,
                                  features: tuple, 
                                  linktype_mode: str = 'ByGender',
                                  name_similarity_mode: tuple = ('LevenshteinDistanceRelative', _LEVENSHTEIN_DISTANCE_THRESHOLD),
                                  n_random_conn_pp: int = 0,
                                  randomseed: int = None,
                                  include_none_dates: bool = False,
                                  max_abs_age_delta: int = ABS_AGE_DELTA_ONE_GENERATION,
                                  **kwargs_features) -> (list, list, list):
        """ person_list: input data
            features: tuple of features examined in the input and added as columns in the output
            linktype_mode: 'ByGender' | 'Neutral' (default: 'ByGender')
            n_randompp: int (default: 0)
            randomseed: int (default: None)
            include_none_dates: bool (default: False)
                Include connection for which the birth_date of one or both persons
                is None. In include such connection the Age Delta cound not be calculated.
            max_abs_age_delta: int (default: ABS_AGE_DELTA_ONE_GENERATION)
            **kwargs_features
        """

        # set feature object list
        mlfeature_list = self.get_mlfeature_list(features)

        # set fieldnames
        fieldnames = MAIN_LINK_PERSON_FIELDNAMES + (TARGET_FIELDNAME,)
        for mlfeature in mlfeature_list:
            fieldnames = fieldnames + (mlfeature.get_title(),)

        # check wether random connections has to be added
        if type(n_random_conn_pp) == int:
            random_connections_per_person = n_random_conn_pp
        else:
            random_connections_per_person = 0
        add_random_connections = random_connections_per_person > 0
        # if random connections has to be added, check whether random.seed has to be set
        if add_random_connections:
            random.seed(a=randomseed)

        # init loop
        n_person = len(person_list)
        connection_list = []
        person_connection_index_list = []
        n_total_connection = 0
        # loop over all persons in the person_list
        for mp_idx in range(n_person):
            mainperson = person_list[mp_idx]

            # get mainperson data
            mainperson_relatives_tuple = mainperson[COL_PERSON_RELATIVES_TUPLE]

            # -----------------------------
            # add known connections
            # -----------------------------

            # init the number of known connections for the current mainperson
            n_person_known_connection = 0

            # add all relatives as linkpersons from the maainperson
            for mainperson_relative in mainperson_relatives_tuple:
                # get the linkperson data from the person list
                linkperson, lp_idx = get_listitem_from_list_by_handle(person_list, mainperson_relative[COL_RELATIVE_PERSON_HANDLE], COL_PERSON_HANDLE)
                # a linkperson could not be found in the person_list for instance when
                # include_none_date = True and the birth_date of the linkperson is unknown
                if linkperson:
                    # get linktype between mainperson and linkperson
                    linktype = mainperson_relative[COL_RELATIVE_LINKTYPE]
                    # in person is "ByGender" the default setting for linktype
                    # map linktype to gender neutral omes in the case of taht linktype_mode
                    if linktype_mode.lower() == "neutral":
                        if linktype == "Vader":
                            linktype = "Ouder"                
                        elif linktype == "Moeder":
                            linktype = "Ouder"                
                        elif linktype == "Man":
                            linktype = "Echtgeno(o)t(e)"                
                        elif linktype == "Vrouw":
                            linktype = "Echtgeno(o)t(e)"

                    # create and add connection
                    connection = (mp_idx, lp_idx, linktype)
                    # calc all feature values for this specific connection_list
                    result = False
                    for mlfeature in mlfeature_list:
                        featurevalue, result = mlfeature.get_value(mainperson, linkperson, linktype,
                            name_similarity_mode=name_similarity_mode,
                            include_none_dates=include_none_dates,
                            max_abs_age_delta=max_abs_age_delta,
                            **kwargs_features)
                        if result:
                            connection = connection + (featurevalue,)
                        else:
                            # Stop the loop and dan't add the connection
                            break
                    # Only add the connections for which all features returns a valid value
                    if result:
                        connection_list.append(connection)
                        n_person_known_connection += 1

            # -----------------------------
            # add random connections
            # -----------------------------

            # init the number of random connections for the current mainperson
            n_person_random_connection = 0

            if add_random_connections:
                # Check whether the list is long enough to get the desired unique random items
                # Otherwise the while loop will not end in finding unique items
                # TODO the randomly chosen handle(s) could (or shuold) also be made unique
                if (random_connections_per_person - 1) < (n_person - len(mainperson_relatives_tuple)):
                    for conn_pp in range(random_connections_per_person):                 
                        random_person_handle = get_random_handle_from_list(person_list, COL_PERSON_HANDLE)
                        while random_person_handle in mainperson_relatives_tuple:
                            random_person_handle = get_random_handle_from_list(person_list, COL_PERSON_HANDLE)

                        # get the linkperson data from the person list
                        linkperson, lp_idx = get_listitem_from_list_by_handle(person_list, random_person_handle, COL_PERSON_HANDLE)
                        # a check on the existance of linkperson isn't necessary because
                        # it's chose from the available ones.
                        # get linktype between mainperson and linkperson
                        linktype = "Onbekend"

                        # create and add connection
                        connection = (mp_idx, lp_idx, linktype)
                        # calc all feature values for this specific connection_list
                        result = False
                        for mlfeature in mlfeature_list:
                            featurevalue, result = mlfeature.get_value(mainperson, linkperson, linktype,
                                name_similarity_mode=name_similarity_mode,
                                include_none_dates=include_none_dates,
                                max_abs_age_delta=max_abs_age_delta,
                                **kwargs_features)
                            if result:
                                connection = connection + (featurevalue,)
                            else:
                                # Stop the loop and dan't add the connection
                                break
                        # Only add the connections fro which all features returns a valid value
                        if result:
                            connection_list.append(connection)
                            n_person_random_connection += 1

            # update person connection index
            person_connection_index_list.append((
                n_total_connection, n_person_known_connection, n_person_random_connection))
            n_total_connection += n_person_known_connection + n_person_random_connection

        return (connection_list, fieldnames, person_connection_index_list)

    def get_personlink_list(self, person_list: list,
                            connection_list: list = None,
                            person_connection_index_list: list = None,
                            features: tuple = None, 
                            name_similarity_mode: tuple = ("LevenshteinDistanceRelative", _LEVENSHTEIN_DISTANCE_THRESHOLD),
                            include_none_dates: bool = False,
                            max_abs_age_delta: int = ABS_AGE_DELTA_ONE_GENERATION,
                            n_proc: int = -1,
                            **kwargs_features) -> list:
        """
            include_none_dates: bool (default: Fales)
                Include connection which for which the birth_date of one or both persons
                is None. In include such connection the Age Delta cound not be calculated.
            max_abs_age_delta: int (default: ABS_AGE_DELTA_ONE_GENERATION)
            n_proc: int (default: -1)
                Number of processors (if available) which has to be used to perform 
                this task. If set to -1 the maximum available processers is used. Value
                0 is treated as 1 and values <-1 as -1.
        """
        def get_personlink_args(mlfeature_list, mainperson, mp_idx: int, lp_idx: int,
                                linktype: str = None,
                                lp_idx_list: list = None,
                                name_similarity_mode: tuple = None,
                                include_none_dates: bool = False,
                                max_abs_age_delta: int = ABS_AGE_DELTA_ONE_GENERATION,
                                **kwargs_features):
            args = None
            # get linkperson by index
            linkperson = person_list[lp_idx]
            # get age delta in years
            result = False
            for mlfeature in mlfeature_list:
                if mlfeature.get_name().lower() == "agedelta":
                    age_delta, result = mlfeature.get_value(mainperson, linkperson, linktype,
                        name_similarity_mode=name_similarity_mode,
                        include_none_dates=include_none_dates,
                        max_abs_age_delta=max_abs_age_delta,
                        **kwargs_features)
                    if result:
                        # Exclude the connection of a person to himself/herself
                        # and links that belongs to the connection_list
                        if (mp_idx != lp_idx) and (lp_idx not in lp_idx_list):
                            args = (mlfeature_list,
                                    name_similarity_mode,
                                    include_none_dates,
                                    max_abs_age_delta,
                                    mp_idx, person_list[mp_idx],
                                    lp_idx, person_list[lp_idx],
                                    linktype, age_delta,
                                    kwargs_features)
                    break
            return args
        
        # set feature object list
        mlfeature_list = self.get_mlfeature_list(features)

        # set fieldnames
        fieldnames = MAIN_LINK_PERSON_FIELDNAMES + (TARGET_FIELDNAME,)
        for mlfeature in mlfeature_list:
            fieldnames = fieldnames + (mlfeature.get_title(),)

        n_cpu = multiprocessing.cpu_count()
        n_person = len(person_list)
        
        use_multiprocesses = (n_proc < 0) or (n_proc > 1) 
        if use_multiprocesses:
            args_list = []
        else:
            personlink_list = []    


        # temp
        now_begin = datetime.now()
        now_interval_begin = now_begin
        c_person = 0
       
       
        for mp_idx in range(n_person):
            # get mainperson by index
            mainperson = person_list[mp_idx]

            # get all the connections for the mainperson
            # get person_connection_index
            person_connection_index = person_connection_index_list[mp_idx]
            # get person_connection_index data
            conn_start_idx = person_connection_index[COL_PERSON_CONNECTION_INDEX_CONNSTARTIDX]
            n_known_conn = person_connection_index[COL_PERSON_CONNECTION_INDEX_NKNOWNCONN]
            n_rand_conn = person_connection_index[COL_PERSON_CONNECTION_INDEX_NRANDCONN]
            conn_end_idx = conn_start_idx+ n_known_conn + n_rand_conn
            mp_connection_sublist = connection_list[conn_start_idx:conn_end_idx]
            lp_idx_list = [connection[COL_CONNECTION_LINKINDEX] for connection in mp_connection_sublist]



            # temp
            c_person += 1
            if c_person % 1000 == 0:
                now_interval = datetime.now()
                print("{} + Processed persons for personlinks: {:,} - {:.3f} = {}".format(
                    now_interval - now_begin, c_person,
                    (now_interval.timestamp() - now_begin.timestamp()) / c_person,
                    now_interval - now_interval_begin))
                now_interval_begin = now_interval

            
            

            # from mp_idx search DOWNWARDS till max_abs_age_delta
            # (because pers_list should be sorted on birth_date the
            # for loop can be stopped when args is set to None)
            for lp_idx in range(mp_idx - 1, -1, -1):
                linktype = get_value_from_list_by_dict(mp_connection_sublist,
                    {COL_CONNECTION_LINKINDEX: lp_idx}, COL_CONNECTION_LINKTYPE)
                args = get_personlink_args(mlfeature_list, mainperson, mp_idx, lp_idx,
                                           linktype=linktype,
                                           lp_idx_list=lp_idx_list,
                                           name_similarity_mode=name_similarity_mode,
                                           include_none_dates=include_none_dates,
                                           max_abs_age_delta=max_abs_age_delta,
                                           **kwargs_features)
                if args:
                    if use_multiprocesses:
                        args_list.append(args)
                    else:
                        personlink = create_personlink(args)
                        if personlink:
                            # Include valid elements only
                            personlink_list.append(personlink)
                else:
                    break
            # from mp_idx search UPWARDS till max_abs_age_delta
            # (because pers_list should be sorted on birth_date the
            # for loop can be stopped when args is set to None)
            for lp_idx in range(mp_idx + 1, n_person, 1):
                linktype = get_value_from_list_by_dict(mp_connection_sublist,
                    {COL_CONNECTION_LINKINDEX: lp_idx}, COL_CONNECTION_LINKTYPE)
                args = get_personlink_args(mlfeature_list, mainperson, mp_idx, lp_idx,
                                           linktype=linktype,
                                           lp_idx_list=lp_idx_list,
                                           name_similarity_mode=name_similarity_mode,
                                           include_none_dates=include_none_dates,
                                           max_abs_age_delta=max_abs_age_delta,
                                           **kwargs_features)
                if args:
                    if use_multiprocesses:
                        args_list.append(args)
                    else:
                        personlink = create_personlink(args)
                        if personlink:
                            # Include valid elements only
                            personlink_list.append(personlink)
                else:
                    break

        if use_multiprocesses:
            if n_proc < 0:
                n_pool = n_cpu
            else:
                # n_proc > 1 (see check above)
                n_pool = min(n_proc, n_cpu)
            p = Pool(n_pool)
            personlink_list = p.map(create_personlink, args_list)
            # Filter None elements (which the mechanism of map can't exclude)
            personlink_list = [personlink for personlink in personlink_list if personlink != None]

        return (personlink_list, fieldnames)


###################################################################
#
# Example / Test
#
###################################################################

if __name__ == "__main__":
    now_begin = datetime.now()
    print(now_begin)

    # -------------------------------------------------------------------
    # Step 1: Create MLGrampsConnect object and load family tree (XML) 
    # -------------------------------------------------------------------
    
    # Get the current directory where this python is stored when run
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))

    # Set selected gramps xml backup
    gramps_filename = cur_dir_path + "/" + "gramps.xml"

    # Check whether the gramps xml backup file exists
    found = os.path.isfile(gramps_filename)
    if not found:
        sys.exit(gramps_filename + " is not found")
    
    # Init a MLGrampsObject and load a Gramps XML Backup with a familytree
    mlgc = MLGrampsConnect()
    mlgc.load(gramps_filename)

    # ---------------------------------------------------------------------
    # Step 2: Load occupation_table and supporting tables
    #         All "occupations" including a generalised form of it in
    #         the field "profession" and the kind of area of it in "sector"
    # ---------------------------------------------------------------------

    # Load occupation replacements
    occupation_replacement_table_csv = cur_dir_path + "/" + "occupation_replacement_table.csv"
    occupation_replacement_table, occupation_replacement_headings = import_list_from_csv(occupation_replacement_table_csv, has_heading=True)
    
    # Load stopwords
    stopword_table_csv = cur_dir_path + "/" + "stopword_table.csv"
    stopword_table, stopword_headings = import_list_from_csv(stopword_table_csv, has_heading=True)
    stopword_words_list = [stopword[0].lower() for stopword in stopword_table]

    # Load places
    place_table_csv = cur_dir_path + "/" + "place_table.csv"
    place_table, place_headings = import_list_from_csv(place_table_csv, has_heading=True)
    place_words_list = [place[0].lower() for place in place_table]

    # Load occupation excludes
    occupation_exclude_table_csv = cur_dir_path + "/" + "occupation_exclude_table.csv"
    occupation_exclude_table, occupation_exclude_headings = import_list_from_csv(occupation_exclude_table_csv, has_heading=True)
    occupation_exclude_words_list = [occupation_exclude[0].lower() for occupation_exclude in occupation_exclude_table]

    # Set occupation_table filename
    occupation_table_filename = cur_dir_path + "/" + "occupation_table.csv"
    # load occupation_table
    GLOBAL_occupation_table, GLOBAL_occupation_table_headings = import_list_from_csv(
        occupation_table_filename, has_heading=True, delimiter=';')

    # ---------------------------------------------------------------------
    # Option 2a: Get option_list occupation_list and save it as CSV file
    #            List of all occuring occupations
    #            This list is only be used to manually create an
    #            occupation_table which is stored also in a CSV file.
    #            This table is load an used in the following steps.
    # ---------------------------------------------------------------------

    # Get occupation_list
    occupation_list, occupation_list_fieldnames = mlgc.get_occupation_list(
        occupation_replacement_table=occupation_replacement_table,
        stopword_words_list=stopword_words_list,
        place_words_list=place_words_list,
        occupation_exclude_words_list=occupation_exclude_words_list,
        sort="Asc")
    print("{} | Number of occupations: {:,}".format(datetime.now() - now_begin, len(occupation_list)))

    # Save occupation_list
    occupation_list_csv = cur_dir_path + "/" + "occupation_list.csv"
    save_list_as_csv(occupation_list_csv,
                     occupation_list, occupation_list_fieldnames)
    print("{} | Filesize occupation_list.csv: {:,}".format(
        datetime.now() - now_begin, os.path.getsize(occupation_list_csv)))

    # ---------------------------------------------------------------------
    # Step 3: Get person_list 
    #         List of persons with all relevant data for feature generation
    # ---------------------------------------------------------------------
    
    # Get person_list
    person_list, person_list_fieldnames = mlgc.get_person_list(include_none_dates=False,
                                                               sort_by_birthdate=True,
                                                               include_empty_occupation=False,
                                                               include_empty_residence=False)
    print("{} | Number of persons: {:,}".format(datetime.now() - now_begin, len(person_list)))

    # ---------------------------------------------------------------------
    # Option 3a: Save person_list as CSV file
    # ---------------------------------------------------------------------

    person_list_csv = cur_dir_path + "/" + "person_list.csv"
    save_list_as_csv(person_list_csv,
                     person_list, person_list_fieldnames)
    print("{} | Filesize person_list.csv: {:,}".format(
        datetime.now() - now_begin, os.path.getsize(person_list_csv)))

    # ---------------------------------------------------------------------
    # Step 4: Init parameters different job settings (if necessary) 
    # ---------------------------------------------------------------------    

    # set features
    FEAT_GENDER_COMBINATION = "GenderCombination"
    FEAT_AGE_DELTA = "AgeDelta"
    FEAT_SURNAME_SIMILARITY = "SurnameSimilarity"
    FEAT_OCCUPATION_CORRESPONDENCE = "OccupationCorrespondence"
    FEAT_RESIDENCE_CORRESPONDENCE = "ResidenceCorrespondence"
    # FEAT_KNOWN_LINKTYPE = "KnownLinktype"
    FEAT_N_SIBLINGS_EQUALITY = "NumberOfSiblingsEquality"
    # set features_sets
    # features_sets = ((FEAT_GENDER_COMBINATION,
    #                   FEAT_AGE_DELTA,
    #                   FEAT_SURNAME_SIMILARITY,
    #                   FEAT_OCCUPATION_CORRESPONDENCE,
    #                   FEAT_RESIDENCE_CORRESPONDENCE
    #                  ),
    #                  (FEAT_GENDER_COMBINATION,
    #                   FEAT_AGE_DELTA,
    #                   FEAT_SURNAME_SIMILARITY,
    #                   FEAT_OCCUPATION_CORRESPONDENCE,
    #                   FEAT_RESIDENCE_CORRESPONDENCE,
    #                   FEAT_N_SIBLINGS_EQUALITY
    #                  )
    #                 )
    features_sets = ((FEAT_GENDER_COMBINATION,
                      FEAT_AGE_DELTA,
                      FEAT_SURNAME_SIMILARITY,
                      FEAT_OCCUPATION_CORRESPONDENCE,
                      FEAT_RESIDENCE_CORRESPONDENCE,
                      FEAT_N_SIBLINGS_EQUALITY
                     ),
                    )
    # temp_linktype_modes = ('ByGender', 'Neutral')
    temp_linktype_modes = ('ByGender',)
    # temp_n_random_conn_pps = (0, 1, 5)
    temp_n_random_conn_pps = (5,)

    for features_set in features_sets:
        for linktype_mode in temp_linktype_modes:
            for n_random_conn_pp in temp_n_random_conn_pps:     
            
                # ---------------------------------------------------------------------
                # Step 5: Get connection_list
                #         For ML model building (classification): feature list and
                #         target (LinkType) regarding relations between two persons.
                #         Including indices to persons in the person_list.
                # ---------------------------------------------------------------------

                kwargs_features = {'use_occupation_table': True,
                                   'occupation_replacement_table': occupation_replacement_table,
                                   'stopword_words_list': stopword_words_list,
                                   'place_words_list': place_words_list,
                                   'occupation_exclude_words_list': occupation_exclude_words_list,
                                   'occupation_table': GLOBAL_occupation_table}

                connection_list, connection_fieldnames, person_connection_index_list = \
                    mlgc.get_connection_list(person_list=person_list,
                                             features=features_set,
                                             linktype_mode=linktype_mode,
                                             name_similarity_mode=('LevenshteinDistanceRelative', 3),
                                             n_random_conn_pp=n_random_conn_pp,
                                             randomseed=None,
                                             max_abs_age_delta=ABS_AGE_DELTA_ONE_GENERATION,
                                             **kwargs_features)
                print("{} | Number of connections: {:,} - (Features: {}, Linktype: {}, n_random_conn_pp: {})".format(
                    datetime.now() - now_begin, len(connection_list),
                    len(features_set), linktype_mode, n_random_conn_pp))

                # ---------------------------------------------------------------------
                # Option 5a: Save connection_list and person_connection_index_list as
                #            CSV files
                # ---------------------------------------------------------------------

                connection_list_csv = cur_dir_path + "/" + "connection_list.csv"
                save_list_as_csv(connection_list_csv,
                                connection_list, connection_fieldnames)
                print("{} | Filesize connection_list.csv: {:,}".format(
                    datetime.now() - now_begin, os.path.getsize(connection_list_csv)))

                person_connection_index_list_csv = cur_dir_path + "/" + "person_connection_index_list.csv"
                save_list_as_csv(person_connection_index_list_csv,
                                person_connection_index_list, PERSON_CONNECTION_INDEX_FIELDNAMES)
                print("{} | Filesize person_connection_index_list.csv: {:,}".format(
                    datetime.now() - now_begin, os.path.getsize(person_connection_index_list_csv)))

                # ---------------------------------------------------------------------
                # Step 6: Get personlink_list
                #         For ML model operation (classification): feature list and
                #         known targets between any combination of persons within a
                #         maximum age difference.
                #         Including indices to persons in the person_list
                # ---------------------------------------------------------------------

                personlink_list, personlink_fieldnames = \
                    mlgc.get_personlink_list(person_list, connection_list,
                        person_connection_index_list,
                        features=features_set,
                        name_similarity_mode=('LevenshteinDistanceBool', 3),
                        include_none_dates=False,
                        max_abs_age_delta=ABS_AGE_DELTA_ONE_GENERATION, n_proc=-1,
                        **kwargs_features)
                print("{} | Number of personlinks: {:,}".format(
                    datetime.now() - now_begin, len(personlink_list)))
        
                # ---------------------------------------------------------------------
                # Option 6a: Save personlink_list as CSV file
                # ---------------------------------------------------------------------

                personlink_list_csv = cur_dir_path + "/" + "personlink_list.csv"
                save_list_as_csv(personlink_list_csv,
                                personlink_list, personlink_fieldnames)
                print("{} | Filesize personlink_list.csv: {:,}".format(
                    datetime.now() - now_begin, os.path.getsize(personlink_list_csv)))

