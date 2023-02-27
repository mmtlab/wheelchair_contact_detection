import os
import csv
import numpy as np
import timer

def test_import():
    '''
    To simply test the import

    Returns
    -------
    None.

    '''
    print('import of BASIC package successfull!')

def is_list(input):
    return(isinstance(input,list))

def is_npArray(input):
    return(isinstance(input,np.ndarray))

def is_listOfList(input):
    '''execution time: around 360 ms, not depending on length of array'''
    if input == []:
        return False
    return is_list(input) and all(isinstance(el, list) for el in input)

def is_listOfNpArray(input):
    '''execution time: around 450 ms, not depending on length of array'''
    if input == []:
        return False
    return is_list(input) and all(isinstance(el, np.ndarray) for el in input)

def is_list_containing_lists_or_npArray(input):
    if input == []:
        return False
    return is_list(input) and all(isinstance(el, list) or isinstance(el, np.ndarray) 
                                  for el in input)

def is_npArray_containing_npArray(input):
    return is_npArray(input) and all(isinstance(el, np.ndarray) for el in input)

def is_emptyList_or_emptyNpArray(input):
    if is_list(input) and input != []:
        return False
    elif is_npArray(input) and np.any(input):
        return False
    else:
        return True

def get_length(arrayOrScalar):
    if np.isscalar(arrayOrScalar):
        return 0
    else:
        return len(arrayOrScalar)

def make_list(input):
    if not isinstance(input, list):
        return [input]
    else:
        return input

def make_listOfList(input):
    if not (is_listOfList(input)):
        return [input]
    else:
        return input

def make_listOfNpArray(input):
    if not (is_listOfNpArray(input)):
        return [input]
    else:
        return input

def make_listOfList_or_listOfNpArray(input):
    if not (is_listOfList(input) or is_listOfNpArray(input) or 
            is_list_containing_lists_or_npArray(input)):
        return [input]
    else:
        return input

def list_files_in_this_dir(directory):
    '''
    Returns a list containing the complete path to all the files contained in 
    the given directory
    '''
    tmp = os.listdir(directory)
    if tmp is None:
        return []
    tmp = make_list(tmp)
    tmp = [os.path.join(directory, item) for item in tmp]
    files = [item for item in tmp if os.path.isfile(item)]
    return files

def count_files_in_this_dir(directory):
    return len(list_files_in_this_dir(directory))

def count_files_in_dirs_inside_this_dir(directory):
    ans = []
    for d in list_dirs_in_this_dir(directory):
        dirName = os.path.split(d)[1]
        n = count_files_in_this_dir(d)
        ans.append([dirName, n])
    return ans

def list_files_in_these_dirs(listDirectories):
    '''
    Returns a list containing the complete path to all the files contained in 
    the given directories
    '''
    files = []
    listDirectories = make_list(listDirectories)
    for directory in listDirectories:
        files.extend(list_files_in_this_dir(directory))
    return files

def list_dirs_in_this_dir(directory):
    '''
    Returns a list containing the complete path to all the directories contained 
    in the given directory
    '''
    tmp = os.listdir(directory)
    if tmp is None:
        return []
    tmp = make_list(tmp)
    tmp = [os.path.join(directory, item) for item in tmp]
    dirs = [item for item in tmp if os.path.isdir(item)]
    return dirs

def list_dirs_deep_this_dir(directory, maxDepth):
    '''
    Iterates inside the directories of directory until maxDepth is reached and returns a list containing the complete path to all the found directories
    '''
    search_dirs = make_list(directory)
    found_dirs = []

    incr = 1
    if maxDepth == -1:
        incr = -1
    counter = -1
    while counter <= maxDepth:
        counter += incr
        new_dirs = []
        for search_dir in search_dirs:
            tmp = list_dirs_in_this_dir(search_dir)
            new_dirs.extend(tmp)
        if new_dirs == []:
            return found_dirs
        found_dirs.extend(new_dirs)
        search_dirs = new_dirs
    return found_dirs

def count_exceding_char(ofThisString, wrtToThisString, char):
    return ofThisString.count(char) -  wrtToThisString.count(char)

def is_correct_depth(ofThisPath, wrtToThisPath, depth):
    return count_exceding_char(ofThisPath, wrtToThisPath, '\\') == depth+1

def is_partial_name_inside(partialName, thisString):
    return partialName in thisString

def is_correct_extension(ext, thisString):
    return thisString.endswith(ext)

def filter_list_partialName(listOfPaths, listPartialName, logic = 'AND'):
    '''
    Given a list of strings, returns a list with all the strings whose name is contains
    - at least one of the string in listPartialName (if filterPartNameLogic == 'OR')
    - all the strings in listPartialName (if filterPartNameLogic == 'AND')
    '''
    assert logic in ['AND', 'OR'], f"logic should be AND or OR, got: {logic}"
    listOfPaths = make_list(listOfPaths)
    listPartialName = make_list(listPartialName)
    # AND: if at least one condition is NOT satisfied, remove from valid list
    if logic == 'AND': 
        validPaths = listOfPaths.copy()
        for partialName in listPartialName:
            for path in validPaths.copy():
                if partialName not in path:
                    validPaths.remove(path)
    # OR: if at least one condition is satisfied, add to valid list
    elif logic == 'OR': 
        validPaths = []
        for partialName in listPartialName:
            for path in listOfPaths:
                if partialName in path:
                    validPaths.append(path)
    return validPaths

def filter_list_extension(listOfPaths, listExtension):
    ''' 
    Returns a list with all the files whose extension is one of the value in listExtension
    '''    
    listOfPaths = make_list(listOfPaths)
    listExtension = make_list(listExtension)
    
    # only works in OR condition: impossible for one file to have two extensions at the same time    
    validPaths = []
    for extension in listExtension:
        for path in listOfPaths:
            if path.endswith(extension):
                validPaths.append(path)
    return validPaths

def filter_list_depth(listOfPaths, mainPath, listDepth):
    '''
    Returns a list with all the paths whose depth wrt to mainPath is equal to 
    one of the value in listDepth 
    If 0, searches only in the specified folder
    If 1, searches only in the folders inside the folder
    If [0,1], searches only in the specified folder and its subfolders
    '''
    listOfPaths = make_list(listOfPaths)
    listDepth = make_list(listDepth)

    # only works in OR condition: impossible for one file to have two depths at the same time
    validPaths = []
    for depth in listDepth:
        for path in listOfPaths:
            if is_correct_depth(path, mainPath, depth):
                validPaths.append(path)
    return validPaths

def remove_duplicates_from_list(myList):
    return list(dict.fromkeys(myList))

def remove_duplicates_from_list_of_list(myListOfList):
    newListOfList = []
    for l in myListOfList:
        if l not in newListOfList:
            newListOfList.append(l)
    return newListOfList

def remove_elements_already_in_list2(list1, list2):
    newList1 = []
    for l in list1:
        if l not in list2:
            newList1.append(l)
    return newList1

def merge_lists_OR(listOfLists):
    '''
    Returns a list with all the elements contained in at least one of the lists 
    without repetition
    '''
    listOfLists = make_listOfList(listOfLists)
    list_all = []
    for l in listOfLists:
        for e in l:
            list_all.append(e)
    return remove_duplicates_from_list(list_all)

def merge_lists_AND(listOfLists):
    '''
    Returns a list with only the elements contained in each one of the lists
    '''
    listOfLists = make_listOfList(listOfLists)
    first_list = listOfLists[0].copy()
    for l in listOfLists[1:]:
        for el in first_list.copy():
            if not el in l:
                first_list.remove(el)
    return first_list

def merge_lists_logic(logic, listOfLists):
    if logic == 'AND':
        return merge_lists_AND(listOfLists)
    elif logic == 'OR':
        return merge_lists_OR(listOfLists)
    else:
        raise Exception('logic in merge_lists_condition should be AND or OR, got {}'.format(logic))
    
def filter_dirs_in_list(dirList, mainDir, listDepth, listPartialName, filterPartNameLogic='AND'):
    '''
    Given a list of directories, returns a list of directories that meet the requirements:
    - their depth wrt to mainDir is equal to one of the values in listDepth
    - their complete path contains: 
        - one of the string in listPartialName (if filterPartNameLogic == 'OR')
        - all the strings in listPartialName (if filterPartNameLogic == 'AND')

    _extended_summary_

    Parameters
    ----------
    dirList : _type_
        _description_
    mainDir : _type_
        _description_
    listDepth : _type_
        _description_
    listPartialName : _type_
        _description_

    Returns
    -------
    string
        contains the valid directories
    '''
    listDepth = make_list(listDepth)
    listPartialName = make_list(listPartialName)
    if listDepth == [-1]:
        valid_dirs_depth = dirList
    else:
        valid_dirs_depth = filter_list_depth(dirList, mainDir, listDepth)
    valid_dirs_partialName = filter_list_partialName(dirList, listPartialName, 
                                                     filterPartNameLogic)
    valid_dirs = merge_lists_logic('AND', [valid_dirs_depth, valid_dirs_partialName])
    return valid_dirs

def filter_files_in_list(dirList, listExt, listPartialName, filterPartNameLogic='AND'):
    '''
     Given a list of directories, returns a list of the files that meet the requirements:
    - their extension is equal to one of the values in listExt
    - their complete path contains: 
        - one of the string in listPartialName (if filterPartNameLogic == 'OR')
        - all the strings in listPartialName (if filterPartNameLogic == 'AND')

    _extended_summary_

    Parameters
    ----------
    dirList : _type_
        _description_
    listExt : _type_
        _description_
    listPartialName : _type_
        _description_

    Returns
    -------
    string
        contains the valid files

    '''
    listExt = make_list(listExt)
    listPartialName = make_list(listPartialName)
    valid_files_ext = filter_list_extension(dirList, listExt)
    valid_files_partialName = filter_list_partialName(dirList, listPartialName, 
                                                      filterPartNameLogic)
    valid_files = merge_lists_logic('AND', [valid_files_ext, valid_files_partialName])
    return valid_files

def print_files_and_dirs(listFilesFound, listDirsFound):
    print('Found files: ')
    for this_file in listFilesFound:
        print(this_file)
    print('-'*10)
    print('Found dirs: ')
    for this_dir in listDirsFound:
        print(this_dir)
    print('-'*10)

def find_files_and_dirs_in_dir(directory, listDepth = [0], listExt = [''], 
    listPartialName = [''], filterPartNameLogic = 'AND', onlyDirs = False, 
    sortOutput = 1, printOutput = False):
    '''
    Given a directory, returns two lists containing the complete paths to every file and to every directory contained for all the depths specified in listDepth.
    If searching files, the extension can be specified in listExt (use "." as first character).
    If searching files or folders, part of the name can be specified in listPartialName. 
    If using filterPartNameLogic == 'AND', only the names containing each partial name specified in list will be considered
    If using filterPartNameLogic == 'OR', only the names containing at least one partial name specified in list will be considered

    onlyDirs can be set to True if searching only for folders to speed up the process

    sortOutput allows to sort the list in output

    printOutput allows to print the output

    Parameters
    ----------
    directory : string
        complete path of the main directory
    listDepth : list, optional
        list of depth (of subfolders) where the files and the dirs are searched, by default 1
        If 0, searches only in the specified folder
        If 1, searches only in the folders inside the folder
        If [0,1], searches only in the specified folder and its subfolders
        If -1, searches iteratively in all the possible subfolders
        by default [0] (only inside the directory specified)
    listExt : list, optional
        list of possible extensions when searching the files, 
        by default [''] (nothing excluded)
    listPartialName : str, optional
        the search excludes all the files and folders not containing it, 
        by default [''] (nothing excluded)
    filterPartNameLogic : str, optional
        If using filterPartNameLogic == 'AND', only the names containing each partial name specified in list will be considered
        If using filterPartNameLogic == 'OR', only the names containing at least one partial name specified in list will be considered
        by default 'AND' (all the partial names should be in the path)
    onlyDirs : bool, optional
        If True, both files and directories are searched
        by default False (both files and dirs are in output)
    sortOutput : bool, optional
        If 1, sorts all the two lists of found files and dirs
        If -1, sorts all the two lists of found files and dirs in reverse
        by default 1
    printOutput : bool, optional
        If True, prints all the found files and dirs, by default False

    Returns
    -------
    tuple 
        of 2 lists containing valid_files and valid_dirs
    '''

    listExt = make_list(listExt)
    listDepth = make_list(listDepth)
    listPartialName = make_list(listPartialName)

    dirs = list_dirs_deep_this_dir(directory, max(listDepth))
    valid_dirs = filter_dirs_in_list(dirs, directory, listDepth, listPartialName, filterPartNameLogic)

    if onlyDirs:
        valid_files = []
    else:
        if listDepth == [-1]:
            # if all directoreis
            dirs_for_files = dirs
            dirs_for_files.append(directory)
        elif listDepth == [0]:
            dirs_for_files = [directory]
        else:
            dirs.append(directory)
            # dirs for files are one level shallower
            dirs_for_files = filter_dirs_in_list(dirs, directory, list(np.array(listDepth)-1), '', filterPartNameLogic)
        files = list_files_in_these_dirs(dirs_for_files)
        valid_files = filter_files_in_list(files, listExt, listPartialName, filterPartNameLogic)
    
    if sortOutput == 1:
        valid_dirs.sort()
        valid_files.sort()
    elif sortOutput == -1:
        valid_dirs.sort(reverse = True)
        valid_files.sort(reverse = True)

    if printOutput:
        print_files_and_dirs(valid_files, valid_dirs)
    
    return (valid_files, valid_dirs)

def write_row_csv(CSVfile, newRow, mode = 'a'):
    '''
    Writes newRow in the csv file specified in CSVfile

    Parameters
    ----------
    CSVfile : string
        complete path to the csv file.
    newRow : list
        row to be added.

    Returns
    -------
    None.

    '''
    # eventually create the csv folder
    os.makedirs(os.path.split(CSVfile)[0], exist_ok= True)
    f = open(CSVfile, mode, encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(newRow)
    f.close()

def write_rows_csv(CSVfile, rows, mode = 'a'):
    '''
    Writes newRow in the csv file specified in CSVfile

    Parameters
    ----------
    CSVfile : string
        complete path to the csv file.
    newRow : list
        row to be added.

    Returns
    -------
    None.

    '''
    # eventually create the csv folder
    os.makedirs(os.path.split(CSVfile)[0], exist_ok= True)
    f = open(CSVfile, mode, encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerows(rows)
    f.close()
    
def chose_option_list(listOfOptions):
    
    for i, opt in zip(range(len(listOfOptions)), listOfOptions):
        print('{:02d} - {}'.format(i, opt))
    print('{:02d} - {}'.format(-1, 'None')) 
    while True:
        ans = input('choice: ')
        try:
            ans = int(ans)
            if ans <= len(listOfOptions) and ans >= 0:
                choice = listOfOptions[ans]
            elif ans == -1:
                choice = None
            print('[{}] was chosen'.format(choice))
            return choice
        except:
            pass
        print('not valid input')

def chose_TF(question = 'True or False?'):    
    while True:
        ans = input('{} [f, n, 0] or [t, y, 1]: '.format(question))
        try:
            if ans.lower() in ['t', 'true', 'y', 'yes', '1']:
                return True
            elif ans.lower() in ['f', 'false', 'n', 'no', '0']:
                return False
            elif ans == '-1':
                return None
        except:
            pass
        print('not valid input')
    
    
        
#%% just to figure out how does it work
if __name__ == '__main__':
    mainDir = os.getcwd()

    timing = timer.Timer()

    files, dirs = find_files_and_dirs_in_dir(mainDir, listDepth = [-1], listExt = ['.py'], listPartialName = '', filterPartNameLogic = 'AND', printOutput = True)

    timing.stop()



    
