import os

def dircheck(dirpath):
    dirpath = os.path.abspath(dirpath)
    dir_list = dirpath.split(os.path.sep)
    for i, val in enumerate(dir_list):
        if i == 0: continue
        dirpath_ = '/'.join(dir_list[:i+1])
        if os.path.isdir(dirpath_) is False:
            os.mkdir(dirpath_)