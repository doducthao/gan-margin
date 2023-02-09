import os


def remove(folder):
    flist = os.listdir(folder)
    for f in flist:
        fn = os.path.join(folder, f)
        if os.path.isfile(fn) and f != 'logging.log':
            os.remove(fn)
        elif os.path.isdir(fn):
            remove(fn)


remove(".")