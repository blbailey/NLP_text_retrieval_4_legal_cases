import os
from gensim.models import Word2Vec
import codecs
import re
import jieba
import shutil
import errno

model_w2v=Word2Vec.load("C:\\Users\\baili\\PycharmProjects\\lawBL\\refers\\zh.bin")
model_w2v.init_sims(replace=False)

def find_chinese(document):
    document_zw=""
    for x in document:
        if re.search(u'[\u4e00-\u9fff]', x):
            document_zw=document_zw+x
    return document_zw

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        return
def get_all_path(path):
    list1=os.listdir(path)
    for l in list1:
        path1=os.path.join(path, l, "pos")
        path2=os.path.join(path, l, "neg")
        print(os.path.exists(os.path.dirname(path2)))
        if not os.path.exists(path2):
            print("running")
            try:
                os.makedirs(path2)
                print("we make it")
            except  OSError as exc:
                print("Ooh")
                if exc.errno!=errno.EEXIST:
                    print("suck!")
                    raise
        browser(path1,path2)
        remove(path1)

        path1=os.path.join(path, l, "pos")
        path2=os.path.join(path, l, "pos1")
        if not os.path.exists(path2):
            try:
                os.makedirs(path2)
            except  OSError as exc:
                if exc.errno!=errno.EEXIST:
                    raise
        browser(path1,path2)
        remove(path1)





def browser(path1,path2):
    """read the file,
     delete chinese symbols,
     jieba tokenization,
     check the existence in the vocabulary
     save them into a new folder"""
    for root, dirs, files in os.walk(path1):
        for i in files:
            dir=os.path.join(root,i)
            document_str_tmp=find_chinese(codecs.open(dir,'r','utf-8').read())
            file_tmp=codecs.open(os.path.join(path2,i), 'w', "utf-8")
            seg_total=""
            for seg in jieba.cut(document_str_tmp,cut_all=False):
                if seg in model_w2v.wv:
                    seg_total=seg_total+seg+" "
            seg_total=seg_total[:-1]
            file_tmp.write(seg_total)
            file_tmp.close()

# browser("C:\\Users\\baili\\PycharmProjects\\sets_copy\\docu")

get_all_path("C:\\Users\\baili\\PycharmProjects\\sets_copy\\docus")