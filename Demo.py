#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import weka.core.jvm as jvm
import pandas as pd
import numpy as np
import csv

jvm.start(packages=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

from weka.core.converters import Loader
l = Loader("weka.core.converters.ArffLoader")
# d = l.load_file("C:\Users\Paolo\Documents\school stuff\4th Year\for Implementation\dataset_Updated.arff")
d = l.load_file("C:\Program Files\Weka-3-8\data\dataset_new.arff")
#print(d)


# In[2]:


import weka.core.jvm as jvm
import weka.core.serialization as serialization
from weka.classifiers import Classifier
data = d
data.class_is_last()
objects = serialization.read_all("NBTree_trained.model")
classifier = Classifier(jobject=objects[0])


# In[3]:


#classifier.classify_instance(inst=data.get_instance(index=4))


# In[4]:


from weka.core import dataset
from weka.core.dataset import Instance


# In[5]:


age, gender, mar_stat, ocd_hist, q2, q5, q10, q12, q13, q15,q17 = input("Input list here : ").split(" ")


# In[6]:


x = [age, gender, mar_stat, ocd_hist, q2, q5, q10, q12, q13, q15, q17]
x.append(Instance.missing_value())
data.add_instance(inst=Instance.create_instance(x))
classify = classifier.classify_instance(inst=data.get_instance(index=data.num_instances - 1)) 
if (classify == 0.0):
    print("No OCD")
else:
    print("OCD")


# In[7]:


#print(data)


# In[8]:


jvm.stop()


# In[ ]:




