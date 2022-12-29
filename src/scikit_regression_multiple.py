import pandas as pd
import numpy as np
from sklearn import linear_model
import math

data_frame = pd.read_csv('./data_set/HiringProcess.csv')
print(data_frame.columns)