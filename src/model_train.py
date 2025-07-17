import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selction import KFold


def data_split(dataframe):
    volunteers = dataframe['volunteers'].unique()



