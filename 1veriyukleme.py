import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veri = pd.read_csv('veriler.csv') #dosya oku

boy = veri[['boy']]
print(boy)
boykilo = veri[['boy','kilo']]
#print(boykilo)

