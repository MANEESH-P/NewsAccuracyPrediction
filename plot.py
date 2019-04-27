import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
label = ['discuss', 'agree']
count = [12 , 2]
index = np.arange(len(label))
def plot_bar_x():
     index = np.arange(len(label))
     plt.bar(index, count)
     plt.xlabel('Stance', fontsize=5)
     plt.ylabel('No of Articles', fontsize=5)
     plt.xticks(index, label, fontsize=5, rotation=30)
     plt.show()



