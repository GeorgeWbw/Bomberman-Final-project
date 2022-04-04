import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("scores4.txt", header = None)

plt.plot(data[0],data[1])
plt.xlabel("steps")
plt.ylabel("scores")
plt.show()
