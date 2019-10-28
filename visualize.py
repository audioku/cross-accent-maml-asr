import matplotlib.pyplot as plt
import pandas as pd

arr1 = []
arr2 = []
indices = []

with open("loss.txt", "r") as file:
    for line in file:
        arr = line.split(",")
        indices.append(float(arr[0]))
        arr1.append(float(arr[1]))
        arr2.append(float(arr[2]))

df = pd.DataFrame({
'train': arr1,
'valid': arr2
}, index=indices)

lines = df.plot.line()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("aishell_loss.jpg")