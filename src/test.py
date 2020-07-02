import pickle
import numpy as np

with open("list1.txt", "rb") as fp:   # Unpickling
	list1 = pickle.load(fp)

with open("list2.txt", "rb") as fp:   # Unpickling
	list2 = pickle.load(fp)



list1 = [min(list1[i]) for i in range(len(list1))]
list2 = [min(list2[i]) for i in range(len(list2))]


print(np.array(list2)-np.array(list1))