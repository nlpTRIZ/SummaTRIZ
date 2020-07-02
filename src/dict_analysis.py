import numpy as np
import glob
import matplotlib.pyplot as plt

dict_list = sorted(glob.glob('../logs/transformation_bert/dict_data*.npy'))
dict_list.reverse()

second_mean_tp =[]
second_mean_fp =[]
second_mean_fn =[]
nb_tp=[]
nb_fp=[]
nb_fn=[]
loss_list= []


for reald in dict_list:
	second_read_dictionary = np.load(reald,allow_pickle='TRUE').item()
	loss = float(reald.split('/')[-1][10:14])

	nb_tp.append(len(second_read_dictionary['sent_tp']))
	nb_fp.append(len(second_read_dictionary['sent_fp']))

	second_mean_tp.append(np.mean(second_read_dictionary['probas_tp']))
	second_mean_fp.append(np.mean(second_read_dictionary['probas_fp']))
	second_mean_fn.append(np.mean(second_read_dictionary['probas_fn']))
	loss_list.append(loss)

print(second_mean_tp)
print(second_mean_fp)
print(second_mean_fn)

fig,ax = plt.subplots()
ax.plot(loss_list,second_mean_tp,label='mean_proba_tp')
ax.plot(loss_list,second_mean_fp,label='mean_proba_fp')
ax.plot(loss_list,second_mean_fn,label='mean_proba_fn')
plt.xlabel('Loss training')
ax.set_xlim(max(loss_list)+0.2, min(loss_list)-0.2)
ax.legend()
plt.show()

fig,ax = plt.subplots()
ax.plot(loss_list,nb_tp,label='nb_tp')
ax.plot(loss_list,nb_fp,label='nb_fp / nb_fn')

plt.xlabel('Loss training')
ax.set_xlim(max(loss_list)+0.2, min(loss_list)-0.2)
ax.legend()
plt.show()