import os
import numpy as np

list_dir = os.listdir('./')
#print (list_dir)


b = []

for mname in list_dir:
    if (mname.split("_")[0] == 'nconst32'):
        a = np.loadtxt('{}/acc.txt'.format(mname))
        b.append([a[0], mname.split("_")] )


b_sorted = sorted ([b[i] for i in range(len(b))], reverse=True)

for b in b_sorted:
    print (b)

