import matplotlib.pyplot as plt
import tater
import numpy as np

x = np.linspace(-80,80,1000)
k = np.sqrt(2/3)

res = []

for xi in x:
    res.append(tater.sn_cn_dn_am(xi, k))

res = np.array(res)

plt.plot(x, res[:,-1])
plt.show()