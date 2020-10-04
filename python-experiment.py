# imports
#import sclblpy as sp
import numpy as np
from sklearn.linear_model import LinearRegression

# generate some data:
n = 100
x = np.random.uniform(0,10,(n,))

# y = 10 + 2x + noise:
y = 10 + 2*x + np.random.normal(0,1,(n,))

# fit a model (note the reshape of the vectors)
mod = LinearRegression()  
mod.fit(x.reshape(-1, 1), y.reshape(-1, 1))
