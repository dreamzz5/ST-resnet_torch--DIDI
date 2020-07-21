import numpy as np
from  hmmlearn import  hmm
from process import *
TrainX, TrainY, TestX, TestY = process_data()
index=int(TrainX[0].shape[0]*0.875)
Train_c=TrainX[0][:index]
Train_c=Train_c*52
Train_c=Train_c.astype(int)
channel =2
# startprob=np.zeros([2,19,18,2])
# transmat =np.zeros([2,19,18,2,2])
# emissionprob = [[] for i in range(19*18)]
for v in range(channel):
    for i in range(Train_c.shape[2]):
        for j in range(Train_c.shape[3]):
            X=np.concatenate([Train_c[:,v,i,j][:,np.newaxis],Train_c[:,v+2,i,j][:,np.newaxis]
                                 ,Train_c[:,v+4,i,j][:,np.newaxis]],axis=1).T
            model=hmm.MultinomialHMM(n_components=2,n_iter=100,tol=1e-5)
            if np.sum(X)==0:
                X[np.random.randint(X.shape[0]),np.random.randint(X.shape[1])]=1
            model.fit(X)
            model.startprob_
            model.transmat_
            model.emissionprob_
            Z=model.predict(X[:,0].reshape(-1,1))
            print(Z)
            print([i,j])
