
import fastai
import sklearn
import pylab as plt

# Spliting the data 
Train_Set = df_trn
Target_Value = y_trn

def data_split(a,n): 
    return a[:n], a[n:]

#declare variables
n_valid = len(df_trn)*.3
n_train = len(df_trn)-n_valid
X_train, X_valid = split(df_trn, n_trn)
y_train, y_valid = split(y_trn, n_trn)
raw_train, raw_valid = split(df_raw, n_trn)

class TreeEnsemble():
    def __init__(self, x,y,n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x= x
        self.y= y
        self.sample_sz=sample_sz
        self.min_leaf = min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    # Creating a tree 
    def create_tree(self):
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[rnd+idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)
    #

    #predict 
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)

# Decision tree constructor
# its takes x, y indxs and min_leaf
class DecisionTree():
    def __init__(self, x,y,indxs=None, min_leaf=5):
        self.x, self.y ,self.indxs,self.min_leaf=min_leaf= x,y,idxs,min_leaf
        self.n,self.c = len(idxs), x.shape[1]
        self

# model 
# m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
