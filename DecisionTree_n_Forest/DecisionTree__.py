import fastai
import sklearn
import pylab as plt

# Spliting the data
Train_Set = df_trn
Target_Value =  y_trn

def data_split(a, n):
    return a[:n], a[n:]


#declare variables
n_valid = len(df_trn)*.3
n_train = len(df_trn)-n_valid
X_train, X_valid = split(df_trn, n_trn)
y_train, y_valid = split(y_trn, n_trn)
raw_train, raw_valid = split(df_raw, n_trn)


class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)


class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):
        if idxs is None: idxs = np.arange(len(y))
         self.x, self.y, self.idxs, self.min_leaf= x,y,idxs,min_leaf
         self.n, self.c = len(idxs), x.shape[1]
         self.val = np.mean(y[idxs])
         self.score = float('int')
         self.find_varsplit()

    def find_varsplit(self):
        for i in range(self.c):
            self.find_better_split(i)

    def find_better_split(self, var_idx): 
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs]

        for i in range(self.n):
            lhs = x<=x[i]
            rhs = x>x[i]
            if rhs.sum()<self.min_leaf or lhs.sum()<self.min_leaf: continue
            lhs_std = y[lhs].std()
            rhs_std = y[rhs].std()
            curr_score = lhs_std*lhs.sum() + rhs_std*rhs.sum()
            if curr_score < self.score:
                self.var_idx,self.score,self.split = var_idx,curr_score,x[i]

    @property
    def split_name(self): return self.x.columns[self.var_idx]

    @property
    def split_col(self): return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self): return self.score == float('int')

    def __repr__(self):
        s = f'n:{self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split{self.split}; val:{self.split_name}'
        return s
