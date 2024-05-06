import torch

class LinearModel:
    def __init__(self):
        self.w = None 
        self.w1 = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
            self.w1 = self.w
        return X@self.w
    
    def predict(self, X):
        s = self.score(X)
        return (s > 0).int()

class LogisticRegression(LinearModel):

    def sigma(self, val):
        return 1 / (1 + torch.exp(-val))

    def loss(self, X, y):
    
        s = self.score(X)
        inner = (-1*y*torch.log(self.sigma(s))) - ((1 - y) *torch.log(1- self.sigma(s)))
        return (1/X.shape[0]) * inner.sum()

    def grad(self, X, y):
        s = self.score(X)
        summation = (self.sigma(s) - y)@X
        return (1/X.shape[0]) * summation
    
    def hessian(self, X):
        s = self.score(X)
        tran_X = X.T
        d = torch.diag_embed(self.sigma(s) * (1 - self.sigma(s)))
        return tran_X@d@X
    
class NewtonOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, alpha):
        """
        Compute one step of the Logistic Regression update using the feature matrix X 
        and target vector y. 
        """
        new_w = alpha*torch.inverse(self.model.hessian(X))@self.model.grad(X,y)
        self.model.w -= new_w

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y, alpha, beta):
        """
        Compute one step of the Logistic Regression update using the feature matrix X 
        and target vector y. 
        """
        temp = torch.clone(self.model.w)
        new_w = alpha*self.model.grad(X,y) - beta*(self.model.w - self.model.w1)
        self.model.w1 = temp
        self.model.w -= new_w