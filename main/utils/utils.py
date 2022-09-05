import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def make_missing(x, prob):
    """
    A function to make some random missing values.
    Returns a modified data with some missing values and a mask matrix
    The element of a mask matrix is 1 if the element of the modified data is not missing at
    the same position and otherwise 0. 
    
    # Parameters  
    x: data (torch type)         
    prob: probability of making missing values (0~1)        

    # Returns
    x_tilde: a modified matrix with some missing values      
    mask: a mask matrix indicating the index of "not" missing values.               
    """
    n, p = x.shape
    mask = torch.LongTensor(np.random.uniform(low=0, high=1, size= (n,p)) >= prob) # mask
    x_tilde= torch.masked_fill(x, mask==0, float('nan'))
    return x_tilde, mask

def make_missing_by_row(x, n_missing= 1): 
    """  
    A function to make some random missing values "by row".    
    Returns a modified data with some missing values and a mask matrix     
    
    # Parameters    
    x: data (torch type)       
    n_missing: the number of missing values to generate by row (1~p)         

    # Returns
    x_tilde: a modified matrix with some missing values                  
    None            
    """
    n, p = x.shape 
    if n_missing >= p: 
        n_missing = p-1 
    x_tilde = x.clone()
    for i in range(n): 
        na_idx = torch.randperm(p)[ :n_missing]
        x_tilde[i, na_idx] = float('nan')
    print(f'every observation has {n_missing} missing values out of {p}...')
    return x_tilde, None

def standardize(X_train):
    cache = {'mean':0, 'std':0}
    cache['mean'], cache['std'] = np.nanmean(X_train, axis= 0, keepdims= True), np.nanstd(X_train, axis= 0, keepdims= True)
    return (X_train-cache['mean'])/cache['std'], cache

def standardize_test(X_test, cache):
    return (X_test-cache['mean'])/cache['std']

def min_max_scaler(X_train):
    cache = {'min':0, 'max':0}
    cache['min'], cache['max'] = np.nanmin(X_train, axis= 0, keepdims= True), np.nanmax(X_train, axis= 0, keepdims= True)
    return 2*(X_train-cache['min'])/(cache['max']-cache['min'])-1, cache

def min_max_scaler_test(X_test, cache):
    return 2*(X_test-cache['min'])/(cache['max']-cache['min'])-1

def div0( a, b, fill=np.nan ):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) \
            else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

# model evaluation
def evaluate(cm, weighted= False):
    """
    Evaluates a model. 
    # Parameters
    cm: confusion matrix
    weighted: calculates weigted recall, precision and f1 score if True.
    returns accuracy, precision, recall and F1-score.
    """
    # cm
    # column: predicted class
    # row: true label
    acc, rec, prec, f1 = 0, 0, 0, 0
    
    diag = np.diag(cm)
    n_samples = np.sum(cm, axis= 1)
    n_preds = np.sum(cm, axis= 0)
    # accuracy
    acc = np.sum(diag)/np.sum(cm)
    # recall
    rec_ = div0(diag, n_samples, 1)
    rec = np.sum(rec_*n_samples)/np.sum(n_samples) if weighted else np.mean(rec_)
    # precision
    prec_ = div0(diag, n_preds, 1)
    prec = np.sum(prec_*n_samples)/np.sum(n_samples) if weighted else np.mean(prec_)
    # f1-score
    f1_ = div0(2*rec_*prec_,(rec_+prec_), 0)
    f1 = np.sum(f1_*n_samples)/np.sum(n_samples) if weighted else np.mean(f1_)

    return acc, rec, prec, f1