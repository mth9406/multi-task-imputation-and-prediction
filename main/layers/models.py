import torch 
from torch import nn
from torch.nn import functional as F
from layers.layers import * 

class NaiveSoftImpute(nn.Module):
    r"""
    Implementation of SoftImpute + prediction
    * Set "n_labels" 1 to make a regression.
    """
    def __init__(self, input_size, n_labels, drop_p= 0.5, stack_fc_lyrs= False,):
        super().__init__()
        self.fc_out = nn.Sequential(
                nn.Linear(input_size, n_labels),
                nn.BatchNorm1d(n_labels),
                nn.ReLU(),
                nn.Linear(n_labels, n_labels)
            ) if stack_fc_lyrs else nn.Linear(input_size, n_labels)
        
        self.imput_dim = input_size
        self.n_labels = n_labels
        self.drop_p = drop_p    

    def forward(self, x, numobs= 1, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make prediction using x_hat

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        predicted labels (size= (bs, n_labels)), imputation
        """    
        # x_hat = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(x['input'].detach().cpu())).to(x['input'].device) 
        y_hat = self.fc_out(x['input'])
        if self.n_labels == 1: 
            y_hat = y_hat.flatten()
        out = {
            'preds': y_hat,
            'imputation':x['input'],
            'regularization_loss': None
        }
        return out

class LinearModel(nn.Module):
    r"""
    Implementation of a linear model
    * Set "n_labels" 1 to make a regression.
    """
    def __init__(self, input_size, n_labels, drop_p= 0.5, stack_fc_lyrs= False,):
        super().__init__()
        self.fc_out = nn.Sequential(
                nn.Linear(input_size, n_labels),
                nn.BatchNorm1d(n_labels),
                nn.ReLU(),
                nn.Linear(n_labels, n_labels)
            ) if stack_fc_lyrs else nn.Linear(input_size, n_labels)
        
        self.imput_dim = input_size
        self.n_labels = n_labels
        self.drop_p = drop_p    

    def forward(self, x, numobs= 1, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make prediction using x_hat

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        predicted labels (size= (bs, n_labels)), imputation
        """    
        # x_hat = torch.FloatTensor(SoftImpute(verbose= False).fit_transform(x['input'].detach().cpu())).to(x['input'].device) 
        y_hat = self.fc_out(x['input'])
        if self.n_labels == 1: 
            y_hat = y_hat.flatten()
        out = {
            'preds': y_hat,
            'imputation': x['input'],
            'regularization_loss': None
        }
        return out

class AutoEncoder(nn.Module):
    r"""
    Implementation of Auto-encoder for imputation
    """
    def __init__(self, input_size, drop_p=0.5, stack_fc_lyrs= False, stack_ae_lyrs= False):
        super().__init__()
        self.encoder_layer = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.decoder_layer = DecoderDoubleLayer(
             input_size//4, input_size//2, input_size, drop_p
        ) if stack_ae_lyrs else DecoderLayer(input_size//4, input_size)

        self.imput_dim = input_size
        self.drop_p = drop_p  

    def forward(self, x, numobs= 1, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        numobs is an dummy argument.

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make complement input x_hat_comp
        (3) calculates loss 

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        predicted labels (size= (bs, n_labels)), imputation
        """
        # x_hat = []
        # for j in range(x['input'].shape[1]): 
        #     m = torch.mean(x['input'][~x['input'][:, j].isnan(), j])
        #     x_hat.append(torch.masked_fill(x['input'][:, j], x['mask'][:, j] == 0, m)) 
        # x_hat = torch.stack(x_hat dim= -1)
        x_hat = x['input']
        # x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
        encoded = self.encoder_layer(x_hat)
        x_hat = self.decoder_layer(encoded)
        if not self.training:
            x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
            x_hat = self.reconstruct_categories(x_hat, cat_features)
        out = {
            'preds': None,
            'imputation':x_hat,
            'regularization_loss': None
        }
        return out
    
    def reconstruct_categories(self, x_hat, cat_features= None):
        if cat_features is None: 
            return x_hat 
        x_hat[:, cat_features] = (torch.sigmoid(x_hat[:, cat_features]) >= 0.5) * 1.
        return x_hat

class VariationalAutoEncoder(nn.Module):
    r"""
    Implementation of Variational Auto-encoder for imputation
    """
    def __init__(self, input_size, drop_p=0.5, stack_fc_lyrs= False, stack_ae_lyrs= False):
        super().__init__()
        self.encoder_layer_mu = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.encoder_layer_log_var = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.decoder_layer = DecoderDoubleLayer(
             input_size//4, input_size//2, input_size, drop_p
        ) if stack_ae_lyrs else DecoderLayer(input_size//4, input_size)    

        self.imput_dim = input_size
        self.drop_p = drop_p  

    def forward(self, x, numobs= 100, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        numobs: takes args.vai_n_samples as an argument.

        cat_features: list of indices of categorical features

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make complement input x_hat_comp
        (3) calculates loss 

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        impuated input
        """     
        x_hat = x['input']

        mu = self.encoder_layer_mu(x_hat)
        log_var = self.encoder_layer_log_var(x_hat)
        
        loss = None
        if self.training: 
            # x_hat_teacher = torch.clone(x_hat).to(device= x_hat.device)
            z, sigma_sq = self.reparameterize(mu, log_var)
            x_hat = self.decoder_layer(z)
            # x_hat = self.reconstruct_categories(x_hat, cat_features)
            x_hat_teacher = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
            # y_hat = self.fc_out(x_hat)
            loss = self.loss_regularization(mu, sigma_sq)        
        else: 
            x_hats = []
            for i in range(numobs):
                z, sigma_sq = self.reparameterize(mu, log_var)
                x_hat = self.decoder_layer(z)
                x_hat = self.reconstruct_categories(x_hat, cat_features)
                x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
                x_hats.append(x_hat)
            x_hats = torch.stack(x_hats, dim=0)
            # aggregates the 'outputs'
            # (1) x_hat
            # x_hats = torch.mean(x_hats, dim=0)
            x_hat = torch.quantile(x_hats, q= 0.5, dim= 0)
            if cat_features is not None: 
                x_hat[:, cat_features], _ = torch.mode(x_hats[..., cat_features], dim=0)
        out = {
            'preds': None,
            'imputation': x_hat,
            'regularization_loss': loss
            }
        return out

    def reparameterize(self, mu, log_var):
        sigma_sq = torch.exp(log_var)
        eps = torch.randn_like(sigma_sq).to(log_var.device)
        return mu + torch.sqrt(sigma_sq) * eps, sigma_sq
    
    def loss_regularization(self, mu, sigma_sq, eps= 1e-10):
        mu_sq = mu ** 2 
        # sigma_sq = sigma ** 2
        return torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq+eps))
    
    def reconstruct_categories(self, x_hat, cat_features= None):
        if cat_features is None: 
            return x_hat 
        if not self.training:
            x_hat[:, cat_features] = (torch.sigmoid(x_hat[:, cat_features]) >= 0.5) * 1.
        else: 
            x_hat[:, cat_features] = torch.sigmoid(x_hat[:, cat_features])
        return x_hat

    def make_imp_distribution(self, x, numobs= 100, cat_features= None): 
        self.training = False 
        out = self.forward(x, numobs, cat_features)
        return out

class AutoImpute(nn.Module):
    r"""
    Implementation of SoftImpute (initial guess) + Auto-encoder + prediction
    """
    def __init__(self, input_size, n_labels, drop_p=0.5, stack_fc_lyrs= False, stack_ae_lyrs= False):
        super().__init__()
        self.encoder_layer = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.decoder_layer = DecoderDoubleLayer(
             input_size//4, input_size//2, input_size, drop_p
        ) if stack_ae_lyrs else DecoderLayer(input_size//4, input_size)
        self.fc_out = nn.Sequential(
                nn.Linear(input_size, n_labels),
                nn.BatchNorm1d(n_labels),
                nn.ReLU(),
                nn.Linear(n_labels, n_labels)
            ) if stack_fc_lyrs else nn.Linear(input_size, n_labels)

        self.imput_dim = input_size
        self.n_labels = n_labels
        self.drop_p = drop_p  

    def forward(self, x, numobs= 1, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        numobs is an dummy argument.

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make complement input x_hat_comp
        (3) calculates loss 

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        predicted labels (size= (bs, n_labels)), imputation
        """
        # x_hat = []
        # for j in range(x['input'].shape[1]): 
        #     m = torch.mean(x['input'][~x['input'][:, j].isnan(), j])
        #     x_hat.append(torch.masked_fill(x['input'][:, j], x['mask'][:, j] == 0, m)) 
        # x_hat = torch.stack(x_hat dim= -1)
        x_hat = x['input']
        # x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
        encoded = self.encoder_layer(x_hat)
        x_hat = self.decoder_layer(encoded)
        if not self.training:
            x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
            x_hat = self.reconstruct_categories(x_hat, cat_features)
        if self.training: 
            x_hat_teacher = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
            y_hat = self.fc_out(x_hat_teacher)
        else:
            y_hat = self.fc_out(x_hat)
            # y_hat = torch.argmax(torch.softmax(y_hat, dim=-1), dim=-1)
        if self.n_labels == 1: 
            y_hat = y_hat.flatten()
        out = {
            'preds': y_hat,
            'imputation':x_hat,
            'regularization_loss': None
        }
        return out
    
    def reconstruct_categories(self, x_hat, cat_features= None):
        if cat_features is None: 
            return x_hat 
        x_hat[:, cat_features] = (torch.sigmoid(x_hat[:, cat_features]) >= 0.5) * 1.
        return x_hat

class IPV(nn.Module):
    r"""
    Implementation of SoftImpute (initial guess) + Variational Auto-encoder + prediction
    """
    def __init__(self, input_size, n_labels, drop_p=0.5, stack_fc_lyrs= False, stack_ae_lyrs= False):
        super().__init__()
        self.encoder_layer_mu = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.encoder_layer_log_var = EncoderDoubleLayer(
            input_size, input_size//2, input_size//4, drop_p
        ) if stack_ae_lyrs else EncoderLayer(input_size, input_size//4)
        self.decoder_layer = DecoderDoubleLayer(
             input_size//4, input_size//2, input_size, drop_p
        ) if stack_ae_lyrs else DecoderLayer(input_size//4, input_size)
        self.fc_out = nn.Sequential(
                nn.Linear(input_size, n_labels),
                nn.BatchNorm1d(n_labels),
                nn.ReLU(),
                nn.Linear(n_labels, n_labels)
            ) if stack_fc_lyrs else nn.Linear(input_size, n_labels)       

        self.imput_dim = input_size
        self.n_labels = n_labels
        self.drop_p = drop_p  

    def forward(self, x, numobs= 100, cat_features= None): 
        r"""
        The input 'x' has several keys:
        input, mask, label
        the firt two are 'features' while the last one is the target.

        numobs: takes args.vai_n_samples as an argument.

        cat_features: list of indices of categorical features

        Update procedures are as follows...
        (1) make soft impute: x_hat
        (2) make complement input x_hat_comp
        (3) calculates loss 

        # The size of the various inputs:
        |x['input']|: batch-size, input_size
        |x['mask']|: batch-size, input_size

        # The size of the outputs(label):
        |x['label']|: batch-size, 

        # Returns:
        predicted labels (size= (bs, n_labels))
        """     
        x_hat = x['input']

        mu = self.encoder_layer_mu(x_hat)
        log_var = self.encoder_layer_log_var(x_hat)
        
        loss = None
        if self.training: 
            # x_hat_teacher = torch.clone(x_hat).to(device= x_hat.device)
            z, sigma_sq = self.reparameterize(mu, log_var)
            x_hat = self.decoder_layer(z)
            # x_hat = self.reconstruct_categories(x_hat, cat_features)
            x_hat_teacher = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
            y_hat = self.fc_out(x_hat_teacher)
            # y_hat = self.fc_out(x_hat)
            loss = self.loss_regularization(mu, sigma_sq)        
        else: 
            x_hats = []
            y_hats = []
            for i in range(numobs):
                z, sigma_sq = self.reparameterize(mu, log_var)
                x_hat = self.decoder_layer(z)
                x_hat = self.reconstruct_categories(x_hat, cat_features)
                x_hat = torch.masked_fill(x['input'], x['mask']==0, 0) + x_hat * (1-x['mask'])
                x_hats.append(x_hat)
                y_hats.append(self.fc_out(x_hat))
            x_hats = torch.stack(x_hats, dim=0)
            y_hats = torch.stack(y_hats, dim=0)
            # aggregates the 'outputs'
            # (1) x_hat
            # x_hats = torch.mean(x_hats, dim=0)
            x_hat = torch.quantile(x_hats, q= 0.5, dim= 0)
            if cat_features is not None: 
                x_hat[:, cat_features], _ = torch.mode(x_hats[..., cat_features], dim=0)
            # (2) y_hat
            if self.n_labels == 1: 
                y_hat = torch.mean(y_hats, dim= 0) # n, 1
            else: 
                # y_hat, _ = torch.mode(torch.argmax(torch.softmax(y_hats, dim=2), dim=2), dim=0) # num_samples, n, 1 -> n, 1
                y_hat = y_hats # num_samples, n, num_classes
        if self.n_labels == 1: 
            y_hat = y_hat.flatten()
        out = {
            'preds': y_hat,
            'imputation': x_hat,
            'regularization_loss': loss
            }
        return out

    def reparameterize(self, mu, log_var):
        sigma_sq = torch.exp(log_var)
        eps = torch.randn_like(sigma_sq).to(log_var.device)
        return mu + torch.sqrt(sigma_sq) * eps, sigma_sq
    
    def loss_regularization(self, mu, sigma_sq, eps= 1e-10):
        mu_sq = mu ** 2 
        # sigma_sq = sigma ** 2
        return torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq+eps))
    
    def reconstruct_categories(self, x_hat, cat_features= None):
        if cat_features is None: 
            return x_hat 
        if not self.training:
            x_hat[:, cat_features] = (torch.sigmoid(x_hat[:, cat_features]) >= 0.5) * 1.
        else: 
            x_hat[:, cat_features] = torch.sigmoid(x_hat[:, cat_features])
        return x_hat

    def make_imp_distribution(self, x, numobs= 100, cat_features= None): 
        self.training = False 
        out = self.forward(x, numobs, cat_features)
        return out