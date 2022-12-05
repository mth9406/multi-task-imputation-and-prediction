from layers.graph_imputer import * 
import torch 
from torch import nn
from torch.nn import functional as F  

class PriorGraphLayer(nn.Module):

    def __init__(self, prior_adj, device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.prior_adj = (prior_adj*1.).to(device)
        self.device = device
        super().__init__()

    def forward(self): 
        return self.prior_adj

class PriorGraphImputer(nn.Module):
    def __init__(self, 
                num_features:int,
                graph_emb_dim:int, 
                num_labels:int,
                cat_vars_pos:list= [], 
                numeric_vars_pos:list= [],
                num_layers:int= 1,
                alpha:float= 3., 
                imp_loss_penalty:float = 1.,
                reg_loss_peanlty:float = 0.1,
                prior_adj = None, 
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                task_type: str = 'cls'
                ):
        super().__init__() 

        # layers 
        # graph sampling layer
        if prior_adj is None: 
            prior_adj = torch.ones((num_features, num_features), device= device)
        self.graph_sampling = PriorGraphLayer(prior_adj, device)
        
        # GraphAutoEncoder layer
        for i in range(num_layers-1):
            setattr(self, f'gc{i}', GraphConvolutionLayer(num_features, F.leaky_relu))
        setattr(self, f'gc{num_layers-1}', GraphConvolutionLayer(num_features, None))

        # prediction layer 
        self.prediction_head = PredictionHead(num_features, num_labels, numeric_vars_pos, cat_vars_pos, task_type)

        # loss
        self.mse_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

        # attributes 
        self.num_features = num_features
        self.num_labels = num_labels 
        self.cat_vars_pos = cat_vars_pos 
        self.numeric_vars_pos = numeric_vars_pos
        self.alpha = alpha
        self.prior_adj = prior_adj.to(device)

        self.device = device 
        self.task_type = task_type

        self.num_layers = num_layers 
        self.imp_loss_penalty = imp_loss_penalty 
        self.reg_loss_peanlty = reg_loss_peanlty
    
        self.soft_impute = SoftImpute(verbose= False)

    # batch {input, mask, label, complete_input}
    # input, mask = make_mask(complete_input)
    def forward(self, batch):
        # inputs
        batch_x, batch_mask = batch.get('input'), batch.get('mask')
        batch_x = torch.FloatTensor(self.soft_impute.fit_transform(batch_x.detach().cpu())).to(self.device)

        # graph sampling and making an adjacency matrix
        adj_mat = self.graph_sampling() # num_features x num_features 
        # adj_mat = torch.sigmoid(logits)
        adj_mat_norm = self.norm_adj(adj_mat)
        
        for i in range(self.num_layers-1):
            batch_x_recon = getattr(self, f'gc{i}')(batch_x, adj_mat_norm)
        batch_x_recon = getattr(self, f'gc{self.num_layers-1}')(batch_x, adj_mat_norm)

        batch_x_recon_num = torch.tanh(batch_x_recon[:, self.numeric_vars_pos]) if len(self.numeric_vars_pos) > 0 else None
        batch_x_recon_cat = torch.sigmoid(batch_x_recon[:, self.cat_vars_pos]) if len(self.cat_vars_pos) > 0 else None 

        # trim the inputs
        assert batch_x_recon_cat is not None or batch_x_recon_num is not None, "Inputs should not be None, one of categorical- or numeric- variable should be a proper input."
        if batch_x_recon_cat is not None and batch_x_recon_num is not None:
            batch_x_hat = torch.cat([batch_x_recon_num, batch_x_recon_cat], dim= 1) 
        elif batch_x_recon_cat is not None and batch_x_recon_num is None: 
            batch_x_hat = batch_x_recon_cat 
        elif batch_x_recon_num is not None and batch_x_recon_cat is None: 
            batch_x_hat = batch_x_recon_num

        y_hat = self.prediction_head(batch.get('complete_input')[:, self.numeric_vars_pos + self.cat_vars_pos]) 
        if self.task_type == 'regr': 
            y_hat = y_hat.ravel()
        
        return {
            'x_recon_num': batch_x_recon_num,
            'x_recon_cat': batch_x_recon_cat,
            'adj_mat': adj_mat,
            'x_imputed': batch_x_hat,
            'y_hat': y_hat
        }
    
    @torch.no_grad()
    def get_adj(self): 
        adj_mat = self.graph_sampling()
        # adj_mat = torch.sigmoid(logits) 
        # adj_mat = (adj_mat > 0.5) * 1. 
        return adj_mat.detach().cpu().numpy()

    def train_step(self, batch): 
        # returns the training loss 
        # (1) feed forward
        # with torch.set_grad_enabled(True)
        # out = {x_recon, adj_mat, x_imputed, y_hat}
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]) if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]) if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr' else self.cls_loss(out.get('y_hat'), batch.get('label'))
        
        # regularization loss 
        regularization_loss = None

        return {
            'num_imp_loss': num_imp_loss,
            'cat_imp_loss': cat_imp_loss,
            'prediction_loss': prediction_loss,
            'regularization_loss': None
        } 

    @torch.no_grad()
    def val_step(self, batch): 
        # with torch.no_grad()
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]) if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]) if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr' else self.cls_loss(out.get('y_hat'), batch.get('label'))

        # regularization loss 
        regularization_loss = None 

        # perf measure 
        perfs = get_perf_num(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr'\
                                 else get_perf_cat(out.get('y_hat'), batch.get('label'), self.num_labels)
        
        perfs['num_imp_loss'] = num_imp_loss
        perfs['cat_imp_loss'] = cat_imp_loss
        perfs['prediction_loss'] = prediction_loss
        perfs['regularization_loss'] = regularization_loss
        
        return perfs

    @torch.no_grad()
    def test_step(self, batch): 
        # with torch.no_grad()
        out = self.forward(batch)
        num, cat = out.get('x_recon_num'), out.get('x_recon_cat')
        
        # imputation loss 
        num_imp_loss = self.mse_loss(num, batch.get('complete_input')[:, self.numeric_vars_pos]).detach().cpu().item() if num is not None else float('nan')
        cat_imp_loss = self.bce_loss(cat, batch.get('complete_input')[:, self.cat_vars_pos]).detach().cpu().item() if cat is not None else float('nan')

        # prediction loss 
        prediction_loss = self.mse_loss(out.get('y_hat').ravel(), batch.get('label')).detach().cpu().item() if self.task_type == 'regr' \
            else self.cls_loss(out.get('y_hat'), batch.get('label')).detach().cpu().item()

        # regularization loss 
        regularization_loss = float('nan')
        # perf measure 
        perfs = get_perf_num(out.get('y_hat').ravel(), batch.get('label')) if self.task_type == 'regr'\
                                 else get_perf_cat(out.get('y_hat'), batch.get('label'), self.num_labels)
        
        perfs['num_imp_loss'] = num_imp_loss
        perfs['cat_imp_loss'] = cat_imp_loss
        perfs['prediction_loss'] = prediction_loss
        perfs['regularization_loss'] = regularization_loss

        return perfs

    def norm_adj(self, adj_mat): 
        return adj_mat/adj_mat.sum(dim=0, keepdim=True)

