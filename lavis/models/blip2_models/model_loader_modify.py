import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter
import math

class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

         # if no centroids, by default just usual weight
        codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

def load_model(arch, code_length, num_cluster=30):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        relu_list = [1,3,6,8,11,13,15,18,20,22,25,27,29]
        for i in relu_list:
            model.features[i] = nn.ReLU(inplace=False)
        model.classifier = model.classifier[:-3]
        model.classifier[1] = nn.ReLU(inplace=False)

        model = ModelWrapper(model, 4096, code_length,num_cluster)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, image_embed=1408,hidden_state=768, num_cluster=30000):
        # num_cluster : 词典长度
        
        super(ModelWrapper, self).__init__()
        # self.model = model
        # self.code_length = code_length
        # self.hash_layer = nn.Sequential(
        #     nn.ReLU(inplace=False),
        #     nn.Linear(last_node, code_length),
        #     nn.Tanh(),)
        # self.hidden_layer = nn.Sequential(
        #     nn.Linear(last_node, last_node),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(last_node, last_node),
        #     nn.ReLU(inplace=False),
        # )
        self.club = CLUBSample(x_dim=hidden_state, y_dim=num_cluster, \
                            hidden_size=32)
        self.mine = MINE(x_dim=image_embed, y_dim=hidden_state, \
                         hidden_size=64)
        self.num_cluster=num_cluster
        # 信息熵的上界和下界
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(last_node*2, last_node*2),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(last_node*2, 3*224*224),
        #     nn.Tanh()
        # )
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(last_node, 256, bias=True),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(256, 2, bias=True),
        #     nn.Sigmoid()
        # )

        # Extract features
        self.extract_features = False
        # self.ce_fc = CosSim(code_length, num_cluster, learn_cent=False)
    
        self.beta = 0.1
        # self.beta=torch.nn.Parameter(torch.tensor(0.1).float(), requires_grad=True)


    def forward(self, vit, qformer,label_pre):
        # print('label_pre: ', label_pre)
        label = label_pre.masked_fill(
                label_pre == -100, 0
            ).to(torch.int64)
        # print('label: ', label)
        # print(label.view(label.shape[0]*label.shape[1]).shape)
        # print(label.view(label.shape[0]*label.shape[1]))
        # print(label.shape)
        # print(label.shape[0])
        # print(label.shape[1])
        # print(self.num_cluster)
        label_one_hot = torch.nn.functional.one_hot(label.view(label.shape[0]*label.shape[1]), self.num_cluster).to(label.device)
        # print(label_one_hot.shape)
        label_one_hot=label_one_hot.view(label.shape[0],label.shape[1],-1)
        I_zy = self.club.forward(qformer.float().mean(dim=1),label_one_hot.float().mean(dim=1)) # b*32*768  b*seq_length*dict
        I_xz = self.mine.forward(vit.float().mean(dim=1),qformer.float().mean(dim=1))  # b*257*1408 b*32*768
        # feature 是image embedding（VIT的输出）  
        # z是经过q-former之后的feature  （Q-former的输出）
        # y是label （batch * onehot）
        # y和z计算的时候可能需要合并两个维度，注意，seq这个维度要保持一致
        obj_loss = I_zy - self.beta*I_xz 
        # if abs(obj_loss) >1 :
        #     obj_loss=torch.Tensor([0]).to(label.device)
        return obj_loss

        """改了这里"""
        # if self.extract_features:
        #     return self.model(x)
        # feature = self.model(x)
        # code = self.hash_layer(feature)
        # logit = self.ce_fc(code)
        
        # z = self.hidden_layer(feature)
        # all = torch.cat([feature,z],dim=1)
        # x_rec = self.decoder(all)

        # domain_label = self.domain_classifier(feature)
        
        # if domain == 'source':
        #     I_zy = self.club.forward(z,y)
        #     I_xz = self.mine.forward(feature,z)  # x是image embedding  z是经过q-former之后的feature  y是label
        #     obj = I_zy - self.beta*I_xz  # 配系数加到LLM loss
             
        #     return logit, code, obj, x_rec, domain_label
        # else:
        #     return logit, code, x_rec, domain_label
        
    
    def get_parameters(self,domain = False):
        if domain:
            parameter_list = [{"params":self.domain_classifier.parameters(), "lr_mult":1, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameter_list

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag



class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        
        self.p_mu1=nn.Linear(x_dim, hidden_size)
        self.p_mu2=nn.Linear(hidden_size, y_dim)
        self.p_logvar1=nn.Linear(x_dim, hidden_size)
        self.p_logvar2=nn.Linear(hidden_size, y_dim)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def get_mu_logvar(self, x_samples): # b*3*768
        mu = self.p_mu1(x_samples) # b*3*768
        mu = self.relu(mu)
        mu = self.p_mu2(mu) # b*3*35572
        
        logvar = self.p_logvar1(x_samples) # b*3*768
        logvar = self.relu(logvar)
        logvar = self.p_logvar2(logvar) # b*3*35572
        logvar = self.tanh(logvar)

        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /(logvar.exp() + 1e-6)-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples): # b*3*768  b*3*35572
        mu, logvar = self.get_mu_logvar(x_samples) # b*3*35572  b*3*35572
        sample_size = x_samples.shape[0] # b
        random_index = torch.randperm(sample_size).long() # [1,2,3...b]
        positive = - (mu - y_samples)**2 / (logvar.exp() + 1e-6)
        negative = - (mu - y_samples[random_index])**2 / (logvar.exp() + 1e-6)
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        # T0 = T0.tanh() * 10
        # T1 = T1.tanh() * 10
        # lower_bound = T0.mean() - torch.log(T1.exp().mean() + 1e-6)
        
        T1 = T1.view(T1.shape[0])
        T1 = torch.logsumexp(T1, dim=0) - math.log(T1.shape[0])
        lower_bound = T0.mean() - T1

        # compute the negative loss (maximise loss == minimise -loss)
        # lower_bound = torch.clamp(lower_bound, 0, 10)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)



# test_class=ModelWrapper(num_cluster=30000)
# for name, parameter in test_class.named_parameters():
#     if 'weight' in name:
#         nn.init.xavier_normal_(parameter)
#     elif 'bias' in name:
#         nn.init.constant_(parameter, 0)
# # nn.init.normal_(test_class.weight, std=0.01)
# vit=torch.ones(16,257,1408)
# qformer=torch.ones(16,32,768)
# y=torch.ones(16,6)
# y=-y
# print(test_class)
# print(test_class(vit,qformer,y))
