
import copy
import numpy as np
import torch as torch

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

def mask_r(r, idx, shape):
    rt = np.zeros(shape)
    rt[0, idx] = r
    return rt

def NaiveGradient(x, 
                num_feat, 
                net, 
                num_classes=2, 
                overshoot=0.02, 
                max_iter=100, 
                bound_min=[], bound_max=[], bound_type=[], 
                alphas=[], feature_selector=None):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        x = x.cuda()
    f = net.forward(Variable(x, requires_grad=True)
                    ).data.cpu().numpy().flatten()
    I = f.argsort()[::-1]
    fk_hat = I[0]
    fk_i_hat = fk_hat

    x_adv = copy.deepcopy(x)
    w_min = np.zeros(num_feat)
    r_total = np.zeros(num_feat)
    x = Variable(x_adv, requires_grad=True)
    fs = net.forward(x)

    nb_iter = 0
    tol = 10e-8

    cont_flag = True
    best_dist_to_centroid = 0
    changed_nb_iter = 0

    fs[0, fk_hat].backward(retain_graph=True)
    grads = x.grad.data.cpu().numpy().copy()
    top_feat_idx = []

    while cont_flag == True:
        fs[0, fk_hat].backward(retain_graph=True)
        grads = x.grad.data.cpu().numpy().copy()

        if len(top_feat_idx) == 0:
            top_feat_idx = np.argsort(grads)[::-1][0]
            if feature_selector:
                top_feat_idx = feature_selector.select(top_feat_idx, num_feat)
            else:
                top_feat_idx = top_feat_idx[:num_feat]

        w_org = grads[0, top_feat_idx]
        r_min = np.inf

        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0, I[k]].backward(retain_graph=True)
            w_cur = x.grad.data.cpu().numpy().copy()[0, top_feat_idx]
            w_diff = w_cur - w_org
            f_diff = (fs[0, I[k]] - fs[0, fk_hat]).data.cpu().numpy()
            r_k = abs(f_diff)/(np.linalg.norm(w_diff.flatten()) + tol)
            if r_k < r_min:
                r_min = r_k
                w_min = w_diff

        r_min = (r_min+1e-4) * w_min / (np.linalg.norm(w_min) + tol)

        r_total = np.float32(r_total + r_min)
        mask_r_ = mask_r(r_total, top_feat_idx, x.shape)
        mask_r_ = mask_r(r_min, top_feat_idx, x.shape)

        try:
            pert_x = pert_x.data.cpu().numpy() + (1+overshoot)*mask_r_
        except:
            pert_x = x_adv.data.cpu().numpy() + (1+overshoot)*mask_r_

        if len(bound_type) > 0:
            for i in range(len(bound_type)):
                if bound_type[i] == True and i in top_feat_idx:
                    pert_x[:, i] = np.around(alphas[i]*pert_x[:, i])
        if len(bound_min) > 0:
            pert_x = np.maximum(pert_x, bound_min)
        if len(bound_max) > 0:
            pert_x = np.minimum(pert_x, bound_max)

        if use_cuda:
            pert_x = torch.from_numpy(pert_x).cuda().float()
        else:
            pert_x = torch.from_numpy(pert_x).float()

        x = Variable(pert_x, requires_grad=True)
        fs = net.forward(x)
        fk_i_hat = np.argmax(fs.data.cpu().numpy().flatten())
        nb_iter += 1

        if fk_i_hat != fk_hat:
            cont_flag = False

        if nb_iter > max_iter:
            cont_flag = False

    previous_pert_x = pert_x.cpu().numpy().copy()
    
    pert_x = previous_pert_x
    if use_cuda:
        pert_x = torch.from_numpy(pert_x).cuda().float()
    else:
        pert_x = torch.from_numpy(pert_x).float()
    x = Variable(pert_x, requires_grad=True)
    fs = net.forward(x)
    fk_i_hat = np.argmax(fs.data.cpu().numpy().flatten())

    pert_x = pert_x.data.cpu().numpy().flatten()
    return fk_hat, fk_i_hat, r_total, pert_x, top_feat_idx, nb_iter
