import torch
import numpy as np
from tqdm import tqdm as tqdm
from torch.autograd import Variable


def generate(x, model, gen_model, args, scaler=None, trainer=None, **kargs):
    for j in range(args.gen_max_features):
        lb_org, lb_new, r, x_adv, feats_idx, nb_iter = gen_model(x=x,
                                                                 num_feat=j+1,
                                                                 net=model,
                                                                 overshoot=args.gen_overshoot,
                                                                 max_iter=args.gen_max_iter,
                                                                 bound_min=kargs["bound_min"],
                                                                 bound_max=kargs["bound_max"],
                                                                 bound_type=kargs["bound_type"],
                                                                 alphas=kargs["alphas"],
                                                                 feature_selector=kargs['feature_selector'])
        if scaler:
            x_adv = scaler.inverse_transform(x_adv.reshape(1, -1))[0]
            lb_new = trainer.predict(x_adv)
        if lb_org != lb_new:
            break

    return lb_org, lb_new, x_adv, feats_idx


def test_grace(model, trainer, test_x, args, method="Naive", scaler=None, **kargs):
    if method == "Naive":
        from src.methods import NaiveGradient
        gen_model = NaiveGradient

    x_advs = []
    rs = []
    changed = []
    preds = []
    preds_new = []
    nb_iters = []
    total_feats_used = []
    feat_indices = []

    bar = range(len(test_x))
    bar = tqdm(range(len(test_x)), bar_format='Generating Contrastive Sample...{percentage:3.0f}%')
    for i in bar:
        x = test_x[i:i+1]
        x_var = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
        lb_org, lb_new, x_adv, feats_idx = generate(x_var, model, gen_model, args,
                                                    scaler=scaler, trainer=trainer, **kargs)
        total_feats_used.append(len(feats_idx))
        x_advs.append(x_adv)
        changed.append(lb_new != lb_org)
        feat_indices.append(feats_idx)

    x_advs = np.array(x_advs)
    avg_feat_changed = np.mean(total_feats_used)
    vals, counts = np.unique(changed, return_counts=True)

    num_correct = counts[np.where(vals == True)[0]]
    fidelity = num_correct/(len(changed))*1.0

    return avg_feat_changed, fidelity
