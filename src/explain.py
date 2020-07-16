import numpy as np
import pandas as pd

def explain_table(x, x_advs, y_pred, y_adv, diff_idx, features, num_normal=3):
    tmp = []
    same_idx = np.delete(list(range(len(x))), diff_idx)
    same_idx = same_idx[:num_normal]
    feat_idx = np.concatenate([same_idx, diff_idx], axis=0)
    tmp_x = {}
    tmp_x_adv = {}
    tmp_x['sample'] = 'Original'
    tmp_x_adv['sample'] = 'Contrastive'
    for feat in feat_idx:
        tmp_x[features[feat]] = "{:.3f}".format(x[feat])
        tmp_x_adv[features[feat]] = "{:.3f}".format(x_advs[feat])
    tmp_x['prediction'] = y_pred
    tmp_x_adv['prediction'] = y_adv
    tmp.append(tmp_x)
    tmp.append(tmp_x_adv)
    df = pd.DataFrame.from_dict(tmp)
    df = df[np.concatenate(
        [['sample', 'prediction'], features[feat_idx]], axis=0)]
    print("\n",df)


def explain_text(x, x_advs, y_pred, y_adv, diff_idx, features, units=["points"]):
    if len(units) == 1 :
        units = [units]*len(features)

    text = "IF "
    for feat in diff_idx:
        if x_advs[feat] != x[feat]:
            direction = "increased" if x_advs[feat] > x[feat] else "decreased"
            distance = "{:.3f}".format(np.abs(x_advs[feat] - x[feat]))
            tmp = "{} {} {} {}, ".format(features[feat], direction, distance, units[feat])
            text += tmp
    text += "the model would have predicted {} RATHER THAN {}".format(y_adv, y_pred)
    print('EXPLANATION:  "{}"'.format(text))
