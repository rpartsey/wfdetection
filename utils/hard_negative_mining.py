import torch


def OHEM(x, y, device, ratio=0.8):
    num_inst = x.size(0)
    num_hns = int(ratio * num_inst)
    x_size = x.size()
    x = x.view(num_inst, -1, 2)
    y_size = y.size()
    y = y.view(num_inst, -1)

    x_ = x.clone().softmax(-1)

    inst_losses = torch.zeros(num_inst).to(device)
    for idx, label in enumerate(y.data):
        inst_losses[idx] = -(x_.data[idx, label].sum())
        # loss_incs = -x_.sum(1)
    _, idxs = inst_losses.topk(num_hns)
    x_hn = x.index_select(0, idxs)
    y_hn = y.index_select(0, idxs)
    x_hn = x_hn.view(num_hns, *x_size[1:])
    y_hn = y_hn.view(num_hns, *y_size[1:])
    return x_hn, y_hn
