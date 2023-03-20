import torch


def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs


def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])
