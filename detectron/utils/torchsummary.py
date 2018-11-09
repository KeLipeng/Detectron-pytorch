import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_data):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if isinstance(input, (list, tuple)):
                #input_shape = [list(i.size())[0:] for i in input]
                input_shape = list(input[0].shape)
            else:
                input_shape = list(input[0].size())
            if isinstance(output, (list, tuple)):
                output_shape = [list(o.shape)[0:] for o in output]
            elif isinstance(output, (dict)):
                output_shape = [list(output[key].shape)[0:] for key in output.keys()]
            else:
                output_shape = list(output.size())
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            #print(input_shape)
            #print(output_shape)
            line_new = "{:>15}  {:>15} {:>25} {:>10}".format(
                input_shape,
                output_shape,
                class_name,
                params,
            )
            print(line_new)

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
            and not (module == model.module)
        ):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    #summary = OrderedDict()
    hooks = []
    

    # register hook
    model.apply(register_hook)

    net_outputs = model(**input_data)
    # remove these hooks
    for h in hooks:
        h.remove()

    return net_outputs
    #print(summary)
