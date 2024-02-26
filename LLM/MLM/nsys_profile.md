
ref: [pytorch的四个hook函数](https://www.cnblogs.com/qizhou/p/17746217.html)
ref: [Understanding loss.backward() and cpu usage](https://discuss.pytorch.org/t/understanding-loss-backward-and-cpu-usage/173445/7)
ref: [PyTorch Model Performance Analysis and Optimization — Part 6](https://towardsdatascience.com/pytorch-model-performance-analysis-and-optimization-part-6-b87412a0371b)

根据如下：
```
def backward_hook_wrapper(module, details=None):
    
    # define register_full_backward_pre_hook function
    def bwd_pre_hook_print(self, output):
        message = f'before backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return output

    # define register_full_backward_hook function
    def bwd_hook_print(self, input, output):
        message = f'after backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return input

    # register hooks
    module.register_full_backward_pre_hook(bwd_pre_hook_print)
    module.register_full_backward_hook(bwd_hook_print)
    return module
```
修改为：

```

def forward_hook_wrapper(module, details=None):
    
    # define register_full_backward_pre_hook function
    def fwd_pre_hook_print(self, args):
        message = f'before backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return output

    # define register_full_backward_hook function
    def bwd_hook_print(self, input, output):
        message = f'after backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return input

    # register hooks
    module.register_forward_pre_hook(bwd_pre_hook_print)
    module.register_forward_hook(bwd_hook_print)
    return module


def backward_hook_wrapper(module, details=None):
    
    # define register_full_backward_pre_hook function
    def bwd_pre_hook_print(self, output):
        message = f'before backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return output

    # define register_full_backward_hook function
    def bwd_hook_print(self, input, output):
        message = f'after backward of {module.__class__.__qualname__}'
        if details:
            message = f'{message}: {details}'
        with torch.profiler.record_function(message):
            return input

    # register hooks
    module.register_full_backward_pre_hook(bwd_pre_hook_print)
    module.register_full_backward_hook(bwd_hook_print)
    return module
```