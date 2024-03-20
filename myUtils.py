# 自定义数据转换函数
import datetime
import inspect
import json
import math
import os
import time
import matplotlib.pyplot as plt
import torch
import numpy
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from numpy.lib.stride_tricks import as_strided as ast
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.utils import make_grid

from config import Config


class ScaleToMinusOneToOne:
    def __call__(self, sample):
        # 将数据从[0, 1]区间缩放到[-1, 1]
        return sample * 2 - 1

class NormalizeMask(transforms.Compose):
    def __init__(self, threshold):
        """
        将mask归一化
        :param threshold: 阈值，大于阈值的位置置1，否则置0
        """
        self.threshold = threshold
        super().__init__([])

    def normalize_mask(self, mask):
        # 大于阈值的位置置1，否则置0
        return torch.where(mask > self.threshold, torch.tensor(1), torch.tensor(0))

    def __call__(self, mask):
        return self.normalize_mask(mask).float()  # 转换为浮点型

# 将mask的01反转
class ReverseMask(transforms.Compose):
    def __init__(self):
        """
        将mask的01反转
        """
        super().__init__([])

    def __call__(self, mask):
        return (1 - mask).float()  # 转换为浮点型

def make_f_mask_list(mask_s_list, mask_t_list, f_mode ='mean'):
    """
    生成融合mask列表
    :param mask_s_list:
    :param mask_t_list:
    :param f_mode: 融合方式，mean, max, min, cat
    :return:
    """
    mask_f_list = []
    for i in range(len(mask_s_list)):
        if f_mode == 'mean':
            mask_f_list.append(mean_tensors(mask_s_list[i], mask_t_list[i]))
        elif f_mode == 'max':
            mask_f_list.append(torch.max(mask_s_list[i], mask_t_list[i]))
        elif f_mode == 'min':
            mask_f_list.append(torch.min(mask_s_list[i], mask_t_list[i]))
        elif f_mode == 'cat':
            mask_f_list.append(torch.cat((mask_s_list[i], mask_t_list[i]), dim=1))
    return mask_f_list


# 设置cuda或者cpu
def set_device(args):
    if torch.cuda.is_available():
        if isinstance(args, list):
            return (item.cuda() for item in args)
        else:
            return args.cuda()
    return args

# 获取优化器
def get_optimizer(config: Config, model):

    # 获取模型可训练参数
    params = filter(lambda p: p.requires_grad, model.parameters())

    if config.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, betas=config.adam_betas)
    elif config.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, momentum=config.sgd_momentum)
    else:
        optimizer = None
    return optimizer

# 获取鉴别器优化器
def get_optimizer_D(config: Config, model):
    if config.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate * config.discriminator_lr_factor,
                                     betas=config.adam_betas)
    elif config.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate * config.discriminator_lr_factor,
                                    momentum=config.sgd_momentum)
    else:
        optimizer = None
    return optimizer

# 获取学习率调度器
def get_scheduler(optimizer, config: Config):
    # 如果不进行学习率调度
    if config.lr_decay == False:
        return optimizer
    if config.lr_decay_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_epoch, gamma=config.step_lr_decay_factor)
    elif config.lr_decay_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_decay_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_decay_policy)
    return scheduler

# 计算已训练时间，剩余时间，预计完成时间
def get_time(total_time_start, now_epoch, total_epoch):
    '''
    :param total_time_start: 开始时间
    :param now_epoch: 当前进度
    :param total_epoch: 总要求进度
    :return: 已训练时间，剩余时间，预计完成时间
    '''
    if now_epoch == 0:
        now_epoch = 1
    # 计算已训练时间
    time_train = time.time() - total_time_start
    # 计算剩余时间
    time_left = time_train * (total_epoch - now_epoch) / (now_epoch)
    # 计算预计完成时间
    time_end = time_train + time_left
    # 计算剩余时间
    time_left = time_end - time_train
    # 计算预计完成时间
    time_end = time_train + time_left
    # 计算预计完成时间
    time_end = total_time_start + time_end
    # 计算预计完成时间
    time_end = time.localtime(time_end)
    # 计算预计完成时间
    time_end = time.strftime("%m-%d %H:%M:%S", time_end)
    # 计算剩余时间
    time_left = str(datetime.timedelta(seconds=int(time_left)))
    # 计算已训练时间
    time_train = str(datetime.timedelta(seconds=int(time_train)))
    return time_train, time_left, time_end

# 将归一化的数据转换为图片
def tensor2img(tensor):
    img = tensor.detach().numpy()
    img = img * 255
    img = img.astype(numpy.uint8)
    return img

# 将[-1, 1]的数据转换为图片
def tensor2img_1(tensor):
    img = tensor.detach().numpy()
    img = (img + 1) / 2
    img = img * 255
    img = img.astype(numpy.uint8)
    return img

# 判断一个tensor是否只包含0和1
def is_binary_tensor(tensor):
    return torch.all(torch.logical_or(tensor == 0, tensor == 1))

# 返回一个tensor中的唯一值
def unique_values(tensor):
    return torch.unique(tensor)
# 返回一个tensor中的唯一值和数量
def unique_values_and_counts(tensor):
    unique_values, counts = torch.unique(tensor, return_counts=True)
    return unique_values, counts
# 将mask归一化
def normalize_mask(mask, threshold):
    # 大于阈值的位置置1，否则置0
    mask_normalized = torch.where(mask > threshold, torch.tensor(1), torch.tensor(0))
    return mask_normalized.float()  # 转换为浮点型


def set_requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# 对于一个batch的图片批次，将原图，mask图，mask叠加图，生成图拼接在一起
def make_val_grid(imgs, masks, img_masked, outputs):

    # 纵向拼接
    grid_imgs = make_grid(imgs, nrow=1, normalize=True, scale_each=True)
    grid_masks = make_grid(masks, nrow=1, normalize=True, scale_each=True)
    grid_imgs_masked = make_grid(img_masked, nrow=1, normalize=True, scale_each=True)
    grid_outputs = make_grid(outputs, nrow=1, normalize=True, scale_each=True)

    # 将图像横向拼接
    grid = torch.cat((grid_imgs, grid_masks, grid_imgs_masked, grid_outputs), dim=2)

    return grid


# 指标计算PSNR
def psnr(img1, img2, data_range=None):
    error = compare_psnr(img1, img2, data_range=data_range)
    return error

# 指标计算SSIM
def ssim(img1, img2, data_range=None):
    error = compare_ssim(img1, img2, channel_axis=0, win_size=51, data_range=data_range)
    return error

def ssim_by_list(frames1, frames2, data_range=None):
    assert len(frames1) == len(frames2)
    error = 0
    for i in range(len(frames1)):
        # print('shape', frames1[i].shape)
        error += ssim(frames1[i], frames2[i], data_range=data_range)
    return error / len(frames1)

def psnr_by_list(frames1, frames2, data_range=None):
    assert len(frames1) == len(frames2)
    error = 0
    for i in range(len(frames1)):
        error += psnr(frames1[i], frames2[i], data_range=data_range)
    return error / len(frames1)

def mean_tensors(tensor1, tensor2):
    """
    计算两个张量的均值
    :param tensor1:
    :param tensor2:
    :return:
    """
    # 检查两个张量的形状是否相同
    if tensor1.size() != tensor2.size():
        raise ValueError("Input tensors must have the same shape")

    # 逐位求均值
    mean_tensor = (tensor1 + tensor2) / 2.0

    return mean_tensor

def get_input_parameter_names(func):
    """
    获取函数的输入参数名称
    :param func: 函数
    :return: 输入参数名称列表
    """
    signature = inspect.signature(func)
    return list(signature.parameters.keys())

def count_parameters(model):
    """
    计算模型的参数数量
    :param model: 模型
    :return: 参数数量
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def count_trainable_parameters(model):
    """
    计算模型的可训练参数数量
    :param model: 模型
    :return: 可训练参数数量
    """
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params

def print_parameters(model, is_million = True):
    """
    打印模型的参数数量
    :param model: 模型
    :param is_million: 是否以百万为单位
    :return:
    """
    print(f"{'Layer':<25}{'Parameter Size':<20}")
    print("=" * 50)
    total = 0

    million = 1e6 if is_million else 1

    for name, module in model.named_children():
        num_params = count_parameters(module)
        print(f"{name:<25}{num_params/million:<20}")
        total += num_params
    print('total params:', total/million)

def print_trainable_parameters(model, is_million = True):
    """
    打印模型的可训练参数数量
    :param model: 模型
    :param is_million: 是否以百万为单位
    :return:
    """
    print(f"{'Layer':<25}{'Parameter Size':<20}")
    print("=" * 50)
    total = 0

    million = 1e6 if is_million else 1

    for name, module in model.named_children():
        num_params = count_trainable_parameters(module)
        print(f"{name:<25}{num_params/million:<20}")
        total += num_params
    print('total trainable params:', total/million)

def get_shapes_from_tuple(input_tuple, print_type = False):
    """
    获取元组中所有张量的形状
    :param input_tuple: 输入元组
    :param print_type: 是否打印类型
    :return:
    """
    shapes_list = []
    # 如果已经是张量
    if hasattr(input_tuple, 'shape'):
        # 是否打印类型
        if print_type:
            temp_str = str(type(input_tuple)) + ": " + str(input_tuple.shape)
        else:
            temp_str = str(input_tuple.shape)
        shapes_list.append(temp_str)
        return shapes_list
    elif hasattr(input_tuple, 'size'):
        # 是否打印类型
        if print_type:
            temp_str = str(type(input_tuple)) + ": " + str(input_tuple.size())
        else:
            temp_str = str(input_tuple.size())
        shapes_list.append(temp_str)
        return shapes_list

    # 如果是 tuple或者list
    for element in input_tuple:
        # 使用 hasattr 检查元素是否具有 shape或者size 属性
        if hasattr(element, 'shape'):
            # 是否打印类型
            if print_type:
                temp_str = str(type(element)) + ": " + str(element.shape)
            else:
                temp_str = str(element.shape)
            shapes_list.append(temp_str)
        elif hasattr(element, 'size'):
            # 是否打印类型
            if print_type:
                temp_str = str(type(element)) + ": " + str(element.size())
            else:
                temp_str = str(element.size())
            shapes_list.append(temp_str)
        else:
            # 判断元素是否还是为 tuple或者list
            if isinstance(element, tuple) or isinstance(element, list):
                # 递归调用,作为一个list插入shapes_list
                shapes_list.append(get_shapes_from_tuple(element))
            else:
                # 如果不是 tuple，也没有 shape或者size 属性，则返回变量类型
                shapes_list.append(str(type(element)) + ": No Shape")

    return shapes_list

def print_model_layers(model, **input_tensor):
    """
    打印模型的层，和输入输出形状
    :param model: 模型
    :param input_tensor: 输入张量，格式为：input_tensor = {'input1': torch.randn(1, 20)}
    :return:
    """

    # 定义forward_hook函数
    def forward_hook(module, input, output):

        # 获取输入参数名称
        # 如果有forward函数，获取forward函数的输入参数名称
        if hasattr(module, 'forward'):
            input_names = get_input_parameter_names(module.forward)
        else:
            input_names = get_input_parameter_names(module)
        # 如果包含self参数，删除self参数
        if 'self' in input_names:
            input_names.remove('self')

        # 打印分割线
        print("-" * 50)
        # 计算参数量
        num_params = count_parameters(module)
        # 计算可训练参数量
        num_trainable_params = count_trainable_parameters(module)
        # 打印模块的类名和参数量
        print(f"{module.__class__.__name__.ljust(20)} | {num_params/1e6:<20} | {num_trainable_params/1e6:<20}")


        for i,shape in enumerate(get_shapes_from_tuple(input)):
            # 打印输入形状,格式为：Input shape_i:<20| name:<20: shape
            print(f"{'Input shape_' + str(i):<20}{' | ' + input_names[i]:<20}{shape}")
        # 换行
        print()
        # 打印输出形状
        for i,shape in enumerate(get_shapes_from_tuple(output)):
            # 打印输出形状,格式为：Output shape_i : shape
            print(f"{'Output_' + str(i):<20}{shape}")
    # 打印分割线
    print("-" * 50)
    # 计算模型总参数量
    total_params = count_parameters(model)
    total_trainable_params = count_trainable_parameters(model)
    # 打印模型参数量
    print('Total params:', total_params/1e6)
    print('Total trainable params:', total_trainable_params/1e6)

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    try:
        # Enable hooks
        model.eval()
        with torch.no_grad():
            model(**input_tensor)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

# 清空log文件
def clear_log(path = 'log.txt'):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('')

# 记录训练日志
def write_log(log = '', path = 'log.txt', print_log = True, bar = None):
    with open(path, 'a', encoding='utf-8') as f:
        # 打印当前时间+日志
        file_log = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '  |  ' + log
        f.write(file_log + '\n')
        if print_log:
            if bar is not None:
                bar.write(log)
            else:
                print(log)

# 根据list绘制折线图, 保存图片, show_max是否显示最大值，最大值显示在标题
def draw_by_list(num_list, title, save_path, x_label = 'epoch', y_label = '', show_max = False):
    plt.plot(num_list)
    # 设置横向从0开始
    plt.xlim(0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if show_max:
        max_num = max(num_list)
        max_index = num_list.index(max_num)
        plt.title(title + ' max:' + str(max_num) + ' at ' + str(max_index))
    plt.savefig(save_path)
    plt.close()

# 将config的参数保存为json
def save_config(config, path):
    # 地址不存在则创建
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config.to_json(), indent=4))


# 根据位置和总长度，构造one-hot向量
def one_hot(index, length):
    """
    根据位置和总长度，构造one-hot向量
    :param index: 位置
    :param length: 总长度
    :return: one-hot向量
    """
    # 创建一个长度为length的全0向量
    one_hot_vector = torch.zeros(length)
    # 将指定位置置为1
    one_hot_vector[index] = 1
    return one_hot_vector

# 返回张量的最大值和最小值
def find_max_min(tensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    return max_val, min_val

# 获取激活函数
def get_activation(activ):
    if activ == 'relu':
        return torch.nn.ReLU()
    elif activ == 'lrelu':
        return torch.nn.LeakyReLU(0.2)
    elif activ == 'tanh':
        return torch.nn.Tanh()
    elif activ == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        return torch.nn.Identity()

if __name__ == '__main__':
    index = torch.tensor(3)
    length = 5
    hot = one_hot(index, length)
    print(hot)
