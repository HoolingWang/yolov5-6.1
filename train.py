# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse   # 解析命令行参数模块
import math   # 数学公式模块
import os   # 与操作系统进行交互的模块 包含文件路径操作和解析
import random   # 生成随机数模块
import subprocess
import sys   # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time   # 时间模块 更底层
from copy import deepcopy   # 深度拷贝模块
from datetime import datetime, timedelta   # datetime模块能以更方便的格式显示日期或对日期进行运算。
from pathlib import Path   # Path将str转换为Path对象 使字符串路径易于操作的模块

try:
    import comet_ml  # must be imported before torch (if installed)
except ImportError:
    comet_ml = None

import numpy as np   # numpy数组操作模块
import torch
import torch.distributed as dist   # 分布式训练模块
import torch.nn as nn   # 对torch.nn.functional的类的封装 有很多和torch.nn.functional相同的函数
import yaml   # yaml是一种直观的能够被电脑识别的的数据序列化格式，容易被人类阅读，并且容易和脚本语言交互。一般用于存储配置文件。
from torch.optim import lr_scheduler   # PyTorch amp自动混合精度训练模块
from tqdm import tqdm   # 进度条模块

FILE = Path(__file__).resolve()  # __file__指的是当前文件(即train.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/train.py
ROOT = FILE.parents[0]  # YOLOv5 root directory  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:  # sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
    sys.path.append(str(ROOT))  # add ROOT to PATH  把ROOT添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径

### 加载自定义模块
import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

### Set DDP variables 分布式训练参数
# LOCAL_RANK 通常用于分布式训练，以标识当前进程在多进程中的位置。以下是几种常见的使用场景：
# 单机多卡训练：在一台机器上使用多个 GPU 进行训练，每个 GPU 由一个进程管理，LOCAL_RANK 用于标识每个进程对应的 GPU 编号。
# 分布式训练：在多台机器上进行训练，每台机器上运行多个进程，LOCAL_RANK 用于标识每个进程在本地机器上的编号。
# 当 LOCAL_RANK 为 -1 时，通常表示未进行分布式训练，代码会按照单 GPU 或 CPU 的模式执行。
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
# 当 RANK 为 -1 时，通常表示未进行分布式训练，代码会按照单 GPU 或 CPU 的模式执行。RANK 与 LOCAL_RANK 的区别在于：
# RANK 标识全局进程编号，在整个分布式系统中唯一。
# LOCAL_RANK 标识本地进程编号，在单台机器上的进程中唯一。
RANK = int(os.getenv("RANK", -1))
# WORLD_SIZE 通常用于分布式训练，以标识整个分布式系统中进程的总数。它用于设置和管理多机多卡训练环境。
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
'''
    查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，
    若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
    rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配。
'''
# check_git_info() 函数用于检查和获取当前 Git 仓库的相关信息，例如当前分支、提交哈希值、提交信息等。
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Trains YOLOv5 model with given hyperparameters, options, and device, managing datasets, model architecture, loss
    computation, and optimizer steps.

    `hyp` argument is path/to/hyp.yaml or hyp dictionary.
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")

    # Directories 设置目录和文件路径，用于保存训练过程中产生的权重文件和结果文件
    w = save_dir / "weights"  # weights dir 是一个路径变量，表示保存权重文件的根目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 'last'和'best'是保存最后训练结果和最好训练结果的权重文件的路径变量。
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters 加载超参数
    if isinstance(hyp, str):   # isinstance()是否是已知类型。 判断hyp是字典还是字符串
        with open(hyp, errors="ignore") as f:
            # 若hyp是字符串，即认定为路径，则加载超参数为字典
            hyp = yaml.safe_load(f)  # load hyps dict
    # 打印超参数 彩色字体
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings   保存训练中的参数hyp和opt
    if not evolve:
        # 保存超参数为yaml配置文件
        yaml_save(save_dir / "hyp.yaml", hyp)
        # 保存超参数为yaml配置文件
        yaml_save(save_dir / "opt.yaml", vars(opt))

    # Loggers   加载相关日志功能:如tensorboard,logger,wandb
    # 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        # 初始化日志记录器实例
        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            # 将日志记录器中的方法与字符串进行绑定
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        # 定义数据集字典
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    ### 画图开关,cuda,种子,读取数据集相关的yaml文件
    # Config   画图
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != "cpu"
    # 设置随机种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 加载数据配置信息
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    # 获取训练集、测试集图片路径
    train_path, val_path = data_dict["train"], data_dict["val"]
    # nc：数据集有多少种类别
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    # names: 数据集所有类别的名字，如果设置了single_cls则为一类
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset

    # Model 载入模型
    # 检查文件后缀是否是.pt
    check_suffix(weights, ".pt")  # check weights
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，可以根据自己的模型选择预训练权重
    pretrained = weights.endswith(".pt")   # type(pretrained) = bool
    if pretrained:
        # torch_distributed_zero_first(RANK): 用于同步不同进程对数据读取的上下文管理器
        with torch_distributed_zero_first(LOCAL_RANK):
            # attempt_download首先判断是否在本地，如果本地不存在就从google云盘中自动下载模型
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        """
        两种加载模型的方式: opt.cfg / ckpt['model'].yaml
        这两种方式的区别：区别在于是否是使用resume
        如果使用resume-断点训练: 
        将opt.cfg设为空，选择ckpt['model']yaml创建模型, 且不加载anchor。
        这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        原因：
        使用断点训练时,保存的模型会保存anchor,所以不需要加载，
        主要是预训练权重里面保存了默认coco数据集对应的anchor，
        如果用户自定义了anchor（先验框），再加载预训练权重进行训练，会覆盖掉用户自定义的anchor。
        """
        # 加载模型
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect

        ### 创建模型
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为ture 则会少加载两个键对（anchors, anchor_grid）
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        # 直接加载模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    """
    冻结模型层,设置冻结层名字即可
    作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    freeze为命令行参数，默认为0，表示不冻结
    """
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 首先遍历所有层
    for k, v in model.named_parameters():
        # 为所有层的参数设置梯度
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 判断是否需要冻结
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            # 冻结训练的层梯度不更新
            v.requires_grad = False

    # Image size   设置训练和测试图片尺寸
    # 获取模型总步长和模型输入图片分辨率
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 检查输入图片分辨率是否能被32整除
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size   设置一次训练所选取的样本数
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # 确保batch size满足要求
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer   优化器
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler   设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    if opt.cos_lr:
        # 使用余弦退火学习率
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        # 使用余弦退火学习率
        def lf(x):
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    # 可视化 scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA 设置ema（指数移动平均），考虑历史值对参数的影响，目的是为了收敛的曲线更加平滑
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    # 断点续训其实就是把上次训练结束的模型作为预训练模型，并从中加载参数
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode  使用单机多卡模式训练，目前一般不使用
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  多卡归一化
    if opt.sync_bn and cuda and RANK != -1:  # 多卡训练，把不同卡的数据做个同步
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader   训练集数据加载
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    '''
      返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    labels = np.concatenate(dataset.labels, 0)
    # 标签编号最大值  也可以理解为有多少类别
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0  创建验证集
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not resume:
            # opt.noautoanchor 为 False，则检查数据集的锚点并根据模型和阈值运行自动锚点调整。
            # Anchors 计算默认锚框anchor与数据集标签框的高宽比
            if not opt.noautoanchor:
                '''
                参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            # 将模型的浮点精度从半精度 (FP16) 调整为单精度 (FP32)。
            model.half().float()  # pre-reduce anchor precision
        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode   如果rank不等于-1,则使用DistributedDataParallel模式
    if cuda and RANK != -1:
        # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU。
        model = smart_DDP(model)

    # Model attributes   根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    return results

def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    """
    argparse 使用方法：
    parse = argparse.ArgumentParser()
    parse.add_argument('--s', type=int, default=2, help='flag_int')
    """
    parser = argparse.ArgumentParser("yolov5超参管理")
    # 配置权重文件 权重的路径./weights/yolov5s.pt.
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    # cfg 配置文件（网络结构） anchor/backbone/numclasses/head，训练自己的数据集需要自己生成
    # 生成方式——例如我的yolov5s_mchar.yaml 根据自己的需求选择复制./models/下面.yaml文件，5个文件的区别在于模型的深度和宽度依次递增
    parser.add_argument("--cfg", type=str, default=ROOT / "./models/yolov5s.yaml", help="model.yaml path")
    # data 数据集配置文件（路径） train/val/label/， 该文件需要自己生成
    # 生成方式——例如我的/data/mchar.yaml 训练集和验证集的路径 + 类别数 + 类别名称
    parser.add_argument("--data", type=str, default=ROOT / "./data/apple-fys.yaml", help="dataset.yaml path")
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    # 训练轮次
    parser.add_argument("--epochs", type=int, default=50, help="total training epochs")
    # 训练批次（每次迭代多少张照片）
    parser.add_argument("--batch-size", type=int, default=2, help="total batch size for all GPUs, -1 for autobatch")
    # 图像尺寸
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    # rect 是否采用矩形训练，默认为False
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    # resume 是否接着上次的训练结果，继续训练
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    # nosave 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # best.pt/ last.pt 不建议运行代码添加 --nosave
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    # noval 最后进行测试, 设置了之后就是训练结束后测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    # noautoanchor 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    # 控制是否在处理过程中显示检测结果的图像
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    # evolve参数进化， 遗传算法调参
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    # 下载population的路径
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    # 从先前保存的状态中加载当前种群的配置、超参数及其适应度等信息
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    # bucket谷歌优盘 / 一般用不到
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    # cache 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    # mage-weights 使用图片采样策略，默认不使用
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    # device 设备选择
    parser.add_argument("--device", default=0, help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # multi-scale 多尺度训练
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    # single-cls 数据集是否多类/默认True
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    # optimizer 优化器选择 / 提供了三种优化器
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    # sync-bn:是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    # workers/dataloader的最大worker数量
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    # 保存路径 / 默认保存路径 ./runs/ train
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    # 实验名称
    parser.add_argument("--name", default="exp", help="save to project/name")
    # 项目位置是否存在 / 默认是都不存在
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    # cos-lr 余弦学习率
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    # 早停止忍耐次数 / 100次不更新就停止训练
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    # --freeze冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    # --save-period 多少个epoch保存一下checkpoint
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    # 设置随机数生成器的种子
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    # --local_rank 进程编号 / 多卡使用
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    # 在线可视化工具，类似于tensorboard工具，想了解这款工具可以查看https://zhuanlan.zhihu.com/p/266337608
    parser.add_argument("--entity", default=None, help="Entity")
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    # 使用数据的版本
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    # 传入的基本配置中没有的参数也不会报错# parse_args()和parse_known_args()
    # parse = argparse.ArgumentParser()
    # parse.add_argument('--s', type=int, default=2, help='flag_int')
    # parser.parse_args() / parse_args()

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """Runs training or hyperparameter evolution with specified options and optional callbacks."""
    ### 检查部分
    # 根据 global_rank 设置日志记录
    if RANK in {-1, 0}:
        # 输出所有训练参数 / 参数以彩色的方式表现
        print_args(vars(opt))
        # 检查代码库的Git状态，确保代码库没有未提交的更改
        check_git_status()
        # 检查安装是否都安装了 requirements.txt， 缺少安装包安装。
        check_requirements(ROOT / "requirements.txt")

    ### 断点训练
    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve: # opt.resume为真，可以通过comet训练，不使用进化算法同时满足
        '''
        opt.resume: 表示是否选择了恢复训练的选项。如果为真，则用户想要从上一次的检查点继续训练。
        check_comet_resume(opt): 这个函数可能用于检查是否可以通过 Comet 继续训练（Comet 是一个用于机器学习实验的跟踪工具）。如果这个函数返回假，说明不能通过 Comet 继续训练。
        opt.evolve: 表示是否选择了进化算法的选项。如果为真，则用户希望进行超参数优化或进化搜索。
        '''
        # resume权重路径，opt.resume不是字符串就加载最后一次运行结果
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        # 加载resume配置文件
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        # resume数据，原数据集
        opt_data = opt.data  # original dataset
        # opt_yaml.is_file() 是 pathlib.Path 对象的方法之一，用于检查路径是否是一个文件。
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                # 使用 yaml 库安全地加载 YAML 文件内容，并将其赋值给变量 d
                d = yaml.safe_load(f)
        # 没有yaml文件就直接加载last.pt
        else:
            # 使用 torch 库加载一个存储在 last 路径下的模型文件，并从中提取名为 opt 的配置数据，将其赋值给变量 d
            d = torch.load(last, map_location="cpu")["opt"]
        # 替换之前的opt为提取的opt
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        # 检查 opt_data 是否是一个 URL，如果是，则通过 check_file 函数处理该 URL 并将结果赋值给 opt.data
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout

    ### 从头训练
    else:
        # 检查文件夹及相关路径
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),  # check_file 检查文件属性是否合理
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        # assert 语句: 用于检查条件是否为真。如果条件为假，则抛出一个 AssertionError 并显示指定的错误消息。
        # opt.cfg 或 opt.weights 至少有一个是非空的。如果这两个参数都为空，则抛出一个断言错误并显示错误信息 "either --cfg or --weights must be specified"。
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        # 如果使用遗传算法调参
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            # 使用遗传算法就重置opt.exist_ok, opt.resume参数
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            # .stem 属性获取文件名的基础名（不含扩展名）
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode  DDP mode 设置分布式数据进行模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train 训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        # 超参数列表(是否包含该超参数 - 最小值 - 最大值)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),
        }  # segment copy-paste (probability)

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        # 加载默认超参数
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )
        """
           遗传算法调参：遵循适者生存、优胜劣汰的法则，即寻优过程中保留有用的，去除无用的。
           遗传算法需要提前设置4个参数: 群体大小/进化代数/交叉概率/变异概率
        """

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def generate_individual(input_ranges, individual_length):
    """Generates a list of random values within specified input ranges for each gene in the individual."""
    """为individual中的每个元素生成指定范围内的随机数组成list"""
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.
    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    # 执行这个脚本/ 调用train函数 / 开启训练
    opt = parse_opt(True)
    for k, v in kwargs.items():
        # setattr() 赋值属性，属性不存在则创建一个赋值
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
