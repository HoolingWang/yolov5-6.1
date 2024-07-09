# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
# 导入相关数据库
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse  # 解析目录行参数
import csv
import os
import platform
import sys
from pathlib import Path

import torch

# 设置路径，获取当前文件的绝对路径，并使用PATH方法添加到Path对象
FILE = Path(__file__).resolve()
# __file__: 这是一个内置变量，包含当前正在执行的 Python 脚本的路径（相对路径或绝对路径，取决于执行方式）。
# Path(__file__): 通过将 \_\_file\_\_ 传递给 Path 类，创建一个 Path 对象，这个对象表示当前文件的路径。
# .resolve()fuze将路径转化为绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 添加自定义模块


@smart_inference_mode()
def run(
    # 载入参数
    weights=ROOT / "yolov5s.pt",  # model path or triton URL   权重文件路径
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)   输入图像的路径
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path   数据集的配置文件
    imgsz=(640, 640),  # inference size (height, width)   
    conf_thres=0.25,  # confidence threshold   置信度阈值
    iou_thres=0.45,  # NMS IOU threshold   配极大值抑制的iou阈值
    max_det=1000,  # maximum detections per image   每张图像的最大检测框数量
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu   设备
    view_img=False,  # show results   是否打印检测结果图像
    save_txt=False,  # save results to *.txt   是否将检测结果保存为txt文件
    save_csv=False,  # save results in CSV format   是否保存为csv格式
    save_conf=False,  # save confidences in --save-txt labels   是否在文本文件包含置信度信息
    save_crop=False,  # save cropped prediction boxes   是否将检测出目标的区域保存为图像文件
    nosave=False,  # do not save images/videos   是否保存图像或者视频
    classes=None,  # filter by class: --class 0, or --class 0 2 3   指定要检测的类别（一类或多类）
    agnostic_nms=False,  # class-agnostic NMS   是否使用类别无关的非极大值抑制
    augment=False,  # augmented inference   是否使用数据增强的方式进行目标检测
    visualize=False,  # visualize features   是否可视化模型中的特征图
    update=False,  # update all models   是否自动更新模型权重文件
    project=ROOT / "runs/detect",  # save results to project/name   结果保存文件夹
    name="exp",  # save results to project/name   结果保存文件夹（与前者拼接）
    exist_ok=False,  # existing project/name ok, do not increment   是否覆盖保存结果的文件夹
    line_thickness=3,  # bounding box thickness (pixels)   检测框线条宽度
    hide_labels=False,  # hide labels   是否隐藏标签信息
    hide_conf=False,  # hide confidences   是否隐藏置信度
    half=False,  # use FP16 half-precision inference   是否使用FP16半精度训练
    dnn=False,  # use OpenCV DNN for ONNX inference   是否使用opencv的dnn作为onnx的后端
    vid_stride=1,  # video frame-rate stride   视频流检测步长（间隔帧数）
):
    # 初始化配置
    # 输入路径转化为
    source = str(source)
    # 是否保存图片和txt文件，如果同时为true则保存图像
    save_img = not nosave and not source.endswith(
        ".txt")  # save inference images
    # 判断source是不是视频、图像文件
    # Path()提取文件名，.suffix获取最后一个组件的文件扩展名
    # python中一般使用pathlib.Path().suffix对路径进行切片，直接获取.后缀方便处理
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否为链接
    # .lower()转化为小写；.upper()转化为大写；.title()转化为首字母大写
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    # 判断source是否为摄像头
    # .isnumeric()是否由数字组成
    webcam = source.isnumeric() or source.endswith(
        ".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    # 检查是不是一个图像或者视频的链接，如果是的话下载
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir是保存运行结果的文件夹名称，通过递增的方式来命名
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    # 如果需要保存txt文件就在创建的保存文件夹下面创建labels文件夹
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model 加载模型
    # 在参数配置部分选择你的设备,
    device = select_device(device)
    # 模型加载，输入信息包含权重文件，设备信息，详情参考common文件DetectMultiBackend函数的实现
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # '''
    # stride:推理用到的步长，默认32，大步长节省空间提高效率，小步长对于小目标检测性能有帮助，模型边界更加细致
    # name:保存推理结果名的列表
    # pt：是否加载pytorch模型
    # '''
    imgsz = check_img_size(imgsz, s=stride)  # check image size 将输入尺寸转化为可被stride整除的尺寸

    # Dataloader   加载数据
    bs = 1  # batch_size
    if webcam:   # 视频信息
        view_img = check_imshow(warn=True)   # check_imshow检查当前环境是否支持图像显示功能。
        # LoadStreams 类用于处理视频流数据，返回格式为源、处理后的图像、原始图像等。
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        '''
        source: 输入数据源；
        image_size: 图片识别前被缩放的大小；
        stride: 识别时的步长；
        atuo: 是否将图像填充为正方形，True表示不需要。
        '''
        bs = len(dataset)  # batch_size批大小
    elif screenshot:
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)
    else:   # 直接读取图像
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * \
        bs   # 前者是保存视频的路径，后者是一个cv2.VideoWriter对象

    # Run inference 开始推理
    # warmup 加载模型输入一张空图像初始化模型
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    '''
    seen：用于记录进程，当前有多少图片被处理
    dt：存储每一步的耗时
    '''
    for path, im, im0s, vid_cap, s in dataset:
        '''
        在dataset中，每次迭代返回的值为self.source, img, img0, None, ''
        path : 文件路径（即source）
        im : resise后的图像
        im0s : 原始图像
        vid_cap = none
        s : 图片的基本信息，比如路径大小
        '''
        with dt[0]:   # 记录图像预处理的时间
            im = torch.from_numpy(im).to(model.device)   #nunpy->torch
            im = im.half() if model.fp16 else im.float()  # uint8 to im.half()->fp16(2字节)/om.float()->32(4字节)
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                '''
                im[None]是一种使用None方法来增加向量维度的方法，相当于im.unsqueeze(0), im.view(1, *im.shape)
                [C, H, W] -> [B, C, H, W]
                '''
            if model.xml and im.shape[0] > 1: 
                ims = torch.chunk(im, im.shape[0], 0)
                '''
                torch.chunk 是 PyTorch 中的一个函数，用于沿指定维度将张量分割成多个子张量。
                torch.chunk(im, im.shape[0], 0) 将 im 沿第0维（批次维度）分割成 im.shape[0] 个子张量。
                这意味着如果 im 的形状是 (N, C, H, W)，则会得到 N 个形状为 (1, C, H, W) 的子张量。
                '''

        # Inference
        with dt[1]:   # 推理时间
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        # 对每张图像进行推理，并增加一个新的维度以便后续拼接
                        # [num_detections, detection_info] -> [1, num_detections, detection_info]
                        # num_detections表示检测框的数量， detection_info表示检测框的信息'x, y, w, h, confidence, class'
                        pred = model(image, augment=augment,
                                     visualize=visualize).unsqueeze(0)   #
                    else:
                        # [1, num_detections, detection_info] -> [batch_size, num_detections, detection_info]
                        pred = torch.cat(
                            (pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)   # 直接推理

        # NMS 使用非极大值抑制去除多余的检测框
        with dt[2]:   # 非极大值抑制时间
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            '''
            pred：网络的输出结果
            conf_thres：置信度阈值
            iou_thres：iou阈值
            classes：是否只保留特定类别 默认为None
            agnostic_nums: 进行nms是否也去除不同类比的框
            max_det：检测框最大的数量
            '''

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file CSV文件路径
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction,
                    "Confidence": confidence}  # CSV文件写入的内容
            with open(csv_path, mode="a", newline="") as f:
                # csv.DictWriter 创建一个写字典数据的 CSV 写入器，fieldnames 参数指定字段名称。
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():  # 如果是第一行就需要写入表头
                    writer.writeheader()
                writer.writerow(data)  # 写入具体内容

        # Process predictions
        for i, det in enumerate(pred):  # per image ，每次迭代处理一张图片
            '''
            i：每个batch的信息
            det：表示5个检测框的信息
            '''
            seen += 1  # 计数功能
            if webcam:  # batch_size >= 1
                # 如果输入源是wecam则batch_size >= 1，取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "  # s后面拼接一个字符串i
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                '''
                LoadImage流水读取文本文件中的照片或者视频 batch_size = 1
                p: 当前图片/视频的绝对路径
                s: 输出信息 原始为''
                im0: 原始图片 letter + pad 之前的图片
                frame: 视频六，此次取得是第几张图片
                '''

            p = Path(p)  # to Path
            # 图片或视频的保存路径
            save_path = str(save_dir / p.name)  # im.jpg
            # 保存坐标框的txt文件路径
            txt_path = str(save_dir / "labels" / p.stem) + \
                ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            # 设置输出图片信息。图片shape（w， h）
            s += "%gx%g " % im.shape[2:]  # print string   %g 是一种格式化浮点数的方式
            # normalization gain whwh
            # 原图的宽和高
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # 保存截图。如果save_crop为True则保存检测框的截图
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 实例化一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # 判断有无检测框
            if len(det):   # 单独处理每个检测框
                # Rescale boxes from img_size to im0 size
                # 将预测框的尺寸调整为原图大小（因为原图在检测时可能会被调整大小）
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印检测到的类别数量
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" /
                                     names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    # allow window resize (Linux)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # update model (to fix SourceChangeWarning)
        strip_optimizer(weights[0])


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT/"yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT/"data/images", help="file/dir/URL/glob/screen/0(webcam)")  #ROOT/"data/images"
    parser.add_argument("--data", type=str, default=ROOT/"data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand 将imgsz的信息扩展成h×w
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    # 检查环境，打印参数
    check_requirements(ROOT / "requirements.txt",
                       exclude=("tensorboard", "thop"))
    # 执行run函数
    run(**vars(opt))
    # **vars(opt): 这是 Python 中的解包操作符，称为关键字参数解包。它将字典中的键值对作为关键字参数传递给函数。
    # 在这种情况下，字典中的每个键值对会作为独立的参数传递给 run 函数。


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
