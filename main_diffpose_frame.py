import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.diffpose_frame import Diffpose

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    #随机种子
    parser.add_argument("--seed", type=int, default=19960903, help="Random seed")
    #配置文件路径
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file")
    #保存运行相关数据的路径
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    #日志文件夹的名称
    parser.add_argument("--doc", type=str, required=True, 
                        help="A string for documentation purpose. "\
                            "Will be the name of the log folder.", )
    #详细级别，包括 info、debug、warning 和 critical
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    #是否禁止交互，适用于 Slurm 作业启动器
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    #要训练/测试的动作，用逗号分隔，或使用 * 表示全部
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    
    #一些与 Diffusion 模型相关的参数，如 skip_type、eta、n_head、dim_model 等。
    ### Diffformer configuration ####
    #Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    # Diffusion model parameters
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor')
    # load pretrained model
    # 载入预训练模型
    parser.add_argument('--model_diff_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--model_pose_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')
    
    #training hyperparameter
    # 训练的超参数
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--lr_gamma', default=0.9, type=float, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--decay', default=60, type=int, metavar='N',
                        help='decay frequency(epoch)')
    
    #test hyperparameter
    # 测试的超参数
    parser.add_argument('--test_times', default=5, type=int, metavar='N',
                    help='the number of test times')
    parser.add_argument('--test_timesteps', default=50, type=int, metavar='N',
                    help='the number of test time steps')
    parser.add_argument('--test_num_diffusion_timesteps', default=500, type=int, metavar='N',
                    help='the number of test times')

    #  解析命令行参数
    args = parser.parse_args()
    # 拼接日志文件地址./exp/
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    # 打开配置文件
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # update configure file
    # 以命令行中的配置为先
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay

    # 训练模式
    if args.train:
        if os.path.exists(args.log_path):# 若日志路径存在
            overwrite = False # 不覆盖
            if args.ni:
                overwrite = True # 覆盖
            else:
                response = input("Folder already exists. Overwrite? (Y/N)") # 询问用户
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path) # 删除现有的
                os.makedirs(args.log_path) # 创建
            else:
                print("Folder exists. Program halted.") #报错
                sys.exit(0)# 退出
        else:# 不存在，则创建
            os.makedirs(args.log_path)

        # 将配置信息保存到日志路径下的 config.yml 文件中
        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        # 设置日志的输出格式、位置和级别
        # 通过 args.verbose.upper() 获取日志级别
        # 使用 getattr(logging, args.verbose.upper(), None) 将其转换为对应的整数级别
        # 如果转换失败，则会引发一个 ValueError 异常
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        # 接着，代码创建了两个日志处理程序 handler1 和 handler2
        # 分别用于控制台输出和文件输出
        # handler1 用于向控制台输出日志，handler2 用于向文件 stdout.txt 写入日志
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))

        # 通过 logging.Formatter 创建了一个格式化器，定义了日志的格式，包括日志级别、文件名、时间戳和消息内容。
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )

        # 将两个处理程序添加到 logger 对象中，并设置了日志记录的级别。
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else: # 测试
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 启用cuda的cudnn加速
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    try:
        runner = Diffpose(args, config)
        runner.create_diffusion_model(args.model_diff_path)                                              
        runner.create_pose_model(args.model_pose_path)
        runner.prepare_data()
        if args.train:
            runner.train()
        else:
            _, _ = runner.test_hyber()
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())
