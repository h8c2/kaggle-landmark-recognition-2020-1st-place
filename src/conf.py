import sys
import importlib
from types import SimpleNamespace
import argparse
# 对2021修改数据格式
sys.path.append("../configs")#加入环境变量

parser = argparse.ArgumentParser(description='')

parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)#只解析知道的变量即config

print("Using config file", parser_args.config)

args = importlib.import_module(parser_args.config).args #导入对应config中的变量

args["experiment_name"] = parser_args.config

args =  SimpleNamespace(**args)

args.img_path_train = args.data_path + 'train/'
args.img_path_val = args.data_path_valid + 'valid/'
args.img_path_test = args.data_path + 'test/'

try:
   if args.data_path_2 is not None:
      args.img_path_train_2 = args.data_path_2 + 'train/' 
except:
   args.data_path_2 = None