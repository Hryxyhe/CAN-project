# CAN-project
包含源码CAN ；多卡CAN_ddp； 多卡去counting模块CAN_ddp_withoutcounting三个项目

1.数据集全在服务器上，地址是home/ipad_ocr/yhx/CAN/dataset，这里已经配置的差不多了，应该可以直接运行

2.多卡运行： 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 dist_train.py

3.每次多卡运行完后运行一下下面这个指令杀死进程，不然下次运行会显示adress被占用
ps -ef | grep dist_train.py | grep -v grep | awk '{print $2}' | xargs kill -9

该命令是用于在Linux或类Unix系统中通过进程名来终止进程的命令。以下是该命令的解释：
  ps -ef: 这个命令用于显示当前系统中所有正在运行的进程的详细信息。
  grep train.py: 通过grep命令过滤出包含"train.py"的进程信息，这样可以筛选出与"train.py"相关的进程。
  grep -v grep: 使用grep -v命令过滤掉包含"grep"的行，这样可以排除掉grep命令本身在进程列表中的信息。
  awk '{print $2}': 使用awk命令提取进程列表中的第二列，即进程ID（PID）。
  xargs kill -9: 使用xargs命令将之前提取的PID作为参数传递给kill -9命令，从而终止这些进程。
