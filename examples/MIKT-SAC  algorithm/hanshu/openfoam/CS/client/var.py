# 常数定义
import sys

DELTA_T = 0.005  #OpenFOAM每一时间步的迭代步长
INTERVENTION_T = 0.2  #干预步长
N_PROCESSOR = 8  #定义并行运算进程数
OF_INIT = 'source /opt/rh/python27/enable && source /opt/rh/devtoolset-7/enable && source $HOME/OpenFOAM/OpenFOAM-8/etc/bashrc WM_LABEL_SIZE=64 WM_MPLIB=OPENMPI FOAMY_HEX_MESH=yes'
SOLVER = 'pisoFoam'
NICE = -10

# PACKAGE_PATH = '/'.join(sys.path[0].split('/'))


# TOTAL_TIME = 30  #总模拟时长
# 干预步长必须大于结果写入文件步长，因为干预的手段是通过修改结果文件来操作的
# 若生成的文件数过多，可以通过修改purgeWrite（>=1）来达到减小文件夹大小的目的
# WRITE_INTERVAL = 5  #结果写入文件步长

# 在shell中初始化OpenFOAM环境变量
# 设定求解器
# 调整NICR值（进程运行优先级），取值范围为-19 ~ 20，数字越小，优先级越高

# 需要学习的边界条件列表，注意后续风速列表应该与此列表一一对应
# minmax_value['name']

