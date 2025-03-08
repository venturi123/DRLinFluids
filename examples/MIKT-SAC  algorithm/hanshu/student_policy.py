from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

import numpy as np
import torch
from torch import nn
# from encoder import mlp
ModuleType = Type[nn.Module]
import torch.nn.functional as F
# from encoder import mlp
# import baselines.hanshu.encoder as encoder
import sys
"""lateral_fc函数的功能是实现学生模型和多个老师模型之间的侧向连接。该函数接收一组嵌入向量 zs（每个老师模型对应一个嵌入向量）
和输出的维度 size。函数首先根据嵌入向量的数量创建学生模型的权重 ps，然后通过Softmax函数将其归一化为概率分布。
然后，函数创建一组权重矩阵 Us，用于将每个老师模型的嵌入向量与学生模型进行线性组合。
最后，函数计算并返回学生模型的输出 teacher_sum 和权重分布 ps。"""

"""student_mlp函数是一个网络模型构建函数，用于创建包含学生模型和多个老师模型之间侧向连接的多层感知机（MLP）模型。
该函数接收一个编码器模型 encoder，一个编码器的作用域 encoder_scope，一组老师模型 teachers 和对应的作用域 teacher_scopes，
以及其他可选参数。函数首先通过编码器模型将输入数据进行编码，并得到嵌入向量 embedding。
然后，函数使用 teachers 模型对 embedding 进行处理，并得到一组嵌入向量 zs。接下来，函数通过多个全连接层对嵌入向量进行处理，
其中每一层的输出会与对应的老师模型输出进行侧向连接。最后，函数返回学生模型的输出 h、侧向连接的权重分布 student_ps，以及嵌入向量 embedding。"""

"""段代码实现了一个多层感知机模型，其中学生模型的每一层的输出与多个老师模型的输出进行侧向连接。
这种侧向连接的设计可以使学生模型在训练过程中利用老师模型的知识来提高性能。"""
class mlp(nn.Module):
    def __init__(self, env_state_shape,sourceenv_state_shape, output_dim: int = 0,num_layers=2,
                 num_hidden=[512,256], activation=nn.ReLU,device=None,variational=None):
        super(mlp, self).__init__()
        input_dim = int(np.prod(env_state_shape))
        self.device = device
        out_dim = int(np.prod(sourceenv_state_shape))
        # self.flatten = nn.Flatten()
        self.layers = nn.Sequential()
        # for i in range(num_layers):
        self.layers.append(nn.Linear(input_dim, num_hidden[0]))  #h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
        self.layers.append(activation())
        self.layers.append(nn.Linear(num_hidden[0], num_hidden[1]))
        self.layers.append(activation())

        self.embedding = nn.Linear(num_hidden[1], out_dim)
        if variational is not None:
            self.variational_logstd=nn.Parameter(torch.zeros(1, out_dim, device=self.device) )  # 这里不再加act维度

    #在前向传播方法forward中，输入X被扁平化处理，并通过线性层和激活函数进行前向传递，最后输出嵌入向量embedding
    def forward(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        h = obs.flatten(1)
        for layer in self.layers:
            h = layer(h )
        embedding = self.embedding(h )
        return embedding

#这里应该写为函数  调用teacher_policy
# class teacher_mlp(nn.Module):
#     def __init__(self, old_polocy = None,):
#         super(teacher_mlp, self).__init__()
#         self.flatten = nn.Flatten()
#         self.teachers = old_polocy
#
#     #在前向传播方法forward中，输入X被扁平化处理，并通过线性层和激活函数进行前向传递，最后输出嵌入向量embedding
#     def forward(self, X):
#         h = self.flatten(X)
#         zs=[]
#         i=0
#         for layer in self.teachers:
#             if i ==0 or i==2:
#                 z = layer(h)
#                 zs.append(z)
#             else:
#                 h = layer(z)
#             i+=1
#         # print('X, embedding', X, embedding)
#         return h, zs

#调用teacher_policy  这里包含actor  critic中间的网络层  获得其网络层输出
def teacher_mlp(old_polocy=None):
    def network_fn(x):
        zs = []
        a,b=old_polocy(x)   #a=logits   b为每层的输出
        zs.append(b[0])
        zs.append(b[2])
        return zs
    return network_fn


"""侧向全连接层（lateral_fc），用于在神经网络中进行侧向连接的操作。侧向连接是指将不同层之间的特征进行交叉融合，以增强网络的表达能力。
具体而言，该函数接收三个参数：scope表示作用域名称，zs是一个包含输入特征的列表，size表示输出特征的维度。
函数内部首先定义了一系列的可训练变量 ps 和 Us。变量 ps 是用来计算输入特征的权重，通过 softmax 函数对其进行归一化。
变量 Us 是连接输入特征和输出特征的权重矩阵。
接下来，通过矩阵乘法和求和的方式，将归一化的输入特征 zs 与权重矩阵 Us 相乘，并对结果进行求和，得到最终的输出特征 teacher_sum。
最后，函数返回输出特征 teacher_sum 和归一化的输入特征权重 ps。"""
# ps  us均需更新
def lateral_fc():   #ps Us不能重新定义   ps 控制policy权重大小  ps[0]+ps[1]=1
    def network_fn(zs, size):
        with torch.no_grad():
            ps = [torch.nn.Parameter(torch.tensor(0.01, requires_grad=True, dtype=torch.float32))]  #len(zs)=1
            ps.insert(0, torch.nn.Parameter(torch.tensor(-0.1, requires_grad=True, dtype=torch.float32)))
            ps = torch.nn.functional.softmax(torch.stack(ps, dim=0))
            #ps =tensor([0.4725, 0.5275])  正确  ps[1:]teacher_ps+ps[0]student_ps=1

            # for i, z in enumerate(zs):
            # Us =  [torch.nn.Parameter(torch.tensor([np.array(zs).shape[1], size], dtype=torch.float32))]
            # 定义一个权重张量
            Us = torch.empty(np.array(zs).shape[1], size, requires_grad=True,device=self.device)  # 假设形状为 [np.array(zs).shape[1], size]
            # 使用 Xavier 均匀初始化方法初始化权重
            nn.init.xavier_uniform_(Us)
            # Us = [nn.Parameter(Us)]

            # reshape so dimensions work out
            # print(ps[1:],np.array(zs).shape,np.array(zs).shape[0],np.array(zs).shape[1])  #tensor([0.5275]) 512出错
            teacher_ps = torch.reshape((ps[1:]), (1, 1, 1))   #shape=(1, 1, 1)  zs.shape=(1, 512)  这里变为三维大小有意义的
            # print(ps[1:], np.array(zs).shape, np.array(teacher_ps))  #(1, 512) [[[0.52747226]]]   (512, 512) [[[0.52747226]]]
            # print(torch.mul(teacher_ps, zs),np.array(torch.mul(teacher_ps, zs)).shape)   #(1, 1, 512)
            # torch.mul形状相同的矩阵相乘 逐元素相乘   torch.matmul矩阵相乘
            # tf.matmul(tf.multiply(teacher_ps, zs), Us)=tf.multiply(teacher_ps, zs)=(1,1,64) 或者 (1,512,64) teacher_sum=(1,512)
            teacher_sum = torch.sum(torch.matmul(torch.mul(teacher_ps, zs), Us ), dim=0,device=self.device)   #torch.mul(teacher_ps, zs)=(1, 1, 512)

            return teacher_sum, ps
    return network_fn

class student_mlp(nn.Module):
    def __init__(self, encoder, teachers,env_state_shape ,output_dim: int = 0,
                 num_layers=2, num_hidden=[512,256], activation=nn.ReLU,device= None,):
        super(student_mlp, self).__init__()
        self.device = device
        input_dim = int(np.prod(env_state_shape))
        self.students =[]
        self.num_hidden=num_hidden
        linear1=[nn.Linear(input_dim, num_hidden[0])]
        linear1+=[activation()]
        linear2=[nn.Linear(num_hidden[0], num_hidden[1])]
        linear2+=[activation()]
        self.students=linear1+linear2
        self.students=nn.Sequential(*self.students)  #自动注册为模型的可训练参数，并且在反向传播过程中会计算其梯度。
        # self.lateral_fc=lateral_fc()   #此处作为定义
        self.encoder=encoder
        self.teachers=teachers
        # with torch.no_grad():  ### 叶子结点是指那些需要用户手动创建的tensor  我们无法对非叶子结点进行deepcopy
        # 定义ps Us   ps 定义网络层权重系数   us是对网络层操作
        # self.ps0 = nn.Parameter(nn.functional.softmax(torch.stack([torch.tensor(-0.1, requires_grad=True, dtype=torch.float32),
        #                                           torch.tensor(0.01, requires_grad=True, dtype=torch.float32)], dim=0)))
        # self.ps1 = nn.Parameter(nn.functional.softmax(torch.stack([torch.tensor(-0.1, requires_grad=True, dtype=torch.float32),
        #                                           torch.tensor(0.01, requires_grad=True, dtype=torch.float32)], dim=0)))
        # self.ps00 = nn.Parameter(torch.tensor([0.4725, ],dtype=torch.float32, requires_grad=True, ))
        # self.ps10 = nn.Parameter(torch.tensor([0.4725, ],dtype=torch.float32, requires_grad=True, ))
        self.ps00 = nn.Parameter(torch.tensor([0.4725, ],dtype=torch.float32, requires_grad=True, ))
        self.ps10 = nn.Parameter(torch.tensor([0.4725, ],dtype=torch.float32, requires_grad=True, ))
        self.Us_1 = nn.Parameter(torch.empty(512, 512, device=self.device,dtype=torch.float32, requires_grad=True, ))
        nn.init.xavier_uniform_(self.Us_1)
        self.Us_2 = nn.Parameter(torch.empty(256, 256,device=self.device,dtype=torch.float32, requires_grad=True, ))
        nn.init.xavier_uniform_(self.Us_2)
        # print(self.ps.requires_grad)  #false

        if output_dim > 0:  #未使用
            self.students += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or num_hidden[-1]
        self.num_hidden=num_hidden

    #在前向传播方法forward中，输入X被扁平化处理，并通过线性层和激活函数进行前向传递，最后输出嵌入向量embedding
    #这里解决MI loss，输出variational_mu,variational_logstd
    def forward(self,  obs: Union[np.ndarray, torch.Tensor],act: Optional[Union[np.ndarray, torch.Tensor]] = None,):
        h = torch.as_tensor(obs, device=self.device, dtype=torch.float32)  # 之前h=16 (1, 120,30)或者(512, 120,30)
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).flatten(1)
        for i in range(np.array(h).shape[2]):
            h_one = h[:, :, i]
            embedding = self.encoder(h_one)  # 只进行120到24  将输出24叠加30次到720
            if i == 0:
                h_all = embedding       #这里为24
            else:
                h_all = torch.cat([h_all, embedding], dim=1)  #这里为24*30=720  并且传输最新状态(当前状态)给variation

        # act_all=torch.ones([512,8])
        # embedding = self.encoder(h)   #现在只进行15→3 不做16→4 只进行120到24  将输出24叠加30次到720
        if act is not None:   #对于critic设计  obs=obs+act
            act = torch.as_tensor(act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)            #改为actor更新完  vf立即更新完 1=1 8 2=2 3 3=4 5 4=6 7 act=[512,4]
            obs = torch.cat([obs, act], dim=1)   #在给定维度中连接给定的 seq 张量序列

            # act_all = torch.unsqueeze(act[:,0], 1)
            # act_all=torch.cat([act_all, torch.unsqueeze(act[:,1], 1),torch.unsqueeze(act[:,1], 1),torch.unsqueeze(act[:,2], 1),
            #                    torch.unsqueeze(act[:,2], 1),torch.unsqueeze(act[:,3], 1),torch.unsqueeze(act[:,3], 1),torch.unsqueeze(act[:,0], 1)], dim=1)  #cat这样操作不会出错
            # print(np.array(act_all).shape)
            embedding_act=torch.cat([h_all, act], dim=1)   #这里需要注意embedding=3，还需要加上act   #0609使用8action
            zs = self.teachers(embedding_act)  # 获得teacher_policy的网络层输出  shape=(2,)  需注意如果用不同的action做迁移，那么old_policy里得改
        else:
            zs = self.teachers(h_all)  # 这里只对actor设计
            # print('actor',obs.requires_grad,self.ps00.requires_grad)
        # print(np.array(zs).shape)   #两个网络层的输出 #(1, 512)(1, 256)   (512, 512)(512, 256)
        self.Us_1.requires_grad = True
        self.Us_2.requires_grad = True
        self.ps01 = torch.ones(1,device=self.device) - self.ps00
        self.ps11 = torch.ones(1,device=self.device) - self.ps10
        if self.ps00.item()>=0.95:
            self.ps00=nn.Parameter(torch.tensor([1, ],dtype=torch.float32, requires_grad=True, ))
        if self.ps10.item()>=0.95:
            self.ps10=nn.Parameter(torch.tensor([1, ],dtype=torch.float32, requires_grad=True, ))

        i=0
        for layer in self.students:
            if i ==0:
                student_out = layer(obs)   #这里改为obs  由于h=15
                # teacher_zs = zs[int(i/2)]    #(1, 512)(1, 256)   (512, 512)(512, 256)
                # teacher_sum, ps = self.lateral_fc(teacher_zs, self.num_hidden[int(i/2)])
                teacher_ps = torch.reshape(torch.clamp((self.ps01), min=0, max=1), (1, 1, 1))
                teacher_sum = torch.sum(torch.matmul(torch.mul(teacher_ps, zs[int(i/2)] ), self.Us_1),dim=0,)
                obs = teacher_sum + torch.clamp((self.ps00), min=0, max=1)* student_out
            elif i==2:
                student_out = layer(obs)
                teacher_ps = torch.reshape((torch.clamp((self.ps11), min=0, max=1)), (1, 1, 1))
                teacher_sum = torch.sum(torch.matmul(torch.mul(teacher_ps, zs[int(i/2)] ), self.Us_2),dim=0,)
                obs = teacher_sum + torch.clamp((self.ps10), min=0, max=1) * student_out
            elif i ==1 or i==3:
                obs = layer(obs)
            i+=1
        # print(self.ps10,self.ps00)
        student_ps = (torch.clamp((self.ps00), min=0, max=1)+torch.clamp((self.ps10), min=0, max=1))/2
        # print(self.ps0,self.ps0.grad,)
        # print(self.ps.requires_grad,self.Us_1.requires_grad) #True True
        #student_ps=[tensor(0.4725), tensor(0.4725)]  h=(1, 256)或者(batch_size, 256)
        return obs, student_ps, embedding   #如果是critic 这里的embedding不包括actor


class student_Net(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[ModuleType] = None,
        activation: Optional[ModuleType] = nn.ReLU,
        device= None,
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        old_policy= None,
        critic_name=None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        encoder=None,
        variational_net=None,
        variational_logstd=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))   #数组内相乘  env
        action_dim = int(np.prod(action_shape)) * num_atoms   #为了扩展到分配RL的网络   env
        if concat:     #输入形状是否由state_shape串联和action_shape，使用，加入action维度   critic使用
            input_dim += action_dim     #action维度在后面几层
            if critic_name==1:
                teachers = teacher_mlp(old_policy.critic1)   #vf_teachers1
            else:
                teachers = teacher_mlp(old_policy.critic2)   #vf_teachers2
            # encoder = mlp(input_dim, int(np.prod(source_env.observation_space.shape)) +int(np.prod(source_env.action_space.shape)),
            #               device=self.device)    #这里input_dim=env  输出为source_env  15到3
        else:    #actor
            teachers = teacher_mlp(old_policy.actor)  # 此处只是定义
            # encoder = mlp(input_dim, source_env.observation_space.shape,device=self.device)
            # variational_mu = variational_net(mapped_state_train)
            # self.variational_logstd = nn.Parameter(torch.zeros(1, int(np.prod(input_dim)),device=self.device))  #env_state_shape=input_dim
        self.variational_net = variational_net
        # self.variational_logstd = nn.Parameter(torch.zeros(1, int(np.prod(state_shape)), device=self.device))  # env_state_shape=input_dim
        self.variational_logstd=variational_logstd
        self.use_dueling = dueling_param is not None   #False
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = student_mlp(
            encoder, teachers, env_state_shape=input_dim,num_layers=2,
            num_hidden=hidden_sizes,  activation=activation,device=self.device)
        self.output_dim = self.model.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        # logits = self.model(obs)   # 一定=MLP中的forward函数的输出  [256,128]
        # print(np.array(obs).shape)
        logits, student_ps, embedding = self.model(obs,act=act)
        bsz = logits.shape[0]
        # return logits, state
        return logits, state, student_ps, embedding


