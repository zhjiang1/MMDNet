import torch
import torch.nn as nn
import torch.nn.functional as F
class HimNet_multimode_v3(nn.Module):
    def __init__(
        self,
        num_nodes1,
        num_nodes2,
        num_nodes3,
        input_dim=3,
        output_dim=1,
        out_steps=1,
        hidden_dim=64,
        num_layers=1,
        cheb_k=2,
        ycov_dim=2,
        tod_embedding_dim=12,
        dow_embedding_dim=12,
        node_embedding_dim=16,
        st_embedding_dim=16,
        tf_decay_steps=4000,
        use_teacher_forcing=True,
    ):
        super().__init__()
        
        self.num_nodes1 = num_nodes1
        self.num_nodes2 = num_nodes2
        self.num_nodes3 = num_nodes3
        self.nodes_all = num_nodes1+num_nodes2+num_nodes3
        
        #self.num_nodes = num_nodes#节点数 121+182+66 = 369
        self.input_dim = input_dim#输入维度 3
        self.hidden_dim = hidden_dim#隐藏维度 64
        self.output_dim = output_dim#输出维度 1
        self.out_steps = out_steps#输出时间步 12
        self.num_layers = num_layers#层数 1
        self.cheb_k = cheb_k#2
        self.ycov_dim = ycov_dim#2
        self.node_embedding_dim = node_embedding_dim#节点嵌入 16
        self.st_embedding_dim = st_embedding_dim#时空嵌入 16
        self.tf_decay_steps = tf_decay_steps#4000 这是什么衰减？
        self.use_teacher_forcing = use_teacher_forcing#True

        self.encoder_s = HimEncoder_share_multi(
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            node_embedding_dim,
            meta_axis="S",
            # meta_axis=None,
        )
        self.encoder_t = HimEncoder_share_multi(
            input_dim,
            hidden_dim,
            cheb_k,
            num_layers,
            tod_embedding_dim + dow_embedding_dim,
            meta_axis="T",
            # meta_axis=None,
        )

        self.decoder = HimDecoder_share(
            64,
            hidden_dim,
            cheb_k,
            num_layers,
            st_embedding_dim,
            # meta_axis=None,
        )

        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        self.tod_embedding1 = nn.Embedding(24, tod_embedding_dim)#一天24小时，每个时间步一个嵌入
        self.tod_embedding2 = nn.Embedding(24, tod_embedding_dim)#一天24小时，每个时间步一个嵌入
        self.tod_embedding3 = nn.Embedding(24, tod_embedding_dim)#一天24小时，每个时间步一个嵌入
        
        self.dow_embedding1 = nn.Embedding(7, dow_embedding_dim)#一周七天，每天一个嵌入
        self.dow_embedding2 = nn.Embedding(7, dow_embedding_dim)#一周七天，每天一个嵌入
        self.dow_embedding3 = nn.Embedding(7, dow_embedding_dim)#一周七天，每天一个嵌入
        
        self.node_embedding1 = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.num_nodes1, self.node_embedding_dim)))
        self.node_embedding2 = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.num_nodes2, self.node_embedding_dim)))
        self.node_embedding3 = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.num_nodes3, self.node_embedding_dim)))
        
        self.mode_embedding_s = nn.init.xavier_normal_(nn.Parameter(torch.empty(3, self.node_embedding_dim)))
        self.mode_embedding_t = nn.init.xavier_normal_(nn.Parameter(torch.empty(3, tod_embedding_dim+dow_embedding_dim)))

        self.st_proj = nn.Linear(self.hidden_dim, self.st_embedding_dim)#64,16
        
        self.sequence_embedding_dim = 10
        self.week_embeddings = nn.Parameter(torch.randn(24, 7), requires_grad=True)#week-hour周期性嵌入
        self.sequence_embeddings = nn.Parameter(torch.randn(7, self.sequence_embedding_dim), requires_grad=True)#sequence-specific pattern 嵌入
        self.weight_pools=nn.Parameter(torch.randn(self.sequence_embedding_dim, self.nodes_all,self.nodes_all), requires_grad=True)#parameters-sharing pool        
        #self.linear_input = nn.Linear(3,64)
        #self.norm1 = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
    def compute_sampling_threshold(self, batches_seen):#计算采样阈值？
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self,x,y_cov,labels=None, batches_seen=None):
        
        #首先是多模式交通的输入分开的 地铁121 自行车182 出租车66
        
        #x_linear = self.linear_input(x)
        #print('x_linear:',x_linear.shape)
        
        x1 = x[:,:,0:self.num_nodes1,:]#地铁
        x2 = x[:,:,self.num_nodes1:(self.num_nodes1+self.num_nodes2),:]#自行车
        x3 = x[:,:,(self.num_nodes1+self.num_nodes2):,:]#出租车
        
        y1_cov = y_cov[:,:,0:self.num_nodes1,:]
        y2_cov = y_cov[:,:,self.num_nodes1:(self.num_nodes1+self.num_nodes2),:]
        y3_cov = y_cov[:,:,(self.num_nodes1+self.num_nodes2):,:]
        
        x1_tod = x1[:, -1, 0, 1]#B
        x1_dow = x1[:, -1, 0, 2]
        x2_tod = x2[:, -1, 0, 1]
        x2_dow = x2[:, -1, 0, 2]
        x3_tod = x3[:, -1, 0, 1]
        x3_dow = x3[:, -1, 0, 2]

        x1_tod_ugg = x1[:, :, 0, 1]
        x1_dow_ugg = x1[:, :, 0, 2]
        x2_tod_ugg = x2[:, :, 0, 1]
        x2_dow_ugg = x2[:, :, 0, 2]
        x3_tod_ugg = x3[:, :, 0, 1]
        x3_dow_ugg = x3[:, :, 0, 2]
        
        #x1_tod_emd_last = self.tod_embedding1((x1_tod * 24).long())#64,12
        #x2_tod_emd_last = self.tod_embedding2((x2_tod * 24).long())
        #x3_tod_emd_last = self.tod_embedding3((x3_tod * 24).long())
        
        x1_tod_emd = self.tod_embedding1((x1_tod_ugg * 24).long())#64,12,12
        x2_tod_emd = self.tod_embedding2((x2_tod_ugg * 24).long())
        x3_tod_emd = self.tod_embedding3((x3_tod_ugg * 24).long())
    
        #print("x1_tod_emd_last:",x1_tod_emd_last.shape)
        #print("x1_tod_emd:",x1_tod_emd.shape)
        
        x1_dow_emb = self.dow_embedding1(x1_dow_ugg.long())
        x2_dow_emb = self.dow_embedding2(x2_dow_ugg.long())
        x3_dow_emb = self.dow_embedding3(x3_dow_ugg.long())

        x1_time_embedding = torch.cat([x1_tod_emd, x1_dow_emb], dim=-1)
        x2_time_embedding = torch.cat([x2_tod_emd, x2_dow_emb], dim=-1)
        x3_time_embedding = torch.cat([x3_tod_emd, x3_dow_emb], dim=-1)
        
        #这里这个矩阵应该是时变-演化的 不过这里首次跑代码暂时用静态的
        
        time_adjacency=torch.einsum('td,dnk->tnk', self.week_embeddings.matmul(self.sequence_embeddings), self.weight_pools)#24,N,N
        #对邻接矩阵进行dropout试试？
        time_adjacency=F.softmax(F.relu(time_adjacency), dim=2)
        
        time_adjacency = self.dropout(time_adjacency)
        
        x1_time = (x1_tod * 24).long()
        #x2_time = (x2_tod * 24).long()
        #x3_time = (x3_tod * 24).long()
        
        x1_time_ugg = (x1_tod_ugg * 24).long()
        x2_time_ugg = (x2_tod_ugg * 24).long()
        x3_time_ugg = (x3_tod_ugg * 24).long()
        
        #B,12,N,N
        x1_time_adj = time_adjacency[x1_time_ugg,:self.num_nodes1,:self.num_nodes1]
        x2_time_adj = time_adjacency[x2_time_ugg,self.num_nodes1:(self.num_nodes1+self.num_nodes2),self.num_nodes1:(self.num_nodes1+self.num_nodes2)]
        x3_time_adj = time_adjacency[x3_time_ugg,(self.num_nodes1+self.num_nodes2):,(self.num_nodes1+self.num_nodes2):]

        #生成自适应矩阵 (N*16) * (16*N)
        #support1 = torch.softmax(torch.relu(self.node_embedding1 @ self.node_embedding1.T), dim=-1)
        #support2 = torch.softmax(torch.relu(self.node_embedding2 @ self.node_embedding2.T), dim=-1)
        #support3 = torch.softmax(torch.relu(self.node_embedding3 @ self.node_embedding3.T), dim=-1)
        
        #共用一个空间encoder和一个时间encoder
        #x1 = x_linear[:,:,0:self.num_nodes1,:]#地铁
        #x2 = x_linear[:,:,self.num_nodes1:(self.num_nodes1+self.num_nodes2),:]#自行车
        #x3 = x_linear[:,:,(self.num_nodes1+self.num_nodes2):,:]#出租车 
        
        
        
        h_s_1,h_s_2,h_s_3= self.encoder_s(x1,x2,x3,x1_time_adj,x2_time_adj,x3_time_adj,self.node_embedding1,
                       self.node_embedding2,self.node_embedding3,self.mode_embedding_s)
        h_t_1,h_t_2,h_t_3= self.encoder_t(x1,x2,x3,x1_time_adj,x2_time_adj,x3_time_adj,x1_time_embedding,
               x2_time_embedding,x3_time_embedding,self.mode_embedding_t)
        
        h_last_1 = (h_s_1 + h_t_1)[:, -1, :, :]  # B, N, hidden (last state)
        h_last_2 = (h_s_2 + h_t_2)[:, -1, :, :]  # B, N, hidden (last state)
        h_last_3 = (h_s_3 + h_t_3)[:, -1, :, :]  # B, N, hidden (last state)
        h_last = torch.cat([h_last_1,h_last_2,h_last_3],dim = 1)
        #print('h_last:',h_last.shape)
        
        st_embedding1 = self.st_proj(h_last_1)  # B, N1, st_emb_dim
        st_embedding2 = self.st_proj(h_last_2)  # B, N2, st_emb_dim
        st_embedding3 = self.st_proj(h_last_3)  # B, N3, st_emb_dim
        big_st_embeddings = torch.cat([st_embedding1,st_embedding2,st_embedding3],dim = 1)
        #support1 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding1, st_embedding1)),
            #dim=-1,) # (B,N1,N1)
        #support2 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding2, st_embedding2)),
            #dim=-1,)# (B,N2,N2)
        #support3 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding3, st_embedding3)),
            #dim=-1,)# (B,N3,N3)
        #support1_2 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding1, st_embedding2)),
            #dim=-1,)# (B,N12,N12)
        #support1_3 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding1, st_embedding3)),
            #dim=-1,)# (B,N13,N13)
        #support2_3 = torch.softmax(
            #torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding2, st_embedding3)),
            #dim=-1,)# (B,N23,N23)
        
        #——————————————模式内图的计算——————————————
        ht_list = [h_last] * self.num_layers
        
        #为什么这里要传入零矩阵？
        #go1 = torch.zeros((x_linear.shape[0], self.nodes_all, self.output_dim), device=x.device)
        
        #print('go1:',go1.shape)
        #print('x_linear:',x_linear.shape)
        #print('h_last:',h_last.shape)
        #go = self.norm1(h_last+x_linear.mean(axis=1))
        out = []
        
        #print(y1_cov.device,go1.device)
        
        for t in range(self.out_steps):
            h_de, ht_list = self.decoder(
                h_last,
                ht_list,
                time_adjacency[x1_time,...],
                big_st_embeddings,
            )
            go = self.out_proj(h_de)
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)

        
        return output

class HimEncoder_share_multi(nn.Module):
    def __init__(
        self,
        input_dim,#3
        output_dim,#64
        cheb_k,#2
        num_layers,#1
        embed_dim,#24/12
        meta_axis="S",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        #369,3,64,2,24/12，
        self.cells = nn.ModuleList(
            [HimGCRU_share_multi(input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU_share(output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x1,x2,x3,support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding):
        
        # x: (B, T, N, C) 16,12,369,3
        #support N*N
        #embeddings N*16
        
        batch_size = x1.shape[0] #16
        in_steps = x1.shape[1] #12个时间步
        
        num_nodes1 = x1.shape[2]
        num_nodes2 = x2.shape[2]
        num_nodes3 = x3.shape[2]
        
        current_input1 = x1 #当前输入 16,12,369,3
        current_input2 = x2 #当前输入 16,12,369,3
        current_input3 = x3 #当前输入 16,12,369,3
        
        output_hidden1 = []
        output_hidden2 = []
        output_hidden3 = []
        
        for cell in self.cells:
            state1 = cell.init_hidden_state(batch_size,num_nodes1).to(x1.device)#初始的state是全为0的张量
            state2 = cell.init_hidden_state(batch_size,num_nodes2).to(x1.device)#初始的state是全为0的张量
            state3 = cell.init_hidden_state(batch_size,num_nodes3).to(x1.device)#初始的state是全为0的张量
            
            inner_states1 = []
            inner_states2 = []
            inner_states3 = []
            
            for t in range(in_steps):
                #current_input[:, t, :, :]:16,369,3
                #state: 16,369,64
                #support: N*N
                #embeddings: N*16
                #x1, x2, x3, state1,state2,state3,support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding
                
                if len(embeddings1.shape)<3:
                    state1,state2,state3 = cell(current_input1[:, t, :, :],current_input2[:, t, :, :],current_input3[:, t, :, :],
                                                state1,state2,state3,support1[:,t,...],support2[:,t,...],support3[:,t,...],
                                                embeddings1,embeddings2,embeddings3,mode_embedding)
                else:
                    state1,state2,state3 = cell(current_input1[:, t, :, :],current_input2[:, t, :, :],current_input3[:, t, :, :],state1,state2,state3,support1[:,t,...],support2[:,t,...],support3[:,t,...],
                                    embeddings1[:,t,...],embeddings2[:,t,...],embeddings3[:,t,...],mode_embedding)                        #16,377,3 
                inner_states1.append(state1)
                inner_states2.append(state2)
                inner_states3.append(state3)
            output_hidden1.append(state1)
            output_hidden2.append(state2)
            output_hidden3.append(state3)
            
            current_input1 = torch.stack(inner_states1, dim=1)
            current_input2 = torch.stack(inner_states2, dim=1)
            current_input3 = torch.stack(inner_states3, dim=1)

        # current_input: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_input1,current_input2,current_input3

class HimGCRU_share_multi(nn.Module):
    #369,3,64,2,24/12
    def __init__(
        self,input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        #self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        
        #input_dim, output_dim, cheb_k, embed_dim
        #3+64,2*64,2,24/12
        self.gate = HimGCN_share_multi(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HimGCN_share_multi(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x1, x2, x3, state1,state2,state3,support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        
        #current_input[:, t, :, :]:16,377,3
        #state: 16,377,64
        #support: N*N
        #embeddings: N*16
        
        
        input_and_state1 = torch.cat((x1, state1), dim=-1)
        input_and_state2 = torch.cat((x2, state2), dim=-1)
        input_and_state3 = torch.cat((x3, state3), dim=-1)
#x1,x2,x3,support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding
        z_r1,z_r2,z_r3 = self.gate(input_and_state1,input_and_state2,input_and_state3, 
                                                 support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding)
        z_r1,z_r2,z_r3 = torch.sigmoid(z_r1),torch.sigmoid(z_r2),torch.sigmoid(z_r3)
        z1, r1 = torch.split(z_r1, self.hidden_dim, dim=-1)# z-(16,68,64), r-(16,68,64)
        z2, r2 = torch.split(z_r2, self.hidden_dim, dim=-1)# z-(16,68,64), r-(16,68,64)
        z3, r3 = torch.split(z_r3, self.hidden_dim, dim=-1)# z-(16,68,64), r-(16,68,64)
        
        candidate1 = torch.cat((x1, z1 * state1), dim=-1)
        candidate2 = torch.cat((x2, z2 * state2), dim=-1)
        candidate3 = torch.cat((x3, z3 * state3), dim=-1)
        
        hc1,hc2,hc3 = self.update(candidate1,candidate2,candidate3,support1,support2,support3,
                                             embeddings1,embeddings2,embeddings3,mode_embedding)
        hc1,hc2,hc3 = torch.tanh(hc1),torch.tanh(hc2),torch.tanh(hc3)
        h1 = r1 * state1 + (1 - r1) * hc1
        h2 = r2 * state2 + (1 - r2) * hc2
        h3 = r3 * state3 + (1 - r3) * hc3
        return h1, h2, h3
    def init_hidden_state(self, batch_size,num_nodes):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim) #16,369,64


class HimGCN_share_multi(nn.Module):

    #3+64,2*64,2,24/12
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)#24/12, 2*67, 128
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))#24/12,128
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )
    
    def GCN_basic(self,x,support,embeddings,weights_pool,bias_pool):
        
        #rint(support.shape)
        
        x_g = []
        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]#2阶图，第一个为对角线上为1其余为0的图，第二个为真实图
            for k in range(2, self.cheb_k):#这里self.cheb_k等于2，所以不会再加其他的图
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))#这里是x_g中加入图和x相乘后的 (68*68) * (16,68,64+3) -> (16,68,67)
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1) #(16,68,134)

        if self.meta_axis:
     
            #当为时间轴时，这里的embedding为(16,24); self.weights_pool (24/12, 2*67, 128)
            
            if self.meta_axis == "T":
                
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, weights_pool
                )  # B, cheb_k*in_dim, out_dim  (16,24) * (24,67*2,128) -> (16,67*2,128)
                bias = torch.matmul(embeddings, bias_pool)  # B, out_dim (16,24)*(24,128) -> (16,128)
                
                #x_g:(16,68,134) bni
                #weights:(16,67*2,64) bio
                #x_gconv:16,68,128
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            
            #当为空间轴时，这里的embedding为(node_num,12)
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            #16,68,128
                
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, weights_pool
                )  # B, N, cheb_k*in_dim, out_dim embeddings 16*68*12 (12,67*2,128)  16*68*134*128
                bias = torch.einsum("bnd,do->bno", embeddings, bias_pool)
                
                #(16,68,134) 16*68*134*128 -> 16,68,128
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, weights_pool) + bias_pool
        return x_gconv
    
    def forward(self,x1,x2,x3,support1,support2,support3,embeddings1,embeddings2,embeddings3,mode_embedding):
           
        multi_weights_pool = torch.einsum("mc,cdf->mcdf",mode_embedding,self.weights_pool)#(M,12,134,128)
        multi_bias_pool = torch.einsum("mc,cf->mcf",mode_embedding,self.bias_pool)
        
        x_gconv1 = self.GCN_basic(x1,support1,embeddings1,multi_weights_pool[0,...],multi_bias_pool[0,...])
        x_gconv2 = self.GCN_basic(x2,support2,embeddings2,multi_weights_pool[1,...],multi_bias_pool[1,...])
        x_gconv3 = self.GCN_basic(x3,support3,embeddings3,multi_weights_pool[2,...],multi_bias_pool[2,...])

        return x_gconv1,x_gconv2,x_gconv3
    

class HimDecoder_share(nn.Module):
    
    #num_nodes,#369
    #output_dim + ycov_dim,#3
    #hidden_dim,#64
    #cheb_k,#2
    #num_layers,#1
    #st_embedding_dim,#12
    
    def __init__(
        self,
        input_dim,#3
        output_dim,#64
        cheb_k,#2
        num_layers,#1
        embed_dim,#12
        meta_axis="ST",
    ):
        super().__init__()
        self.input_dim = input_dim#3
        self.num_layers = num_layers#1
        self.cells = nn.ModuleList(
            [HimGCRU_share(input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU_share(output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )
    
    #embeddings: 16,68,16
    
    def forward(self, xt, init_state, support, embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden
    
class HimGCRU_share(nn.Module):
    #369,3,64,2,24/12
    def __init__(
        self,input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        #self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        
        #input_dim, output_dim, cheb_k, embed_dim
        #3+64,2*64,2,24/12
        self.gate = HimGCN_share(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HimGCN_share(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x, state, support, embeddings):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        
        #current_input[:, t, :, :]:16,377,3
        #state: 16,377,64
        #support: N*N
        #embeddings: N*16
        
        
        input_and_state = torch.cat((x, state), dim=-1) #16,377,64+3
        
        #input_and_state: 16,377,64+3
        #support: N*N
        #embeddings: N*16
        
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))#B,N,64
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)# z-(16,68,64), r-(16,68,64)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h
    def init_hidden_state(self, batch_size,num_nodes):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim) #16,369,64
    
class HimGCN_share(nn.Module):

    #3+64,2*64,2,24/12
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)#24/12, 2*67, 128
                )
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))#24/12,128
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        
        #input_and_state(x): 16,377,64+3
        #support: N*N
        #embeddings: N*16
        
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]#2阶图，第一个为对角线上为1其余为0的图，第二个为真实图
            for k in range(2, self.cheb_k):#这里self.cheb_k等于2，所以不会再加其他的图
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))#这里是x_g中加入图和x相乘后的 (68*68) * (16,68,64+3) -> (16,68,67)
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1) #(16,68,134)

        if self.meta_axis:
            
            #当为时间轴时，这里的embedding为(16,24); self.weights_pool (24/12, 2*67, 128)
            
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim  (16,24) * (24,67*2,128) -> (16,67*2,128)
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim (16,24)*(24,128) -> (16,128)
                
                #x_g:(16,68,134) bni
                #weights:(16,67*2,64) bio
                #x_gconv:16,68,128
                x_gconv = (
                    torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            
            #当为空间轴时，这里的embedding为(16,12)
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                    torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            #16,68,128
                
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim embeddings 16*68*12 (12,67*2,128)  16*68*134*128
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                
                #(16,68,134) 16*68*134*128 -> 16,68,128
                x_gconv = (
                    torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv