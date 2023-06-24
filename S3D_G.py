
import torch
import torch.nn as nn

class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm3d(out_channel,
                                 eps=0.001, # value found in tensorflow
                                )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channel,
                                 eps=0.001, # value found in tensorflow
                                )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class S3D_G_block(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(S3D_G_block, self).__init__()
        # out_channel=[1x1x1,3x3x3_reduce,3x3x3,3x3x3_reduce,3x3x3,pooling_reduce]


        self.branch1 = BasicConv3d(in_channel,out_channel[0], kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[1], kernel_size=1, stride=1),
            BasicConv3d(out_channel[1], out_channel[1],kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(out_channel[1], out_channel[2], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channel, out_channel[3], kernel_size=1, stride=1),
            BasicConv3d(out_channel[3], out_channel[3], kernel_size=(1, 3, 3), stride=1, padding= (0, 1, 1)),
            BasicConv3d(out_channel[3], out_channel[4], kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3,stride=1,padding=1),
            BasicConv3d(in_channel, out_channel[5], kernel_size=(3,1,1), stride=1,padding=(1,0,0))
        )
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        # we replace weight matrix with 1D conv to reduce the para
        self.excitation = nn.Conv1d(1, 1, (3,1,1), stride=1,padding=(1,0,0))
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x=torch.cat([x1,x2,x3,x4], 1)
        input = x
        x=self.squeeze(x)
        x=self.excitation(x.permute(0,2,1,3,4))
        x=self.sigmoid(x)
        return x.permute(0,2,1,3,4)*input



class S3D_G(nn.Module):
    # Input size: 64x224x224
    def __init__(self, num_class):
        super(S3D_G, self).__init__()

        self.conv1=BasicConv3d(3,64,kernel_size=7,stride=2,padding=3)
        self.pool1=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.conv2=BasicConv3d(64,64,kernel_size=1,stride=1)
        self.conv3=BasicConv3d(64,192,kernel_size=3,stride=1,padding=1)
        self.pool2=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.Inception1=nn.Sequential(S3D_G_block(192, [64,96,128,16,32,32]),
                                      S3D_G_block(256, [128, 128, 192, 32, 96, 64]))
        self.pool3=nn.MaxPool3d(kernel_size=(3,3,3),stride=(2,2,2),padding=(1,1,1))
        self.Inception2=nn.Sequential(S3D_G_block(480,[192,96,208,16,48,64]),
                                      S3D_G_block(512, [160, 112, 224, 24, 64, 64]),
                                      S3D_G_block(512, [128, 128, 256, 24, 64, 64]),
                                      S3D_G_block(512, [112, 144, 288, 32, 64, 64]),
                                      S3D_G_block(528, [256, 160, 320, 32, 128, 128]))
        self.pool4=nn.MaxPool3d(kernel_size=(2,2,2),stride=2)
        self.Inception3=nn.Sequential(S3D_G_block(832,[256,160,320,32,128,128]),
                                      S3D_G_block(832, [384, 192, 384, 48, 128, 128]))
        self.avg_pool=nn.AvgPool3d(kernel_size=(8,7,7))
        self.dropout = nn.Dropout(0.4)
        self.linear=nn.Linear(1024,num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.Inception1(x)
        x = self.pool3(x)
        x = self.Inception2(x)
        x = self.pool4(x)
        x = self.Inception3(x)
        x = self.avg_pool(x)
        x = self.dropout(x.view(x.size(0),-1))
        return self.linear(x)

# class Pose3DEmbNet(nn.Module):
#     # Input size: 4,185,10,57,87
#     def __init__(self,output_channels,drop_out_rate):
#         super(Pose3DEmbNet, self).__init__()
#
#         self.conv1= S3D_G_block(185, [32, 16, 16, 16, 16, 16])
#         # out_channel=[*1x1x1,3x3x3_reduce,*3x3x3,3x3x3_reduce,*3x3x3,*pooling_reduce]
#         # self.conv2 = S3D_G_block(80, [8, 4, 4, 4, 4, 4])
#         self.avg_pool = nn.AvgPool3d(kernel_size=(3, 5, 5))
#         self.dropout = nn.Dropout(drop_out_rate)
#         self.linear = nn.Linear(20, output_channels)
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.avg_pool(x)
#         x = self.dropout(x.view(x.size(0), -1))
#         return self.linear(x) # B, output_channels

class Pose3DEmbNet(nn.Module):
    # Input size: 4,185,10,57,87
    def __init__(self,input_channels,output_channels):
        super(Pose3DEmbNet, self).__init__()
        self.conv1 = S3D_G_block(input_channels, [output_channels,2*output_channels,2*output_channels,output_channels//2,output_channels,output_channels])
        self.avg_pool = nn.AvgPool3d(kernel_size=(5, 5, 5))
        self.linear = nn.Linear(320, output_channels)


    def forward(self, x):
        x = self.conv1(x) # 5,5,5
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0), -1)) # B, output_channels

class TraceEmbNet(nn.Module):
    # Input size B,C,F,  here F means followers.
    def __init__(self, input_channels,output_channels):
        super(TraceEmbNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 4 * input_channels, kernel_size=2),
            nn.BatchNorm1d(4*input_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4 * input_channels, 4 * input_channels, kernel_size=2),
            nn.BatchNorm1d(4 * input_channels),
            nn.ReLU()
        )
        self.linear = nn.Linear(12*input_channels, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x=self.conv2(x)
        return self.linear(x.view(x.size(0), -1))  # B, output_channels

class Agg1DBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(Agg1DBlock,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 4 * input_channels, kernel_size=2),
            nn.BatchNorm1d(4 * input_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4*input_channels, 8 * input_channels, kernel_size=2),
            nn.BatchNorm1d(8 * input_channels),
            nn.ReLU()
        )
        self.maxpool=nn.MaxPool1d(3)
        self.conv3 = nn.Sequential(
            nn.Conv1d(8 * input_channels, 16 * input_channels, kernel_size=1),
            nn.BatchNorm1d(16 * input_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(16 * input_channels, output_channels, kernel_size=1),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
        self.conv=nn.Sequential(
            self.conv1,self.conv2,self.maxpool,self.conv3,self.conv4
        )

    def forward(self, x):
        return self.conv(x) # -1, output_channels,1

class PoseTraceNet(nn.Module):
    def __init__(self, app_channels, trace_channels,output_channels,dropout=0.4):
        super(PoseTraceNet, self).__init__()
        self.conv1 = S3D_G_block(app_channels,
                                 [16, 32, 32, 8,16,16])
        self.ave_pool = nn.AvgPool3d(kernel_size=(1, 5, 5))

        # self.agg1D=Agg1DBlock(16+32+16+16+trace_channels,output_channels*8)
        self.aggGRU=nn.GRU(16+32+16+16+trace_channels,output_channels*8,bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(output_channels*8*2, output_channels)

    def forward(self, app, trace):
        # app: B*T*MAX_N app_channels F K K
        # trace: B*T*MAX_N trace_channels F
        app = self.conv1(app)  # F,K,K
        app = self.ave_pool(app).squeeze() # F
        agg=torch.cat([app,trace],dim=1) # -1, 85+trace_channels, F
        # solution1: 1D conv
        # x=self.agg1D(agg)
        # solution2: GRU
        self.aggGRU.flatten_parameters()
        _,x=self.aggGRU(agg.permute(2,0,1)) # x: 2,-1,output*8
        x=x=x.permute(1,0,2)
        return self.linear(x.reshape(x.size(0), -1))  # B, output_channels

if __name__=='__main__':
    # net=S3D_G_block(185,[32,16,16,16,16,16])
    # # out_channel=[*1x1x1,3x3x3_reduce,*3x3x3,3x3x3_reduce,*3x3x3,*pooling_reduce]
    # pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    # net2 = S3D_G_block(80, [8, 4, 4, 4, 4, 4])
    # avg_pool = nn.AvgPool3d(kernel_size=(3, 7, 7))
    input=torch.randn(4,185,3,5,5)
    # output1=net(input) # 4,80,5,57,87
    # output2=pool3(output1)
    # output3=net2(output2)
    # output4=avg_pool(output3)
    net=Pose3DEmbNet(185,32)
    # net=TraceEmbNet(5,6)
    output4=net(input)
    print(output4.shape) # 4,20,1,4,6