import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_flow
from ST_MFNet import models
from iSeeBetter.rbpn import Net as RBPN
from dataset import Vimeo90k_septuplet2
import argparse

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')
# model parameters
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')

class Net(nn.Module):
    def __init__(self, model1, model2):
        super(Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, input1, neigbor1, flow1, input3, neigbor3, flow3, input4, neigbor4, flow4, input5, neigbor5, flow5, input7, neigbor7, flow7):
        frame1 = self.model1(input1, neigbor1, flow1)
        frame3 = self.model1(input3, neigbor3, flow3)
        frame4 = self.model1(input4, neigbor4, flow4)
        frame5 = self.model1(input5, neigbor5, flow5)
        frame7 = self.model1(input7, neigbor7, flow7)
        x = self.model2(frame1, frame3, frame5, frame7)
        return x, frame1, frame3, frame4, frame5, frame7
    
class FIVSRNet(nn.Module):
    def __init__(self, model1, model2, mode=True):
        super(FIVSRNet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.mode = mode

    def forward(self, I1, I2, I3, I4, I5, I6, I7):
        up_I1 = F.interpolate(I1, scale_factor=4, mode='bicubic', align_corners=False)
        up_I2 = F.interpolate(I2, scale_factor=4, mode='bicubic', align_corners=False)
        up_I3 = F.interpolate(I3, scale_factor=4, mode='bicubic', align_corners=False)
        up_I4 = F.interpolate(I4, scale_factor=4, mode='bicubic', align_corners=False)
        up_I5 = F.interpolate(I5, scale_factor=4, mode='bicubic', align_corners=False)
        up_I6 = F.interpolate(I6, scale_factor=4, mode='bicubic', align_corners=False)
        up_I7 = F.interpolate(I7, scale_factor=4, mode='bicubic', align_corners=False)
        batch_length = I1.shape[0]
        if self.mode:
            I25 = self.model1(up_I1, up_I2, up_I3, up_I4)
            I35 = self.model1(up_I2, up_I3, up_I4, up_I5)
            I45 = self.model1(up_I3, up_I4, up_I5, up_I6)
            I55 = self.model1(up_I4, up_I5, up_I6, up_I7)

            down_I25 = F.interpolate(I25, scale_factor=0.25, mode='bicubic', align_corners=False)
            down_I35 = F.interpolate(I35, scale_factor=0.25, mode='bicubic', align_corners=False)
            down_I45 = F.interpolate(I45, scale_factor=0.25, mode='bicubic', align_corners=False)
            down_I55 = F.interpolate(I55, scale_factor=0.25, mode='bicubic', align_corners=False)
            input = I4
            neigbor = [down_I25, I3, down_I35, down_I45, I5, down_I55]
            flow = []
            for i in range(len(neigbor)):
                flow_temp = [get_flow(input[j].permute(1, 2, 0).cpu().detach().numpy().copy(), neigbor[i][j].permute(1, 2, 0).cpu().detach().numpy().copy(), flag=False) for j in range(batch_length)]
                flow_temp = [torch.from_numpy(j.transpose(2,0,1)).cuda().float() for j in flow_temp]
                flow_temp = torch.stack(flow_temp)
                flow.append(flow_temp)
            
            hr_f4 = self.model2(input, neigbor, flow)
            return hr_f4
        else:
            input   = self.model1(up_I1, up_I3, up_I5, up_I7)
            pred_up_I2      = self.model1(up_I5, up_I1, up_I3, up_I7)
            pred_up_I6      = self.model1(up_I1, up_I5, up_I7, up_I3)
            input = F.interpolate(input, scale_factor=0.25, mode='bicubic', align_corners=False)
            pred_down_I2 = F.interpolate(pred_up_I2, scale_factor=0.25, mode='bicubic', align_corners=False)
            pred_down_I6 = F.interpolate(pred_up_I6, scale_factor=0.25, mode='bicubic', align_corners=False)
            neigbor = [I1, pred_down_I2, I3, I5, pred_down_I6, I7]
            flow = []
            for i in range(len(neigbor)):
                flow_temp = [get_flow(input[j].permute(1, 2, 0).cpu().detach().numpy().copy(), neigbor[i][j].permute(1, 2, 0).cpu().detach().numpy().copy(), flag=False) for j in range(batch_length)]
                flow_temp = [torch.from_numpy(j.transpose(2,0,1)).cuda().float() for j in flow_temp]
                flow_temp = torch.stack(flow_temp)
                flow.append(flow_temp)
            hr_f4 = self.model2(input, neigbor, flow)
            return hr_f4

class FIVSRNet2(nn.Module):
    def __init__(self, model1, model2):
        super(FIVSRNet2, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, I1, I2, I3, I4, I5, I6, I7):
        up_I1 = F.interpolate(I1, scale_factor=4, mode='bicubic', align_corners=False)
        up_I3 = F.interpolate(I3, scale_factor=4, mode='bicubic', align_corners=False)
        up_I5 = F.interpolate(I5, scale_factor=4, mode='bicubic', align_corners=False)
        up_I7 = F.interpolate(I7, scale_factor=4, mode='bicubic', align_corners=False)
        batch_length = I1.shape[0]
        
        input   = self.model1(up_I1, up_I3, up_I5, up_I7)
        input = F.interpolate(input, scale_factor=0.25, mode='bicubic', align_corners=False)
        neigbor = [I1, I2, I3, I5, I6, I7]
        flow = []
        for i in range(len(neigbor)):
            flow_temp = [get_flow(input[j].permute(1, 2, 0).cpu().detach().numpy().copy(), neigbor[i][j].permute(1, 2, 0).cpu().detach().numpy().copy(), flag=False) for j in range(batch_length)]
            flow_temp = [torch.from_numpy(j.transpose(2,0,1)).cuda().float() for j in flow_temp]
            flow_temp = torch.stack(flow_temp)
            flow.append(flow_temp)
        hr_f4 = self.model2(input, neigbor, flow)
        return hr_f4
        
def test():
    args = parser.parse_args()
    model1 = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=7, scale_factor=4).cuda()
    model2 = getattr(models, 'STMFNet')(args).cuda()
    model = FIVSRNet2(model2, model1, False).cuda()
    model.eval()
    with torch.no_grad():
        shape = (1, 3, 64, 64)
        frames = [torch.randint(0, 256, size=shape, dtype=torch.uint8).cuda().float() for n in range(7)]
        print(model(frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], frames[6]).shape)


if __name__ == "__main__":
    test()

            
