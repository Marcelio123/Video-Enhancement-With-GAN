import torch
import os
import utility
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
from torchvision.utils import save_image as imwrite
from torchvision.transforms import Compose, ToTensor
from ST_MFNet import models
from iSeeBetter.rbpn import Net as RBPN
from model import Net
from dataset import Vimeo90k_septuplet
import argparse

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--net', type=str, default='STMFNet')
parser.add_argument('--dataset', type=str, default='Ucf101_quintuplet')
parser.add_argument('--metrics', nargs='+', type=str, default=['PSNR', 'SSIM'])
parser.add_argument('--weight_fi', type=str, default='weights\\stmfnet.pth')
parser.add_argument('--weight_vsr', type=str, default='weights\\netG_epoch_4_1.pth')
parser.add_argument('-c', '--combined_model', action='store_true', required=False, help="")
parser.add_argument('--weight', type=str, default='weights\\stmfnet.pth')
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")
parser.add_argument('--batch_size', type=int, default=4)

# model parameters
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')

def transform():
    return Compose([
        ToTensor(),
    ])

def evaluate(model, test_loader, metrics, outdir, batch_size):
    model.eval()
    tmp_dict = {metric:[] for metric in metrics}
    vsr_tmp_dict = {metric+"_vsr":[] for metric in metrics}
    loop = tqdm(test_loader, leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(loop):
            dataForVSR, Frames = batch
            input1, target1, neigbor1, flow1, bicubic1 = dataForVSR[0]
            input1, target1, bicubic1 = input1.cuda(), target1.cuda(), bicubic1.cuda()
            neigbor1 = [j.cuda() for j in neigbor1]
            flow1 = [j.cuda().float() for j in flow1]

            input3, target3, neigbor3, flow3, bicubic3 = dataForVSR[1]
            input3, target3, bicubic3 = input3.cuda(), target3.cuda(), bicubic3.cuda()
            neigbor3 = [j.cuda() for j in neigbor3]
            flow3 = [j.cuda().float() for j in flow3]

            input4, target4, neigbor4, flow4, bicubic4 = dataForVSR[2]
            input4, target4, bicubic4 = input4.cuda(), target4.cuda(), bicubic4.cuda()
            neigbor4 = [j.cuda() for j in neigbor4]
            flow4 = [j.cuda().float() for j in flow4]

            input5, target5, neigbor5, flow5, bicubic5 = dataForVSR[3]
            input5, target5, bicubic5 = input5.cuda(), target5.cuda(), bicubic5.cuda()
            neigbor5 = [j.cuda() for j in neigbor5]
            flow5 = [j.cuda().float() for j in flow5]

            input7, target7, neigbor7, flow7, bicubic7 = dataForVSR[4]
            input7, target7, bicubic7 = input7.cuda(), target7.cuda(), bicubic7.cuda()
            neigbor7 = [j.cuda() for j in neigbor7]
            flow7 = [j.cuda().float() for j in flow7]


            # Forward pass
            predicted_frame4, _, _, frame4, _, _ = model(input1, neigbor1, flow1, input3, neigbor3, flow3, input4, neigbor4, flow4, input5, neigbor5, flow5, input7, neigbor7, flow7)

            # Calculate metrics
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    scores = getattr(utility, 'calc_{}'.format(metric.lower()))(Frames[3].cuda(), predicted_frame4)
                    scores_vsr = getattr(utility, 'calc_{}'.format(metric.lower()))(frame4, predicted_frame4)
                    for score, vsr_score in zip(scores, scores_vsr):
                        tmp_dict[metric].append(score.item())
                        vsr_tmp_dict[metric+"_vsr"].append(vsr_score.item())
                        #print(metric, score.item(), vsr_score.item())
            for i in range(len(frame4)):
                imwrite(predicted_frame4[i], os.path.join(outdir, "pd_psnr_{}_ssim_{}_vsr_psnr{}_ssim_{}.png".format(tmp_dict['PSNR'][idx*batch_size+i], tmp_dict['SSIM'][idx*batch_size+i], vsr_tmp_dict['PSNR_vsr'][idx*batch_size+i], vsr_tmp_dict['SSIM_vsr'][idx*batch_size+i])), range=(0, 1))
                imwrite(Frames[3][i], os.path.join(outdir, "gt_batch_{}_idx_{}.png".format(idx, i)), range=(0, 1))
                imwrite(frame4[i], os.path.join(outdir, "vsr_batch_{}_idx_{}.png".format(idx, i)), range=(0, 1))
            loop.set_description("Avg PSNR: {:.4f}, Avg SSIM: {:.4f}".format(
                sum(vsr_tmp_dict['PSNR_vsr']) / len(vsr_tmp_dict['PSNR_vsr']),
                sum(tmp_dict['SSIM']) / len(tmp_dict['SSIM'])
            ))
    print("average psnr: ", sum(tmp_dict['PSNR'])/len(tmp_dict['PSNR']))
    print("average ssim: ", sum(tmp_dict['SSIM'])/len(tmp_dict['SSIM']))
    print("average psnr compared to vsr output: ", sum(vsr_tmp_dict['PSNR_vsr'])/len(vsr_tmp_dict['PSNR_vsr']))
    print("average ssim compared to vsr output: ", sum(vsr_tmp_dict['SSIM_vsr'])/len(vsr_tmp_dict['SSIM_vsr']))
    tmp_dict.update(vsr_tmp_dict)
    return tmp_dict

def main():
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    model1 = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=7, scale_factor=4).cuda()
    model2 = getattr(models, 'STMFNet')(args).cuda()
    weight_vsr = torch.load(args.weight_vsr)
    weight_fi = torch.load(args.weight_fi)
    model2.load_state_dict(weight_fi['state_dict'])
    model1.load_state_dict(weight_vsr)
    model = Net(model1, model2).cuda()
    if args.combined_model:
        weight = torch.load(args.weight)
        model.load_state_dict(weight['state_dict'])
    test_set = Vimeo90k_septuplet(db_dir=args.data_dir, train=False, upscale_factor=4, transform=transform())
    testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, testing_data_loader, ['PSNR' ,'SSIM'], args.out_dir, args.batch_size)
    file_name = 'metrics.json'
    # Save the dictionary to a JSON file
    with open(os.path.join(args.out_dir, file_name), "w") as json_file:
        json.dump(metrics, json_file)

if __name__ == "__main__":
    main()
