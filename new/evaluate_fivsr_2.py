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
from model import FIVSRNet2
from dataset import Vimeo90k_septuplet2
import argparse

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--net', type=str, default='STMFNet')
parser.add_argument('--dataset', type=str, default='Ucf101_quintuplet')
parser.add_argument('--metrics', nargs='+', type=str, default=['PSNR', 'SSIM'])
parser.add_argument('--weight_fi', type=str, default='weights\\stmfnet.pth')
parser.add_argument('--weight_vsr', type=str, default='weights\\netG_epoch_4_1.pth')
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('-m', '--mode', action='store_true', required=False, help="Print debug spew.")

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
    loop = tqdm(test_loader, leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(loop):
            rawFrames, lrFrames = batch
            rawFrames = [j.cuda() for j in rawFrames]
            lrFrames = [j.cuda() for j in lrFrames]
            predicted_frame4 = model(lrFrames[0], lrFrames[1], lrFrames[2], lrFrames[3], lrFrames[4], lrFrames[5], lrFrames[6])

            # Calculate metrics
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    scores = getattr(utility, 'calc_{}'.format(metric.lower()))(rawFrames[3], predicted_frame4)
                    for score in scores:
                        tmp_dict[metric].append(score.item())
            for i in range(len(predicted_frame4)):
                imwrite(predicted_frame4[i], os.path.join(outdir, "pd_psnr_{}_ssim_{}.png".format(tmp_dict['PSNR'][idx*batch_size+i], tmp_dict['SSIM'][idx*batch_size+i])), range=(0, 1))
                imwrite(rawFrames[3][i], os.path.join(outdir, "gt_batch_{}_idx_{}.png".format(idx, i)), range=(0, 1))
            loop.set_description("Avg PSNR: {:.4f}, Avg SSIM: {:.4f}".format(
                sum(tmp_dict['PSNR']) / len(tmp_dict['PSNR']),
                sum(tmp_dict['SSIM']) / len(tmp_dict['SSIM'])
            ))
    print("average psnr: ", sum(tmp_dict['PSNR'])/len(tmp_dict['PSNR']))
    print("average ssim: ", sum(tmp_dict['SSIM'])/len(tmp_dict['SSIM']))
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
    model = FIVSRNet2(model2, model1).cuda()
    test_set = Vimeo90k_septuplet2(db_dir=args.data_dir, train=False, upscale_factor=4, transform=transform())
    testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, testing_data_loader, ['PSNR' ,'SSIM'], args.out_dir, args.batch_size)
    file_name = 'metrics.json'
    # Save the dictionary to a JSON file
    with open(os.path.join(args.out_dir, file_name), "w") as json_file:
        json.dump(metrics, json_file)

if __name__ == "__main__":
    main()
