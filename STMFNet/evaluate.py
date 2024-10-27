import argparse
import torch
import models
import os
import json
from tqdm import tqdm
from data.datasets import Vimeo90k_quintuplet
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import utility

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--net', type=str, default='STMFNet')
parser.add_argument('--dataset', type=str, default='Ucf101_quintuplet')
parser.add_argument('--metrics', nargs='+', type=str, default=['PSNR', 'SSIM'])
parser.add_argument('--weight', type=str, default='weights\\stmfnet.pth')
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')
parser.add_argument('--batch_size', type=int, default=4)

# model parameters
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')

def evaluate(model, test_loader, metrics, outdir, batch_size):
    model.eval()
    loop = tqdm(test_loader, leave=True)
    tmp_dict = {metric:[] for metric in metrics}

    with torch.no_grad():
        for idx, batch in enumerate(loop):
            frame1, frame3, frame4, frame5, frame7 = batch
            frame1, frame3, frame4, frame5, frame7 = frame1.cuda(), frame3.cuda(), frame4.cuda(), frame5.cuda(), frame7.cuda()

            # Forward pass
            predicted_frame4 = model(frame1, frame3, frame5, frame7)

            # Calculate metrics
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    scores = getattr(utility, 'calc_{}'.format(metric.lower()))(frame4, predicted_frame4)
                    for score in scores:
                        tmp_dict[metric].append(score.item())
            for i in range(len(frame1)):
                imwrite(predicted_frame4[i], os.path.join(outdir, "pd_psnr_{}_ssim_{}.png".format(tmp_dict['PSNR'][idx*batch_size+i], tmp_dict['SSIM'][idx*batch_size+i])), range=(0, 1))
                imwrite(frame4[i], os.path.join(outdir, "gt_psnr_{}_ssim_{}.png".format(tmp_dict['PSNR'][idx*batch_size+i], tmp_dict['SSIM'][idx*batch_size+i])), range=(0, 1))
            loop.set_description("Avg PSNR: {:.4f}, Avg SSIM: {:.4f}".format(
                sum(tmp_dict['PSNR']) / len(tmp_dict['PSNR']),
                sum(tmp_dict['SSIM']) / len(tmp_dict['SSIM'])
            ))
    print("average psnr: ", sum(tmp_dict['PSNR'])/len(tmp_dict['PSNR']))
    print("average ssim: ", sum(tmp_dict['SSIM'])/len(tmp_dict['SSIM']))
    return tmp_dict
            

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    model = getattr(models, args.net)(args).cuda()

    print('Loading the model...')

    weight = torch.load(args.weight)
    model.load_state_dict(weight['state_dict'])

    testsets = Vimeo90k_quintuplet(args.data_dir, train=False,  crop_sz=(256, 256), augment_s=False, augment_t=False)
    test_loader = DataLoader(dataset=testsets, batch_size=args.batch_size, shuffle=False, num_workers=0)

    metrics = evaluate(model, test_loader, ['PSNR' ,'SSIM'], args.out_dir, args.batch_size)
    file_name = 'metrics.json'
    # Save the dictionary to a JSON file
    with open(os.path.join(args.out_dir, file_name), "w") as json_file:
        json.dump(metrics, json_file)



if __name__ == "__main__":
    main()