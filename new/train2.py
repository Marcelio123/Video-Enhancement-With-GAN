import argparse
import os
import datetime
import json
import torch
from torch import optim
from tqdm import tqdm
import utility
from dataset import Vimeo90k_septuplet_fi
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from ST_MFNet import models
from iSeeBetter.rbpn import Net as RBPN
from model import Net
import ST_MFNet.losses as losses

parser = argparse.ArgumentParser(description='STMFNet')

# parameters
# model
parser.add_argument('--net', type=str, default='STMFNet')
parser.add_argument('--weight_fi', type=str, default='weights\\stmfnet.pth')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--data_dir', type=str, help='root dir for all datasets')
parser.add_argument('--out_dir', type=str, default='train_results')
parser.add_argument('--ckpt_dir', type=str, default='checkpoint')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=1, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Lap+5*T_WGAN_GP', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='crop size')
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type, other options include plateau')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--patience', type=int, default=None, help='number of epochs without improvement after which lr will be reduced for plateau scheduler')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for feature extractor
parser.add_argument('--featc', nargs='+', type=int, default=[64, 128, 256, 512])
parser.add_argument('--featnet', type=str, default='UMultiScaleResNext')
parser.add_argument('--featnorm', type=str, default='batch')
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--finetune_pwc', dest='finetune_pwc', default=False,  action='store_true')

def train(model, train_loader, optimizer, loss_fn, logfile, epoch, outdir, ckpt_dir, test_loader):
    psnr_list = []
    loss_list = []
    loop = tqdm(train_loader, leave=True)
    for idx, batch in enumerate(loop):
        model.train()
        frame1, _, frame3, frame4, frame5, _, frame7 = batch

        frame1, frame3, frame4, frame5, frame7 = frame1.cuda(), frame3.cuda(), frame4.cuda(), frame5.cuda(), frame7.cuda()

        optimizer.zero_grad()

        predicted_frame4= model(frame1, frame3, frame5, frame7)

        loss = loss_fn(predicted_frame4, frame4, [frame3, frame5])
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        psnr = utility.calc_psnr(frame4, predicted_frame4).detach()
        psnr_list.append(psnr.tolist()) # (B,)
        loop.set_description("PSNR: {:.4f}, train loss: {:.4f}".format(
                psnr[0].item(),
                loss.item()
            ))
        if(idx % 10 == 0):
            msg = "index: {}\n train loss: {}\n".format(idx, loss.item())
            logfile.write(msg)
            train_psnr = 'train_psnr.json'
            train_loss = 'train_loss.json'
            # Save the dictionary to a JSON file
            with open(os.path.join(outdir, train_psnr), "w") as json_file:
                json.dump(psnr_list, json_file)
            with open(os.path.join(outdir, train_loss), "w") as json_file:
                json.dump(loss_list, json_file)
        
        if idx%100==0 and idx!=0:
            logfile.write("saving checkpoint\n")
            save_checkpoint(epoch, idx, model, ckpt_dir)

        if idx%250==0 and idx!=0:
            logfile.write("checkpoint saved at {ckpt_dir}\n")
            validate(model, test_loader, epoch, outdir, logfile)

def save_checkpoint(current_epoch, current_step, model, ckpt_dir):
    torch.save({'epoch': current_epoch, 'state_dict': model.state_dict()}, \
            os.path.join(ckpt_dir, 'model_epoch'+str(current_epoch)+'_step'+str(current_step).zfill(3)+'.pth'))

def validate(model, test_loader, current_epoch, outdir, logfile):
    logfile.write("start validating for epoch {}\n".format(current_epoch))
    model.eval()
    metrics = ["PSNR", "SSIM"]
    tmp_dict = {metric:[] for metric in metrics}
    loop = tqdm(test_loader, leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(loop):
            frame1, _, frame3, frame4, frame5, _, frame7 = batch
            frame1, frame3, frame4, frame5, frame7 = frame1.cuda(), frame3.cuda(), frame4.cuda(), frame5.cuda(), frame7.cuda()

            # Forward pass
            predicted_frame4 = model(frame1, frame3, frame5, frame7)

            # Calculate metrics
            for metric in metrics:
                if metric in ['PSNR' ,'SSIM']:
                    scores = getattr(utility, 'calc_{}'.format(metric.lower()))(frame4, predicted_frame4)
                    for score in scores:
                        tmp_dict[metric].append(score.item())
            loop.set_description("Avg PSNR: {:.4f}, Avg SSIM: {:.4f}".format(
                sum(tmp_dict['PSNR']) / len(tmp_dict['PSNR']),
                sum(tmp_dict['SSIM']) / len(tmp_dict['SSIM'])
            ))
    msg = "average psnr: {:.4f}\n".format(sum(tmp_dict['PSNR'])/len(tmp_dict['PSNR']))
    msg += "average ssim: {:.4f}\n".format(sum(tmp_dict['SSIM'])/len(tmp_dict['SSIM']))
    logfile.write(msg)
    file_name = 'metrics_epoch{}.json'.format(current_epoch)
    # Save the dictionary to a JSON file
    with open(os.path.join(outdir, file_name), "w") as json_file:
        json.dump(tmp_dict, json_file)

def transform():
    return Compose([
        ToTensor(),
    ])
def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    vimeo90k_train = Vimeo90k_septuplet_fi(db_dir=args.data_dir, train=True, upscale_factor=4, transform=transform(), augment_s=True, augment_t=True)
    train_loader = DataLoader(dataset=vimeo90k_train, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    vimeo90k_test = Vimeo90k_septuplet_fi(db_dir=args.data_dir, train=False, upscale_factor=4, transform=transform())
    test_loader = DataLoader(dataset=vimeo90k_test, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)

    model = getattr(models, 'STMFNet')(args).cuda()
    weight_fi = torch.load(args.weight_fi)
    model.load_state_dict(weight_fi['state_dict'])

    # Set requires_grad to False for all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    for param in model.dyntex_generator.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    loss = losses.Loss(args)
    logfile = open(os.path.join(args.out_dir, 'log.txt'), 'a', buffering=1)
    logfile.write('\n********STARTING FROM EPOCH {}********\n'.format(0))
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logfile.write(f"[{formatted_datetime}]\n")
    for epoch in range(args.epochs):
        logfile.write("epoch:{}\n".format(epoch))
        train(model, train_loader, optimizer, loss, logfile, epoch, args.out_dir, args.ckpt_dir, test_loader)
        logfile.write("saving checkpoint\n")
        save_checkpoint(epoch, -1, model, args.ckpt_dir)
        logfile.write("checkpoint saved at {ckpt_dir}\n")
        validate(model, test_loader, epoch, args.out_dir, logfile)

if __name__ == "__main__":
    main()
