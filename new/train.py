import argparse
import os
import datetime
import json
import torch
from torch import optim
from tqdm import tqdm
import utility
from dataset import Vimeo90k_septuplet
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
parser.add_argument('--weight_vsr', type=str, default='weights\\netG_epoch_4_1_FullDataSet.pth')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--data_dir', type=str, help='root dir for all datasets')
parser.add_argument('--out_dir', type=str, default='train_results')
parser.add_argument('--ckpt_dir', type=str, default='checkpoint')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=1, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Lap+100*T_WGAN_GP', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='crop size')
parser.add_argument('--threads', type=int, default=1, help="Dataloader Threads")

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type, other options include plateau')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')


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

        # _, _, frame3, frame4, frame5, _, _ = Frames
        # frame3, frame4, frame5 = frame3.cuda(), frame4.cuda(), frame5.cuda()

        optimizer.zero_grad()

        predicted_frame4, _, hr_frame3, hr_frame4, hr_frame5, _ = model(input1, neigbor1, flow1, input3, neigbor3, flow3, input4, neigbor4, flow4, input5, neigbor5, flow5, input7, neigbor7, flow7)

        loss = loss_fn(predicted_frame4, hr_frame4, [hr_frame3, hr_frame5])
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        psnr = utility.calc_psnr(hr_frame4, predicted_frame4).detach()
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

        if idx%1000==0 and idx!=0:
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
            loop.set_description("Avg PSNR: {:.4f}, Avg SSIM: {:.4f}".format(
                sum(tmp_dict['PSNR']) / len(tmp_dict['PSNR']),
                sum(tmp_dict['SSIM']) / len(tmp_dict['SSIM'])
            ))
    msg = "average psnr: {:.4f}\n".format(sum(tmp_dict['PSNR'])/len(tmp_dict['PSNR']))
    msg += "average ssim: {:.4f}\n".format(sum(tmp_dict['SSIM'])/len(tmp_dict['SSIM']))
    msg += "average psnr compared to vsr output: {:.4f}\n".format(sum(vsr_tmp_dict['PSNR_vsr'])/len(vsr_tmp_dict['PSNR_vsr']))
    msg += "average ssim compared to vsr output: {:.4f}\n".format(sum(vsr_tmp_dict['SSIM_vsr'])/len(vsr_tmp_dict['SSIM_vsr']))
    logfile.write(msg)
    tmp_dict.update(vsr_tmp_dict)
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

    vimeo90k_train = Vimeo90k_septuplet(db_dir=args.data_dir, train=True, upscale_factor=4, transform=transform(), augment_s=True, augment_t=True)
    train_loader = DataLoader(dataset=vimeo90k_train, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    vimeo90k_test = Vimeo90k_septuplet(db_dir=args.data_dir, train=False, upscale_factor=4, transform=transform())
    test_loader = DataLoader(dataset=vimeo90k_test, num_workers=args.threads, batch_size=args.batch_size, shuffle=False)

    model1 = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=7, scale_factor=4).cuda()
    model2 = getattr(models, 'STMFNet')(args).cuda()
    weight_vsr = torch.load(args.weight_vsr)
    weight_fi = torch.load(args.weight_fi)
    model2.load_state_dict(weight_fi['state_dict'])
    model1.load_state_dict(weight_vsr)

    # Set requires_grad to False for all parameters initially
    for param in model1.parameters():
        param.requires_grad = False
    for param in model2.parameters():
        param.requires_grad = False

    for param in model2.dyntex_generator.parameters():
        param.requires_grad = True

    model = Net(model1, model2).cuda()

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
