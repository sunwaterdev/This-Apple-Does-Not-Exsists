import torch
import torchvision.utils as vutils
import numpy as np
from generator import Generator 
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_images(args):

	# set up device
        device = torch.device('cuda:0' 
		if (torch.cuda.is_available() and args.ngpu>0)  
		else 'cpu')
	# load generator model
        print('[+] Loading model... ')
        netG = Generator(args).to(device)
        netG.load_state_dict(torch.load(args.netG, map_location=device))

        print('[+] Creating noise... ')
        # create random noise
        noise = torch.randn(args.n, args.nz, 1, 1, device=device)

        print('[+] Generating image... ')
        fake = netG(noise).detach().cpu()

        # resize images to 256x256
        fake = F.interpolate(fake, size=(1080, 1080), mode='bilinear', align_corners=True)

        # save image
        img = vutils.make_grid(fake, padding=2, normalize=True)
        plt.axis("off")
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.savefig(args.output_path)
