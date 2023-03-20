import argparse, os, sys, glob
import cv2
import builtins
import torch
import numpy as np
import shutil
import datetime
from einops import rearrange, repeat

import importlib
importlib.invalidate_caches()
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

import warnings
warnings.filterwarnings("ignore")

def load_model_from_config(config, ckpt, verbose=False):
    model = instantiate_from_config(config.model)
    
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/test",
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling, for smaller steps)", 
    )
    parser.add_argument(
        "--rsize",
        type=int,
        default=256,
        help="result size of sample image",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size for sampling",
    )
    parser.add_argument(
        "--max_img",
        type=int,
        default=50,
        help="max sample count",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: batch_size // 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/cin256-v2.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/cin256-v2/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    init_distributed_mode(opt)
    setup_for_distributed(is_main_process())
    device = get_rank()
    num_tasks = dist.get_world_size()
    NUM_CLASSES = 1000
    if opt.plms:
        opt.eta = 0
    if opt.plms and opt.dpm_solver:
        raise ValueError("Cannot use both plms and dpm_solver")
    seed_everything(opt.seed + device)

    config = OmegaConf.load(f"{opt.config}")
    print(OmegaConf.to_yaml(config))
    model = load_model_from_config(config, f"{opt.ckpt}", verbose = True)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model.module)
    elif opt.plms:
        sampler = PLMSSampler(model.module)
    else:
        sampler = DDIMSampler(model.module)

    if os.path.exists(opt.outdir) and device == 0:
        shutil.rmtree(opt.outdir)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    
    shape = (model.module.channels, model.module.image_size, model.module.image_size)
    n_rows = opt.n_rows if opt.n_rows > 0 else opt.batch_size // 2
    base_count = 0
    grid_count = 0
    n_iter = opt.max_img // opt.batch_size if opt.max_img % opt.batch_size == 0 else opt.max_img // opt.batch_size + 1
    print('n_iter', n_iter)
    print('start loop')
    with torch.no_grad():
        with model.module.ema_scope():
            for i in tqdm(range(n_iter)):
                if i % num_tasks != device:
                    continue
                
                batch_size = opt.batch_size if i < n_iter - 1 else opt.max_img - (n_iter - 1) * opt.batch_size
                uc = None
                if opt.scale != 1.0:
                    uc = model.module.get_learned_conditioning(
                        {model.module.cond_stage_key: torch.tensor(batch_size*[NUM_CLASSES]).to(model.device)}
                        )
                all_samples = list()
                xc = torch.randint(0, NUM_CLASSES, (batch_size,)).to(model.device)
                c = model.module.get_learned_conditioning({model.module.cond_stage_key: xc})
                
                samples, _ = sampler.sample(S=opt.steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=shape,
                                            sample_max_value = 1.,
                                            dynamic_thresholding_ratio = 1.0,
                                            clip_sample = False,
                                            thresholding = False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc, 
                                            verbose = False,
                                            eta=opt.eta)
                x_samples = model.module.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
                x_image_torch = torch.from_numpy(x_samples).permute(0, 3, 1, 2)
                
                for idx, x_sample in enumerate(x_image_torch):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = img.resize((opt.rsize, opt.rsize), resample=Image.BICUBIC)
                    img.save(os.path.join(sample_path, f"{base_count:04}_gpu{device}.png"))
                    base_count += 1
                print('base_count', base_count)
                if not opt.skip_grid and batch_size == opt.batch_size:
                    all_samples.append(x_image_torch)
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)
                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}_gpu{device}.png'))
                    grid_count += 1
                    

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
