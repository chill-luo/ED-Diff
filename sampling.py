import torch
import numpy as np

from nitm_net import UNet
from utils import mse_with_gaussian_blur


def sample_with_NiTM(x, t, y, sigma_t, encoder=None, decoder=None, args=None, ref=None):
    '''
    Use Encoder-Decoder structure like XXN.
    '''
    iters = args.ae_iters
    ae_lr = args.ae_lr
    sigma = args.sigma
    omega = args.omega
    bsize = args.num_sample

    l2_loss = torch.nn.MSELoss().cuda()
    step = args.total_steps - t[0].item() - 1

    # custom schedule
    ITER_MIN = 1
    LR_MIN = 0.005
    iters = int((iters - ITER_MIN) * np.cos(np.pi / 2 * step / (args.total_steps - 1))) + ITER_MIN
    ae_lr = (ae_lr - LR_MIN) * np.cos(np.pi / 2 * step / (args.total_steps - 1)) + LR_MIN

    # train NiTM
    x_in = y if step < 30  else x
    with torch.enable_grad():
        parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=ae_lr)
        encoder.train()
        decoder.train()
        
        # S2
        if args.cropped_size > 0:
            _, _, h, w = y.shape
            y_tmp = y.clone()
            x_in_tmp = x_in.clone()

        ae_loss_avg = 0
        for i in range(iters):
            if args.cropped_size > 0:
                hs = np.random.randint(0, h - args.cropped_size + 1)
                ws = np.random.randint(0, w - args.cropped_size + 1)
                y = y_tmp[:, :, hs:hs+args.cropped_size, ws:ws+args.cropped_size]
                x_in = x_in_tmp[:, :, hs:hs+args.cropped_size, ws:ws+args.cropped_size]

            n_rot = np.random.randint(0, 4)
            y = torch.rot90(y, k=n_rot, dims=[2, 3])

            y_lat = encoder(y)
            eps = y_lat.clone().normal_()
            out = decoder(y_lat + eps)

            y = torch.rot90(y, k=-n_rot, dims=[2, 3])
            y_lat = torch.rot90(y_lat, k=-n_rot, dims=[2, 3])
            out = torch.rot90(out, k=-n_rot, dims=[2, 3])
        
            rec_loss = 0.5 * l2_loss(out, y)
            mid_loss = 1/(2*sigma**2) * l2_loss(y_lat, x_in)
            ae_loss = rec_loss + mid_loss
            
            optimizer.zero_grad()
            ae_loss.backward()
            optimizer.step()

            ae_loss_avg += ae_loss.item()
        ae_loss_avg /= iters
        
    if args.cropped_size > 0:
        y = y_tmp
        with torch.no_grad():
            y_lat = encoder(y)
            out = decoder(y_lat)
    
    # S3
    flag_cond_list = [f"+{str(step).zfill(2)}"] * bsize
    with torch.no_grad():
        for ind in range(bsize):
            p_control = mse_with_gaussian_blur(y_lat[ind], y[ind], 1.0, 2.0)
            if p_control < ref["thr"][ind]:
                ref["img"][ind] = y_lat[ind]
                ref["thr"][ind] = p_control
                ref["step"][ind] = str(step).zfill(2)
            elif p_control > 3 * ref["thr"][ind]:
                flag_cond_list[ind] = f"-{ref['step'][ind]}"
    
    x_cond = torch.zeros_like(y_lat)
    for ind in range(bsize):
        x_cond[ind] = ref["img"][ind] if "-" in flag_cond_list[ind] else y_lat[ind]

    rho = omega * sigma**2 / sigma_t**2
    x_out = (x_cond + rho * x) / (1 + rho)
    
    print(f"[{str(step).zfill(2)}/{str(args.total_steps-1).zfill(2)}] | loss: {ae_loss_avg:.5f}, iters: {str(iters).zfill(2)}, lr: {ae_lr:.4f}")

    return encoder, decoder, x_out, y_lat, out, ref


def eddiff_sample(model, diffusion, noisy, p1, p2, args):
    ref = {
            "img": [None] * args.num_sample, 
            "thr": [100000] * args.num_sample,
            "step": ["000"] * args.num_sample
            }

    encoder = UNet(1, 1, need_sigmoid=False)
    encoder = encoder.cuda()
    decoder = UNet(1, 1, need_sigmoid=False)
    decoder = decoder.cuda()
    cond_fn = lambda x, t, sigma, encoder, decoder, ref: sample_with_NiTM(x, t, y=noisy, sigma_t=sigma, encoder=encoder, decoder=decoder, args=args, ref=ref)

    # S1
    spaced_t_steps = torch.Tensor([args.total_steps-1] * args.num_sample).cuda().to(torch.long)
    noise = diffusion.q_sample(noisy, spaced_t_steps)

    model_kwargs = {}
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    all_samples, encoder_new, decoder_new, all_predict, all_latent, all_reconstruct, ref_info = sample_fn(
        model,
        (args.num_sample, 1, noisy.shape[2], noisy.shape[3]),
        noise=noise,
        clip_denoised=args.clip_denoised,
        encoder=encoder,
        decoder=decoder,
        cond_fn=cond_fn,
        model_kwargs=model_kwargs,
        ref=ref
    )

    return all_samples, encoder_new, decoder_new, all_predict, all_latent, all_reconstruct, ref_info
