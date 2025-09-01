import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import numpy as np
import shutil
from DCT_utils import zigzag_order, reverse_zigzag_order

from configs.celeba64_uvit_small_2by2 import get_config

config = get_config()


def eval_step(n_samples, sample_steps, algorithm, Y_bound, path,
              config, dataset, train_state, score_model_ema,
              reverse_order, accelerator, device):
    logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                 f'mini_batch_size={config.sample.mini_batch_size}, samples save into {path}')

    def sample_fn(_n_samples):
        _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
        if config.train.mode == 'uncond':
            kwargs = dict()
        elif config.train.mode == 'cond':
            kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
        else:
            raise NotImplementedError

        if algorithm == 'euler_maruyama_sde':
            return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
        elif algorithm == 'euler_maruyama_ode':
            return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
        elif algorithm == 'dpm_solver':
            noise_schedule = NoiseScheduleVP(schedule='linear', SNR_scale=config.dataset.SNR_scale)
            model_fn = model_wrapper(
                score_model_ema.noise_pred,
                noise_schedule,
                time_input_type='0',
                model_kwargs=kwargs
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule)
            return dpm_solver.sample(
                _x_init,
                steps=sample_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
        else:
            raise NotImplementedError

    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)

    utils.DCTsample2dir(
        accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
        tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
        reverse_order=reverse_order, resolution=config.dataset.resolution,
        block_sz=config.dataset.block_sz, Y_bound=Y_bound
    )

    _fid = 0
    if accelerator.is_main_process:
        _fid = calculate_fid_given_paths((dataset.fid_stat, path))
        logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
        with open(os.path.join(config.workdir, f'eval_{n_samples}_{sample_steps}_new.log'), 'a') as f:
            print(f'step={train_state.step} Algorithm {algorithm} fid of {n_samples} samples for NFE {sample_steps}={_fid}', file=f)
        #shutil.rmtree(path)
        np.savez(f'Samples_{algorithm}_nfe{sample_steps}.npz', paths=path)
        print(f'Samples_{algorithm}_nfe{sample_steps}.npz')

    _fid = torch.tensor(_fid, device=device)
    _fid = accelerator.reduce(_fid, reduction='sum')
    return _fid.item()
