# Copyright 2021 MosaicML. All Rights Reserved.

import glob
import os
from typing import List

import pytest

import composer
import composer.algorithms as algorithms
import composer.trainer as trainer
from composer.algorithms import ScaleScheduleHparams
from composer.core.precision import Precision
from composer.datasets.hparams import SyntheticHparamsMixin
from composer.trainer.devices import CPUDeviceHparams
from tests.utils.synthetic_utils import configure_dataset_for_synthetic, configure_model_for_synthetic

modeldir_path = os.path.join(os.path.dirname(composer.__file__), 'yamls', 'models')
model_names = glob.glob(os.path.join(modeldir_path, '*.yaml'))
model_names = [os.path.basename(os.path.splitext(mn)[0]) for mn in model_names]


def get_model_algs(model_name: str) -> List[str]:
    algs = algorithms.list_algorithms()
    algs.remove("no_op_model")

    is_image_model = any(x in model_name for x in ("resnet", "mnist", "efficientnet", "timm", "vit", "deeplabv3"))
    is_language_model = any(x in model_name for x in ("gpt2", "bert"))

    if is_image_model:
        algs.remove("alibi")
        algs.remove("seq_length_warmup")
    if "alibi" in algs:
        pytest.importorskip("transformers")
        pytest.importorskip("datasets")
        pytest.importorskip("tokenizers")
    if is_language_model:
        algs.remove('blurpool')
        algs.remove('channels_last')
        algs.remove('cutmix')
        algs.remove('cutout')
        algs.remove('factorize')
        algs.remove('ghost_batchnorm')
        algs.remove('label_smoothing')
        algs.remove('layer_freezing')
        algs.remove('squeeze_excite')
        algs.remove('swa')
        algs.remove('mixup')
        algs.remove('scale_schedule')
        algs.remove('stochastic_depth')
        algs.remove('colout')
        algs.remove('progressive_resizing')
        algs.remove('randaugment')
        algs.remove('augmix')
        algs.remove('sam')
        algs.remove('selective_backprop')
    if model_name in ("unet"):
        algs.remove("mixup")
        algs.remove("cutmix")
    return algs


@pytest.mark.parametrize('model_name', model_names)
@pytest.mark.timeout(15)
def test_load(model_name: str):
    if 'timm' in model_name:
        pytest.importorskip("timm")
    if "vit" in model_name:
        pytest.importorskip("vit_pytorch")
    if model_name in ['unet']:
        pytest.importorskip("monai")
    if model_name in ['deeplabv3_ade20k_unoptimized', 'deeplabv3_ade20k_optimized']:
        pytest.importorskip("mmcv")
        pytest.skip(f"Model {model_name} requires GPU")

    trainer_hparams = trainer.load(model_name)
    trainer_hparams.precision = Precision.FP32
    trainer_hparams.algorithms = algorithms.load_multiple(*get_model_algs(model_name))

    assert trainer_hparams.train_dataset is not None
    configure_dataset_for_synthetic(trainer_hparams.train_dataset, model_hparams=trainer_hparams.model)
    trainer_hparams.train_subset_num_batches = 1

    assert trainer_hparams.val_dataset is not None
    configure_dataset_for_synthetic(trainer_hparams.val_dataset)
    trainer_hparams.eval_subset_num_batches = 1

    trainer_hparams.dataloader.num_workers = 0
    trainer_hparams.dataloader.pin_memory = False
    trainer_hparams.dataloader.prefetch_factor = 2
    trainer_hparams.dataloader.persistent_workers = False

    configure_model_for_synthetic(trainer_hparams.model)

    trainer_hparams.device = CPUDeviceHparams()
    my_trainer = trainer_hparams.initialize_object()

    assert isinstance(my_trainer, trainer.Trainer)


@pytest.mark.parametrize("ssr", ["0.25", "0.33", "0.50", "0.67", "0.75", "1.00", "1.25"])
@pytest.mark.filterwarnings("ignore:ScaleScheduleDeprecationWarning:DeprecationWarning")
def test_scale_schedule_load(ssr: str):
    trainer_hparams = trainer.load("classify_mnist")
    trainer_hparams.precision = Precision.FP32
    algs = [f"scale_schedule/{ssr}"]
    trainer_hparams.algorithms = algorithms.load_multiple(*algs)
    assert isinstance(trainer_hparams.train_dataset, SyntheticHparamsMixin)
    trainer_hparams.train_subset_num_batches = 1
    trainer_hparams.train_dataset.use_synthetic = True

    assert isinstance(trainer_hparams.val_dataset, SyntheticHparamsMixin)
    trainer_hparams.eval_subset_num_batches = 1
    trainer_hparams.val_dataset.use_synthetic = True
    trainer_hparams.device = CPUDeviceHparams()
    assert len(trainer_hparams.algorithms) == 1
    alg = trainer_hparams.algorithms[0]
    assert isinstance(alg, ScaleScheduleHparams)
    assert alg.ratio == float(ssr)
    my_trainer = trainer_hparams.initialize_object()
    assert isinstance(my_trainer, trainer.Trainer)
