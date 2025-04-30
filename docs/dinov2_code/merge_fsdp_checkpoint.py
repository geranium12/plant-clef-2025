# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import argparse
import glob
import io
import logging
import os
import pickle
import warnings
from functools import partial
from typing import Any, Dict, List, Optional

import timm
import torch
import torch.distributed._shard
import torch.nn as nn
from fvcore.common.checkpoint import (
    Checkpointer,
    _IncompatibleKeys,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp._runtime_utils import _reshard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.serialization import (
    FILE_LIKE,
    MAP_LOCATION,
    StorageType,
    _check_dill_version,
    _get_restore_location,
    _is_torchscript_zip,
    _is_zipfile,
    _legacy_load,
    _maybe_decode_ascii,
    _open_file_like,
    _open_zipfile_reader,
    _weights_only_unpickler,
)
from tqdm.auto import tqdm

import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.models import build_model_from_cfg
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup

torch.backends.cuda.matmul.allow_tf32 = (
    True  # PyTorch 1.12 sets this to False by default
)
logger = logging.getLogger("dinov2")


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x):
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    if is_sharded_fsdp(x):
        handles = x._handles
        true_list = [True for h in handles]
        _reshard(x, handles, true_list)


def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.
        Args:
            name (str): name of the file or path.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}

        try:
            from torch.distributed.checkpoint.state_dict import get_state_dict

            data["model"] = get_state_dict(self.model)
        except (ImportError, TypeError):
            try:
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                    data["model"] = self.model.state_dict()
            except Exception as e:
                print(f"Warning: Using direct state_dict() due to error: {e}")
                data["model"] = self.model.state_dict()

        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        if "model" in data:
            for key in list(data["model"].keys()):
                if "_flat_param" in key:
                    del data["model"][key]

        if os.path.isabs(name):
            save_file = f"{name}.{rankstr()}.pth"
            save_dir = os.path.dirname(name)
        else:
            basename = f"{name}.{rankstr()}.pth"
            save_file = os.path.join(self.save_dir, basename)
            save_dir = self.save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.logger.info(f"Saving checkpoint to {save_file}")
        with open(save_file, "wb") as f:
            torch.save(data, f)

        if save_dir == self.save_dir:
            self.tag_last_checkpoint(os.path.basename(save_file))

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.
        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


ShardedGradScaler = ShardedGradScaler


class ShardedTensor(torch.distributed._shard.sharded_tensor.api.ShardedTensor):
    def __setstate__(self, state):
        (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        ) = state


def _load_monke(
    zip_file, map_location, pickle_module, pickle_file="data.pkl", **pickle_load_args
):
    restore_location = _get_restore_location(map_location)
    loaded_storages = {}

    def load_tensor(dtype, numel, key, location):
        name = f"data/{key}"
        storage = (
            zip_file.get_storage_from_record(name, numel, torch.UntypedStorage)
            ._typed_storage()
            ._untyped_storage
        )
        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location),
            dtype=dtype,
            _internal=True,
        )
        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage
        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]
        assert typename == "storage", (
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        )
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype
        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = load_tensor(
                dtype, nbytes, key, _maybe_decode_ascii(location)
            )
        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor",
        "torch.distributed._shard.sharded_tensor.api": __name__,
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            # print(mod_name, name)
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record(pickle_file))
    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
    torch._utils._validate_loaded_sparse_tensors()
    return result


def torch_load_monke(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    **pickle_load_args: Any,
) -> Any:
    torch._C._log_api_usage_once("torch.load")
    UNSAFE_MESSAGE = (
        "Weights only load failed. Re-running `torch.load` with `weights_only` set to `False`"
        " will likely succeed, but it can result in arbitrary code execution."
        "Do it only if you get the file from a trusted source. WeightsUnpickler error: "
    )
    # Add ability to force safe only weight loads via environment variable
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in [
        "1",
        "y",
        "yes",
        "true",
    ]:
        weights_only = True
    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
    else:
        if pickle_module is None:
            pickle_module = pickle
    _check_dill_version(pickle_module)
    if "encoding" not in pickle_load_args.keys():
        pickle_load_args["encoding"] = "utf-8"
    with _open_file_like(f, "rb") as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "'torch.load' received a zip file that looks like a TorchScript archive"
                        " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                        " silence this warning)",
                        UserWarning,
                    )
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if weights_only:
                    try:
                        return _load_monke(
                            opened_zipfile,
                            map_location,
                            _weights_only_unpickler,
                            **pickle_load_args,
                        )
                    except RuntimeError as e:
                        raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
                return _load_monke(
                    opened_zipfile, map_location, pickle_module, **pickle_load_args
                )
        assert False, "Unreachable code path, legacy_load not implemented"
        if weights_only:
            try:
                return _legacy_load(
                    opened_file,
                    map_location,
                    _weights_only_unpickler,
                    **pickle_load_args,
                )
            except RuntimeError as e:
                raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
        return _legacy_load(
            opened_file, map_location, pickle_module, **pickle_load_args
        )


def recursive_fuse(shards, current_key=None):
    print(f"Current key: {current_key}")
    print(f"Current shard types: {[type(s) for s in shards]}")

    if isinstance(shards[0], ShardedTensor):
        return shards[0]
    elif isinstance(shards[0], torch.Tensor):
        print(f"Processing tensor for key: {current_key}")
        for i, s in enumerate(shards):
            print(f"Shard {i} shape: {s.shape}, numel: {s.numel()}")

        # get the expected shape from model's state dict if possible
        total_elements = sum(s.numel() for s in shards if s.numel() > 0)
        print(f"Total elements across shards: {total_elements}")

        # concatenate all non-empty shards
        non_empty_shards = [s for s in shards if s.numel() > 0]
        if not non_empty_shards:
            return torch.tensor([], device=shards[0].device)

        try:
            # first concatenate all shards
            concatenated = torch.cat([s.flatten() for s in non_empty_shards])
            print(f"Concatenated shape: {concatenated.shape}")
            return concatenated
        except Exception as e:
            print(f"Error during concatenation for {current_key}: {e}")
            # if concatenation fails, return the first non-empty shard
            return non_empty_shards[0]

    elif isinstance(shards[0], dict):
        print(f"Processing dictionary with {len(shards)} shards")
        assert all(isinstance(s, dict) for s in shards)
        all_keys = set.union(*map(lambda s: set(s.keys()), shards))
        print(f"Found keys: {all_keys}")

        return {
            k: recursive_fuse([s[k] for s in shards if k in s], current_key=k)
            for k in all_keys
        }

    else:
        print(f"Fallback case for type: {type(shards[0])}")
        assert all(s == shards[0] for s in shards)
        return shards[0]


class AntiFSDPCheckpointer(FSDPCheckpointer):
    def load(
        self, path: str, checkpointables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load from the given checkpoint.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}

        # Handle glob pattern for multiple checkpoints
        all_paths = glob.glob(path)
        if not all_paths:
            raise FileNotFoundError(f"No checkpoints found matching pattern: {path}")

        self.logger.info("[Checkpointer] Loading from {} ...".format(all_paths))

        # load and fuse all shards
        shards = []
        for path in tqdm(all_paths):
            shards.append(torch_load_monke(path, map_location="cpu"))
        checkpoint = recursive_fuse(shards)

        try:
            # from torch.distributed.checkpoint.state_dict import set_state_dict

            incompatible = self._load_model(checkpoint)
        except ImportError:
            # fallback to old method
            try:
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                    incompatible = self._load_model(checkpoint)
            except Exception as e:
                print(f"Warning: Using direct load due to error: {e}")
                incompatible = self._load_model(checkpoint)

        if incompatible is not None:
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []

        # handle reshaping
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_shape = model_state_dict[k].shape
                checkpoint_tensor = checkpoint_state_dict[k]

                if checkpoint_tensor.numel() == model_state_dict[k].numel():
                    try:
                        checkpoint_state_dict[k] = checkpoint_tensor.reshape(
                            model_shape
                        )
                        print(
                            f"Successfully reshaped {k} from {checkpoint_tensor.shape} to {model_shape}"
                        )
                    except Exception as e:
                        print(f"Failed to reshape {k}: {e}")
                        incorrect_shapes.append(
                            (k, checkpoint_tensor.shape, model_shape)
                        )
                        checkpoint_state_dict.pop(k)
                else:
                    print(f"Cannot reshape {k} - different number of elements")
                    incorrect_shapes.append((k, checkpoint_tensor.shape, model_shape))
                    checkpoint_state_dict.pop(k)

        # load with strict=False to allow partial loading
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)

        # create a new _IncompatibleKeys object with all three attributes
        return type(
            "_IncompatibleKeys",
            (),
            {
                "missing_keys": incompatible.missing_keys,
                "unexpected_keys": incompatible.unexpected_keys,
                "incorrect_shapes": incorrect_shapes,
            },
        )


###below are functions to load the model depending on model _cfg
def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def main(args, save_path):
    cfg = setup(args)

    student, teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    model = SSLMetaArch(
        cfg=cfg, student_backbone=student, teacher_backbone=teacher, embed_dim=embed_dim
    ).to(torch.device("cuda"))

    # create the AntiFSDPCheckpointer
    checkpointer = AntiFSDPCheckpointer(
        model=model,
        save_dir=save_path,  # Use the provided save_path
        save_to_disk=True,
    )

    # load the checkpoint
    # checkpoint_path = "/mnt/storage1/shared_data/plant_clef_2025/moved_models/dinov2_no_labels_pretrain/model-only-classifier-then-all-bs32-ep81-lr8e-5-lucas/model_0101249.**.pth"
    checkpoint_path = "/mnt/storage1/shared_data/plant_clef_2025/moved_models/dinov2_no_labels_pretrain/model-only-classifier-then-all-bs32-ep100-lr1e-3-lucas/model_0074999.**.pth"
    checkpointer.load(checkpoint_path)

    # # checkpointer.save(save_name)

    timm_model = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=7806,
    )

    # rename the keys in the state dict replace every backbone. with "
    teacher_state_dict = checkpointer.model.teacher.state_dict()
    for k in list(teacher_state_dict.keys()):
        if "backbone." in k:
            new_key = k.replace("backbone.", "")
            teacher_state_dict[new_key] = teacher_state_dict[k]
            del teacher_state_dict[k]

    teacher_state_dict["reg_token"] = teacher_state_dict.pop("register_tokens")
    placeholder_layer = nn.Linear(in_features=768, out_features=7806)
    teacher_state_dict["head.weight"] = placeholder_layer.weight
    teacher_state_dict["head.bias"] = placeholder_layer.bias
    teacher_state_dict["pos_embed"] = teacher_state_dict["pos_embed"][:, 1:, :]

    for k in [
        "dino_head.mlp.0.weight",
        "dino_head.mlp.0.bias",
        "dino_head.mlp.2.weight",
        "dino_head.mlp.2.bias",
        "dino_head.mlp.4.weight",
        "dino_head.mlp.4.bias",
        "dino_head.last_layer.weight_g",
        "dino_head.last_layer.weight_v",
        "mask_token",
    ]:
        if k in teacher_state_dict:
            del teacher_state_dict[k]

    timm_model.load_state_dict(teacher_state_dict)

    # save the unsharded model
    # save_name = "unsharded_model-only-classifier-then-all-bs32-ep81-lr8e-5-lucas_model_0101249.pth"
    save_name = (
        "model-only-classifier-then-all-bs32-ep100-lr1e-3-lucas_model_0074999.pth"
    )

    torch.save(model.state_dict(), os.path.join(save_path, save_name))


if __name__ == "__main__":
    # uv run merge_fsdp_checkpoint.py --config-file /mnt/storage1/hherasimchyk/kaggle/dinov2/dinov2/configs/train/vitb14reg4_lucas.yaml

    save_path = "/mnt/storage1/shared_data/plant_clef_2025/unsharded_moved_models/"
    os.makedirs(save_path, exist_ok=True)
    args = get_args_parser(add_help=True).parse_args()
    main(args, save_path=save_path)
