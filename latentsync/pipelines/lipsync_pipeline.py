# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect  # 导入inspect模块，用于检查活动对象
import os  # 导入os模块，用于与操作系统交互
import shutil  # 导入shutil模块，用于高级文件操作
from typing import Callable, List, Optional, Union  # 导入typing模块中的类型提示
import subprocess  # 导入subprocess模块，用于生成新进程

import numpy as np  # 导入numpy模块，用于数值计算
import torch  # 导入torch模块，用于深度学习
import torchvision  # 导入torchvision模块，用于计算机视觉任务

from diffusers.utils import is_accelerate_available  # 从diffusers.utils导入is_accelerate_available函数
from packaging import version  # 导入packaging模块中的version类

from diffusers.configuration_utils import FrozenDict  # 从diffusers.configuration_utils导入FrozenDict类
from diffusers.models import AutoencoderKL  # 从diffusers.models导入AutoencoderKL类
from diffusers.pipeline_utils import DiffusionPipeline  # 从diffusers.pipeline_utils导入DiffusionPipeline类
# 从diffusers.schedulers导入多个调度器类
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging  # 从diffusers.utils导入deprecate和logging

from einops import rearrange  # 从einops导入rearrange函数
import cv2  # 导入cv2模块，用于计算机视觉任务

from ..models.unet import UNet3DConditionModel  # 从上级目录的models.unet模块导入UNet3DConditionModel类
from ..utils.image_processor import ImageProcessor  # 从上级目录的utils.image_processor模块导入ImageProcessor类
from ..utils.util import read_video, read_audio, write_video  # 从上级目录的utils.util模块导入多个函数
from ..whisper.audio2feature import Audio2Feature  # 从上级目录的whisper.audio2feature模块导入Audio2Feature类
import tqdm  # 导入tqdm模块，用于显示进度条
import soundfile as sf  # 导入soundfile模块，用于读写音频文件
import time
import pickle

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        # 检查调度器配置并进行必要的更新
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        # 检查调度器配置中的 `clip_sample` 参数并进行必要的更新
        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        # 检查 U-Net 版本和样本大小配置，并进行必要的更新
        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        # 注册模型组件
        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        # 计算 VAE 缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # 设置进度条配置
        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    """
    启用模型的顺序CPU卸载功能。

    如果安装了accelerate库，则将指定的模型（unet、text_encoder、vae）卸载到GPU上。
    如果未安装accelerate库，则抛出ImportError异常。

    参数:
        gpu_id (int): 要使用的GPU的ID，默认为0。

    异常:
        ImportError: 如果未安装accelerate库。
    """
    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        
    """
    确定模型执行的设备。

    如果模型设备不是'meta'或者模型没有'_hf_hook'属性，则返回模型当前的设备。
    否则，遍历模型的所有模块，查找具有'_hf_hook'属性且该钩子具有非空的'execution_device'属性的模块，
    并返回该设备。如果未找到，则返回模型当前的设备。

    Returns:
        torch.device: 模型执行的设备。
    """
    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        """
        解码潜在变量。

        该函数将输入的潜在变量进行解码，首先将其进行缩放和平移操作，
        然后重新排列形状，最后通过变分自编码器（VAE）进行解码并采样。

        参数:
            latents (Tensor): 输入的潜在变量张量。

        返回:
            Tensor: 解码后的潜在变量样本。
        """
        # 对潜在变量进行缩放和平移操作
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        # 重新排列潜在变量的形状
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # 使用VAE解码潜在变量并进行采样
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        为调度器步骤准备额外的关键字参数，因为不同的调度器可能具有不同的函数签名。
        参数 eta (η) 仅在使用 DDIMScheduler 时生效，其他调度器将忽略此参数。
        eta 对应于 DDIM 论文 (https://arxiv.org/abs/2010.02502) 中的 η，取值范围应在 [0, 1] 之间。

        参数:
            generator: 随机数生成器，用于生成随机数。
            eta: eta 参数，仅在使用 DDIMScheduler 时生效。

        返回:
            包含额外关键字参数的字典，用于调度器步骤。
        """
        # 检查调度器的 step 函数是否接受 eta 参数
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # 检查调度器是否接受 generator 参数
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        """
        检查输入参数的有效性。

        本函数确保输入的height和width相等且能够被8整除，同时验证callback_steps参数，
        确保其为正整数。这些检查是为了确保后续流程能够正确执行，因为可能涉及到与这些
        参数相关的操作。

        参数:
            height (int): 输入的高度值。
            width (int): 输入的宽度值。
            callback_steps (int, optional): 回调步骤的次数，必须为正整数。

        Raises:
            AssertionError: 如果height和width不相等。
            ValueError: 如果height或width不能被8整除，或者callback_steps不是正整数。
        """

        # 检查height和width是否相等
        assert height == width, "Height and width must be equal"

        # 检查height和width是否能被8整除
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查callback_steps是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        """
        准备用于视频生成模型的潜在变量。

        参数:
            batch_size (int): 批次中视频的数量。
            num_frames (int): 每个视频中的帧数。
            num_channels_latents (int): 潜在变量的通道数。
            height (int): 视频的高度。
            width (int): 视频的宽度。
            dtype (torch.dtype): 潜在变量的数据类型。
            device (torch.device): 潜在变量放置的设备。
            generator (torch.Generator): 随机数生成器。

        返回:
            torch.Tensor: 准备好的潜在变量。
        """
        # 计算潜在变量的形状
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        # 选择生成随机数的设备
        rand_device = "cpu" if device.type == "mps" else device
        # 生成初始潜在变量
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        # 在帧维度上重复潜在变量
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # 根据调度器要求的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        """
        准备用于扩散模型的掩码和掩码图像。该函数通过调整大小、编码和对齐设备及数据类型来处理掩码和掩码图像。

        参数:
            mask: 掩码图像。
            masked_image: 被掩码的图像。
            height: 图像的高度。
            width: 图像的宽度。
            dtype: 数据类型。
            device: 设备（例如：CPU 或 GPU）。
            generator: 随机数生成器，用于采样。
            do_classifier_free_guidance: 是否进行分类器自由引导。

        返回值:
            返回处理后的掩码和掩码图像的潜在表示。
        """

        # 将掩码调整为潜在变量形状，并在转换数据类型之前执行此操作以避免使用 cpu_offload 和半精度时出现问题
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # 将掩码图像编码到潜在空间，以便可以将其与潜在变量连接
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # 对齐设备以防止在连接潜在模型输入时出现设备错误
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # 假设批次大小为 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        # 如果启用分类器自由引导，则重复掩码和掩码图像潜在表示两次
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        """
        准备图像潜在变量(latents)。

        该函数将输入的图像转换为潜在变量(latents)，这是图像到图像生成过程中必要的一步。
        使用VAE（变分自编码器）对输入图像进行编码，然后将得到的潜在变量进行必要的转换，
        以适应后续生成过程的需求。

        参数:
        - images: 输入的图像数据。
        - device: 设备信息，用于指定计算是在CPU还是GPU上进行。
        - dtype: 数据类型，用于指定图像数据在进行计算时的数据精度。
        - generator: 随机数生成器，用于在潜在变量的生成过程中提供随机性。
        - do_classifier_free_guidance: 是否执行分类器自由引导，这会影响潜在变量的处理方式。

        返回:
        - image_latents: 处理后的图像潜在变量。
        """
        
        # 将图像数据移动到指定的设备和数据类型
        images = images.to(device=device, dtype=dtype)
        
        # 使用VAE的编码器将图像数据编码为潜在变量，并进行采样
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        
        # 对潜在变量进行必要的转换，以适应后续生成过程的需求
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        # 重新排列潜在变量的维度，以适应后续生成过程的需求
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        
        # 如果执行分类器自由引导，则对潜在变量进行复制
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
        
        # 返回处理后的图像潜在变量
        return image_latents

    def set_progress_bar_config(self, **kwargs):
        """
        更新进度条的配置信息
        
        该方法通过接受的关键字参数来更新进度条的配置信息如果进度条配置信息尚未初始化，
        则创建一个空字典进行初始化然后，将传入的关键字参数更新到这个配置字典中
        
        参数:
        **kwargs: 关键字参数，包含需要更新的进度条配置信息
        """
        # 检查是否已存在进度条配置信息，若不存在则初始化为空字典
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        
        # 将关键字参数更新到进度条配置信息中
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        """
        将周围像素贴回，因为我们只想改变嘴巴区域

        该函数旨在仅修改图像中的嘴巴区域，同时保留周围区域的细节。
        它通过使用掩码将解码后的潜在表示与原始像素值进行融合来实现这一点。

        参数:
        - decoded_latents: 解码后的潜在表示，表示修改后的图像部分。
        - pixel_values: 图像的原始像素值，用于贴回未更改的周围像素。
        - masks: 用于区分需要修改的区域和保持不变的区域的掩码。
        - device: 设备信息（例如 'cuda' 或 'cpu'），用于指定张量的操作设备。
        - weight_dtype: 权重数据类型，用于指定张量的数据类型。

        返回值:
        - combined_pixel_values: 融合后的像素值，其中只有嘴巴区域被修改，其余部分保持不变。
        """

        # 将像素值和掩码移动到指定设备并转换为指定数据类型
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)

        # 使用掩码融合解码后的潜在表示和原始像素值
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)

        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        """
        将像素值转换为图像格式。

        此函数的目的是将模型输出的像素值转换为易于可视化的图像格式。这包括调整数据的排列方式，
        归一化像素值到0-255的范围，并将其转换为适当的图像数据类型。

        参数:
        - pixel_values (torch.Tensor): 模型输出的像素值，期望的形状为 (f, c, h, w)。

        返回:
        - images (np.ndarray): 转换后的图像数据，数据类型为 uint8，形状为 (f, h, w, c)。
        """
        # 调整数据的维度顺序，以便符合图像数据常见的格式
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        
        # 将像素值从模型输出的格式转换为0-1范围的浮点数
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        
        # 将像素值转换为图像格式，即乘以255得到0-255范围的值，并转换为uint8类型
        images = (pixel_values * 255).to(torch.uint8)
        
        # 将数据移动到CPU并转换为numpy数组，以便于后续的处理或显示
        images = images.cpu().numpy()
        
        return images

    def affine_transform_video(self, video_path):
        """
        对视频进行仿射变换处理。
        
        读取视频中的所有帧，并对每一帧进行仿射变换处理，以检测和提取人脸。
        
        参数:
        video_path (str): 视频文件的路径。
        
        返回:
        faces (torch.Tensor): 所有检测到的人脸图像的张量。
        video_frames (list): 视频的所有帧。
        boxes (list): 所有人脸的边界框。
        affine_matrices (list): 所有的仿射变换矩阵。
        """
        # 获取视频文件所在的目录
        video_dir = os.path.dirname(video_path)
        # 使用视频文件名（不含扩展名）作为缓存文件的基础名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 创建缓存文件的完整路径
        cache_file = os.path.join(video_dir, f"{video_name}_cache.npy")
        # 检查缓存文件是否存在，如果存在则加载缓存
        if os.path.exists(cache_file):
            print(f"加载缓存文件: {cache_file}")
            with open(cache_file, "rb") as f:
                faces, video_frames, boxes, affine_matrices = pickle.load(f)
                return faces, video_frames, boxes, affine_matrices

        # 读取视频帧
        video_frames = read_video(video_path, use_decord=False)
        faces = []
        boxes = []
        affine_matrices = []
        print(f"正在对 {len(video_frames)} 张脸进行仿射变换...")
        for frame in tqdm.tqdm(video_frames):
            # 对每一帧尝试进行仿射变换，如果未检测到人脸，则跳过当前帧
            # face, box, affine_matrix = self.image_processor.affine_transform(frame)
            # faces.append(face)
            # boxes.append(box)
            # affine_matrices.append(affine_matrix)
            try:
                face, box, affine_matrix = self.image_processor.affine_transform(frame)
                faces.append(face)
                boxes.append(box)
                affine_matrices.append(affine_matrix)
            except Exception as e:
                print("未检测到人脸")

        # 将人脸图像列表转换为张量
        faces = torch.stack(faces)
        # 存储缓存
        with open(cache_file, "wb") as f:
            pickle.dump((faces, video_frames, boxes, affine_matrices), f)
        return faces, video_frames, boxes, affine_matrices

    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        """
        恢复视频帧，通过将增强后的人脸图像替换到原始视频帧中。

        该函数根据人脸边界框和仿射变换矩阵对齐增强后的人脸，并将其整合到原始视频帧中。

        参数:
        - faces: 包含增强后人脸图像的张量。
        - video_frames: 原始视频帧列表。
        - boxes: 包含人脸边界的坐标列表，每个元素为一个四元组 (x1, y1, x2, y2)。
        - affine_matrices: 仿射变换矩阵列表，用于对齐人脸在视频帧中的位置。

        返回值:
        - 恢复后的视频帧，以 numpy 数组形式返回。
        """

        # 确保视频帧数量与人脸数量匹配
        video_frames = video_frames[: faces.shape[0]]
        out_frames = []
        print(f"正在恢复 {len(faces)} 张人脸...")

        # 遍历每张人脸并进行处理
        for index, face in enumerate(tqdm.tqdm(faces)):
            # 获取当前人脸的边界框坐标
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)

            # 调整人脸图像大小以匹配边界框尺寸
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)

            # 将通道维度从 "c h w" 转换为 "h w c"
            face = rearrange(face, "c h w -> h w c")

            # 对人脸图像进行归一化处理
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()

            # 使用图像处理器修复视频帧，将增强后的人脸整合到帧中
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)

        # 将所有处理后的帧堆叠成一个 numpy 数组并返回
        return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """
        主调用方法，用于生成与音频同步的唇形动画视频。

        参数:
            video_path (str): 输入视频路径。
            audio_path (str): 输入音频路径。
            video_out_path (str): 输出视频路径。
            video_mask_path (str, optional): 输出掩码视频路径，默认为 None。
            num_frames (int): 每个推理步骤中的帧数，默认为 16。
            video_fps (int): 视频帧率，默认为 25。
            audio_sample_rate (int): 音频采样率，默认为 16000。
            height (Optional[int]): 视频高度，默认为 None。
            width (Optional[int]): 视频宽度，默认为 None。
            num_inference_steps (int): 推理步骤数量，默认为 20。
            guidance_scale (float): 引导比例，默认为 1.5。
            weight_dtype (Optional[torch.dtype]): 权重数据类型，默认为 torch.float16。
            eta (float): DDIM 调度器参数，默认为 0.0。
            mask (str): 掩码类型，默认为 "fix_mask"。
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]): 随机数生成器，默认为 None。
            callback (Optional[Callable[[int, int, torch.FloatTensor], None]]): 回调函数，默认为 None。
            callback_steps (Optional[int]): 回调步数，默认为 1。
            **kwargs: 其他关键字参数。

        返回:
            无返回值，直接生成输出视频文件。
        """
        print('开始运行', time.time())
        is_train = self.unet.training
        self.unet.eval()

        # 定义调用参数
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 进行仿射变换以提取面部特征
        faces, original_video_frames, boxes, affine_matrices = self.affine_transform_video(video_path)
        print('人脸提取完成', time.time())
        audio_samples = read_audio(audio_path)

        # 1. 设置默认的高度和宽度
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_rate

        # 2. 检查输入参数的有效性
        self.check_inputs(height, width, callback_steps)

        # 判断是否启用分类器自由引导
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. 准备额外的关键字参数
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        # 处理音频特征
        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
        else:
            num_inferences = len(faces) // num_frames
        # 音频特征处理非常快
        print('处理音频特征完成', time.time())

        synced_video_frames = []
        masked_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # 准备潜在变量
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )
    
        print('潜在变量完成', time.time())
        # 进行推理
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. 准备掩码潜在变量
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. 准备图像潜在变量
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. 噪声去除循环
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    # 如果我们正在进行无分类器引导，则会扩大延迟
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    # 在通道维度上连接潜变量、遮罩、掩蔽图像潜变量
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # predict the noise residual
                    # 预测噪声残差
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    # 执行指导
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # 计算前一个噪声样本 x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    # 如果提供了回调函数，则调用它
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # 恢复像素值
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            # masked_video_frames.append(masked_pixel_values)

        print('推理完成', time.time())
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        print('视频帧恢复完成', time.time())
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        # )
        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)
        # write_video(video_mask_path, masked_video_frames, fps=25)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)