# coding: utf-8
import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from controlnet_aux import MLSDdetector, HEDdetector

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.llms import AzureOpenAI
import json
# from compel import Compel
# Grounding DINO
# import groundingdino.datasets.transforms as T
import transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import wget
import bingapi, prompt_template
import openai

os.makedirs('image', exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def blend_gt2pt(old_image, new_image, sigma=0.15, steps=100):
    new_size = new_image.size
    old_size = old_image.size
    easy_img = np.array(new_image)
    gt_img_array = np.array(old_image)
    pos_w = (new_size[0] - old_size[0]) // 2
    pos_h = (new_size[1] - old_size[1]) // 2

    kernel_h = cv2.getGaussianKernel(old_size[1], old_size[1] * sigma)
    kernel_w = cv2.getGaussianKernel(old_size[0], old_size[0] * sigma)
    kernel = np.multiply(kernel_h, np.transpose(kernel_w))

    kernel[steps:-steps, steps:-steps] = 1
    kernel[:steps, :steps] = kernel[:steps, :steps] / kernel[steps - 1, steps - 1]
    kernel[:steps, -steps:] = kernel[:steps, -steps:] / kernel[steps - 1, -(steps)]
    kernel[-steps:, :steps] = kernel[-steps:, :steps] / kernel[-steps, steps - 1]
    kernel[-steps:, -steps:] = kernel[-steps:, -steps:] / kernel[-steps, -steps]
    kernel = np.expand_dims(kernel, 2)
    kernel = np.repeat(kernel, 3, 2)

    weight = np.linspace(0, 1, steps)
    top = np.expand_dims(weight, 1)
    top = np.repeat(top, old_size[0] - 2 * steps, 1)
    top = np.expand_dims(top, 2)
    top = np.repeat(top, 3, 2)

    weight = np.linspace(1, 0, steps)
    down = np.expand_dims(weight, 1)
    down = np.repeat(down, old_size[0] - 2 * steps, 1)
    down = np.expand_dims(down, 2)
    down = np.repeat(down, 3, 2)

    weight = np.linspace(0, 1, steps)
    left = np.expand_dims(weight, 0)
    left = np.repeat(left, old_size[1] - 2 * steps, 0)
    left = np.expand_dims(left, 2)
    left = np.repeat(left, 3, 2)

    weight = np.linspace(1, 0, steps)
    right = np.expand_dims(weight, 0)
    right = np.repeat(right, old_size[1] - 2 * steps, 0)
    right = np.expand_dims(right, 2)
    right = np.repeat(right, 3, 2)

    kernel[:steps, steps:-steps] = top
    kernel[-steps:, steps:-steps] = down
    kernel[steps:-steps, :steps] = left
    kernel[steps:-steps, -steps:] = right

    pt_gt_img = easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]]
    gaussian_gt_img = kernel * gt_img_array + (1 - kernel) * pt_gt_img  # gt img with blur img
    gaussian_gt_img = gaussian_gt_img.astype(np.int64)
    easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]] = gaussian_gt_img
    gaussian_img = Image.fromarray(easy_img)
    return gaussian_img


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)


def LoraAgent(text):
    openai.api_key = openai_api_key
    openai_history = []
    input_text = 'The user-inputted scene description:' + text
    openai_history.append({"role": "system", "content": prompt_template.LORA_AGENT_PREFIX})
    openai_history.append({"role": "user", "content": input_text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=openai_history
    )
    chat_message = response['choices'][0]['message']['content']
    print(f"\n LoRA : {chat_message}")
    return chat_message


class InstructPix2Pix:
    def __init__(self, device):
        print(f"Initializing InstructPix2Pix to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           safety_checker=None,
                                                                           torch_dtype=self.torch_dtype).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @prompts(name="Instruct Image Using Text",
             description="useful when you want to the style of the image to be like the text. "
                         "like: make it look like a painting. or make it like a robot. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the text. ")
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        print(f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=self.torch_dtype)
        self.pipe.to(device)
        self.quality_prompt = 'High resolution, HDR,  best quality'
        self.n_quality_prompt = ''

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.quality_prompt
        image = self.pipe(prompt, negative_prompt=self.n_quality_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the canny image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        print(f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}")
        return updated_image_path


class CannyText2Image:
    def __init__(self, device):
        print(f"Initializing CannyText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-canny",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)

        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt = 'High resolution, HDR,  best quality'
        self.n_quality_prompt = ''
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Canny Image",
             description="useful when you want to generate a new remote sensing image from both the user description and a canny image."
                         " like: generate a real image of a object or something from this canny image,"
                         " or generate a new remote sensing image of a object or something from this edge image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        image.save(updated_image_path)
        print(f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    @prompts(name="Line Detection On Image",
             description="useful when you want to detect the straight line of the image. "
                         "like: detect the straight lines of this image, or straight line detection on image, "
                         "or perform straight line detection on this image, or detect the straight line image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        mlsd = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        print(f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}")
        return updated_image_path


class LineText2Image:
    def __init__(self, device):
        print(f"Initializing LineText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-mlsd",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt ='High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Line Image",
             description="useful when you want to generate a new remote sensing image from both the user description "
                         "and a straight line image. "
                         "like: generate a real image of a object or something from this straight line image, "
                         "or generate a new remote sensing image of a object or something from this straight lines. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description. ")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        image.save(updated_image_path)
        print(f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
              f"Output Text: {updated_image_path}")
        return updated_image_path


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    @prompts(name="Hed Detection On Image",
             description="useful when you want to detect the soft hed boundary of the image. "
                         "like: detect the soft hed boundary of this image, or hed boundary detection on image, "
                         "or perform hed boundary detection on this image, or detect soft hed boundary image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        hed = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        hed.save(updated_image_path)
        print(f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}")
        return updated_image_path


class HedText2Image:
    def __init__(self, device):
        print(f"Initializing HedText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-hed",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt ='High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Soft Hed Boundary Image",
             description="useful when you want to generate a new remote sensing image from both the user description "
                         "and a soft hed boundary image. "
                         "like: generate a real image of a object or something from this soft hed boundary image, "
                         "or generate a new remote sensing image of a object or something from this hed boundary. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        image.save(updated_image_path)
        print(f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')

    @prompts(name="Sketch Detection On Image",
             description="useful when you want to generate a scribble of the image. "
                         "like: generate a scribble of this image, or generate a sketch from this image, "
                         "detect the sketch from this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        scribble.save(updated_image_path)
        print(f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}")
        return updated_image_path


class ScribbleText2Image:
    def __init__(self, device):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-scribble",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt = 'High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Sketch Image",
             description="useful when you want to generate a new remote sensing image from both the user description and "
                         "a scribble image or a sketch image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        image.save(updated_image_path)
        print(f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path

class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline('depth-estimation')

    @prompts(name="Predict Depth On Image",
             description="useful when you want to detect depth of the image. like: generate the depth from this image, "
                         "or detect the depth map on this image, or predict the depth for this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)['depth']
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        depth.save(updated_image_path)
        print(f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class DepthText2Image:
    def __init__(self, device):
        print(f"Initializing DepthText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt ='High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Depth",
             description="useful when you want to generate a new remote sensing image from both the user description and depth image. "
                         "like: generate a real image of a object or something from this depth image, "
                         "or generate a new remote sensing image of a object or something from the depth map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        image.save(updated_image_path)
        print(f"\nProcessed DepthText2Image, Input Depth: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        self.bg_threhold = 0.4

    @prompts(name="Predict Normal Map On Image",
             description="useful when you want to detect norm map of the image. "
                         "like: generate normal map from this image, or predict normal map of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image.save(updated_image_path)
        print(f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}")
        return updated_image_path


class NormalText2Image:
    def __init__(self, device):
        print(f"Initializing NormalText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-normal", torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt ='High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Normal Map",
             description="useful when you want to generate a new remote sensing image from both the user description and normal map. "
                         "like: generate a real image of a object or something from this normal map, "
                         "or generate a new remote sensing image of a object or something from the normal map. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        image.save(updated_image_path)
        print(f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class VisualQuestionAnswering:
    def __init__(self, device):
        print(f"Initializing VisualQuestionAnswering to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Answer Question About The Image",
             description="useful when you need an answer for a question based on an image. "
                         "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the question")
    def inference(self, inputs):
        image_path, question = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
              f"Output Answer: {answer}")
        return answer

class SegText2Image:
    def __init__(self, device):
        print(f"Initializing SegText2Image to {device}")
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.seed = -1
        self.quality_prompt ='High resolution, HDR, best quality'
        self.n_quality_prompt = ' '
        self.class_prompt = 'remote sensing, birds_eye_view'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new remote sensing image from both the user description and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new remote sensing image of a object or something from these segmentations. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f'{self.class_prompt}, {instruct_text}, {self.quality_prompt}'
        image = self.pipe(prompt, image, num_inference_steps=25, eta=0.0, negative_prompt=self.n_quality_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path

class Segmenting:
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints", "sam")

        self.download_parameters()
        self.sam = build_sam(checkpoint=self.model_checkpoint_path).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([1])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 1])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def get_mask_with_boxes(self, image_pil, image, boxes_filt):

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks

    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt, pred_phrases):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        updated_image_path = get_new_image_name(image_path, func_name="segmentation")
        plt.axis('off')
        plt.savefig(
            updated_image_path,
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        return updated_image_path

    @prompts(name="Segment the Image",
             description="useful when you want to segment all the part of the image, but not segment a certain object."
                         "like: segment all the object in this image, or generate segmentations on this image, "
                         "or segment the image,"
                         "or perform segmentation on this image, "
                         "or segment all the object in this image."
                         "The input to this tool should be a string, representing the image_path")
    def inference_all(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(16, 16))
        img = np.ones((image.shape[0], image.shape[1], 3))
        plt.imshow(img)
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m)))

        updated_image_path = get_new_image_name(image_path, func_name="segment-image")
        plt.axis('off')
        plt.savefig(
            updated_image_path,
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        return updated_image_path


class Text2Box:
    def __init__(self, device):
        print(f"Initializing ObjectDetection to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints", "groundingdino")
        self.model_config_path = os.path.join("checkpoints", "grounding_config.py")
        self.download_parameters()
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.grounding = (self.load_model()).to(self.device)

    def download_parameters(self):
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url, out=self.model_checkpoint_path)
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        if not os.path.exists(self.model_config_path):
            wget.download(config_url, out=self.model_config_path)

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image
        transform = T.Compose(
            [
                T.RandomResize([512], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_boxes(self, image, caption, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.grounding.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=2)

        return image_pil, mask

    @prompts(name="Detect the Give Object",
             description="useful when you only want to detect or find out given objects in the picture"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.load_image(image_path)

        boxes_filt, pred_phrases = self.get_grounding_boxes(image, det_prompt)

        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases, }

        image_with_box = self.plot_boxes_to_image(image_pil, pred_dict)[0]

        updated_image_path = get_new_image_name(image_path, func_name="detect-something")
        updated_image = image_with_box.resize(size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ObejectDetecting, Input Image: {image_path}, Object to be Detect {det_prompt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class Inpainting:
    def __init__(self, device):
        self.device = device
        self.revision = 'fp16' if 'cuda' in self.device else None
        self.torch_dtype = torch.float16 if 'cuda' in self.device else torch.float32

        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)

    def __call__(self, prompt, image, mask_image, height=512, width=512, num_inference_steps=50):
        update_image = self.inpaint(prompt=prompt, image=image.resize((width, height)),
                                    mask_image=mask_image.resize((width, height)), height=height, width=width,
                                    num_inference_steps=num_inference_steps).images[0]
        return update_image

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([512], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    @prompts(name="Inpainting the Image",
             description="Useful when you want to inpaint only the image within the masked area."
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, mask_path, the textual description used to describe new image.")
    def inference(self, inputs):
        image_path, mask_path, prompt = inputs.split(",")
        print(f"\nimage_path={image_path}, mask_path={mask_path}, text_prompt={prompt}")
        image = Image.open(image_path).convert("RGB").resize((512, 512))  # load image
        mask_image = Image.open(mask_path).convert("L").resize((512, 512))  # load image
        update_image = self(image, mask_image, prompt)
        updated_image_path = get_new_image_name(update_image, func_name="Inpainting")
        update_image.save(updated_image_path)
        print(
            f"\nProcessed Inpainting, Input Image: {image_path},Input Mask: {mask_path}, Prompt {prompt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class InfinityOutPainting:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, ImageCaptioning, Inpainting, VisualQuestionAnswering):
        self.llm = OpenAI(temperature=0)
        self.ImageCaption = ImageCaptioning
        self.inpaint = Inpainting
        self.ImageVQA = VisualQuestionAnswering
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'fewer digits, cropped, worst quality, low quality'

    def get_BLIP_vqa(self, image, question):
        inputs = self.ImageVQA.processor(image, question, return_tensors="pt").to(self.ImageVQA.device,
                                                                                  self.ImageVQA.torch_dtype)
        out = self.ImageVQA.model.generate(**inputs)
        answer = self.ImageVQA.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed VisualQuestionAnswering, Input Question: {question}, Output Answer: {answer}")
        return answer

    def get_BLIP_caption(self, image):
        inputs = self.ImageCaption.processor(image, return_tensors="pt").to(self.ImageCaption.device,
                                                                            self.ImageCaption.torch_dtype)
        out = self.ImageCaption.model.generate(**inputs)
        BLIP_caption = self.ImageCaption.processor.decode(out[0], skip_special_tokens=True)
        return BLIP_caption

    def check_prompt(self, prompt):
        check = f"Here is a paragraph with adjectives. " \
                f"{prompt} " \
                f"Please change all plural forms in the adjectives to singular forms. "
        return self.llm(check)

    def get_imagine_caption(self, image, imagine):
        BLIP_caption = self.get_BLIP_caption(image)
        background_color = self.get_BLIP_vqa(image, 'what is the background color of this image')
        style = self.get_BLIP_vqa(image, 'what is the style of this image')
        imagine_prompt = f"let's pretend you are an excellent painter and now " \
                         f"there is an incomplete painting with {BLIP_caption} in the center, " \
                         f"please imagine the complete painting and describe it" \
                         f"you should consider the background color is {background_color}, the style is {style}" \
                         f"You should make the painting as vivid and realistic as possible" \
                         f"You can not use words like painting or picture" \
                         f"and you should use no more than 50 words to describe it"
        caption = self.llm(imagine_prompt) if imagine else BLIP_caption
        caption = self.check_prompt(caption)
        print(f'BLIP observation: {BLIP_caption}, ChatGPT imagine to {caption}') if imagine else print(
            f'Prompt: {caption}')
        return caption

    def resize_image(self, image, max_size=1000000, multiple=8):
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(math.sqrt(max_size * aspect_ratio))
        new_height = int(new_width / aspect_ratio)
        new_width, new_height = new_width - (new_width % multiple), new_height - (new_height % multiple)
        return image.resize((new_width, new_height))

    def dowhile(self, original_img, tosize, expand_ratio, imagine, usr_prompt):
        old_img = original_img
        while (old_img.size != tosize):
            prompt = self.check_prompt(usr_prompt) if usr_prompt else self.get_imagine_caption(old_img, imagine)
            crop_w = 15 if old_img.size[0] != tosize[0] else 0
            crop_h = 15 if old_img.size[1] != tosize[1] else 0
            old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))
            temp_canvas_size = (expand_ratio * old_img.width if expand_ratio * old_img.width < tosize[0] else tosize[0],
                                expand_ratio * old_img.height if expand_ratio * old_img.height < tosize[1] else tosize[
                                    1])
            temp_canvas, temp_mask = Image.new("RGB", temp_canvas_size, color="white"), Image.new("L", temp_canvas_size,
                                                                                                  color="white")
            x, y = (temp_canvas.width - old_img.width) // 2, (temp_canvas.height - old_img.height) // 2
            temp_canvas.paste(old_img, (x, y))
            temp_mask.paste(0, (x, y, x + old_img.width, y + old_img.height))
            resized_temp_canvas, resized_temp_mask = self.resize_image(temp_canvas), self.resize_image(temp_mask)
            image = self.inpaint(prompt=prompt, image=resized_temp_canvas, mask_image=resized_temp_mask,
                                 height=resized_temp_canvas.height, width=resized_temp_canvas.width,
                                 num_inference_steps=50).resize(
                (temp_canvas.width, temp_canvas.height), Image.ANTIALIAS)
            image = blend_gt2pt(old_img, image)
            old_img = image
        return old_img

    @prompts(name="Extend An Image",
             description="useful when you need to extend an image into a larger image."
                         "like: extend the image into a resolution of 2048x1024, extend the image into 2048x1024. "
                         "The input to this tool should be a comma separated string of two, representing the image_path and the resolution of widthxheight")
    def inference(self, inputs):
        image_path, resolution = inputs.split(',')
        width, height = resolution.split('x')
        tosize = (int(width), int(height))
        image = Image.open(image_path)
        image = ImageOps.crop(image, (10, 10, 10, 10))
        out_painted_image = self.dowhile(image, tosize, 4, True, False)
        updated_image_path = get_new_image_name(image_path, func_name="outpainting")
        out_painted_image.save(updated_image_path)
        print(f"\nProcessed InfinityOutPainting, Input Image: {image_path}, Input Resolution: {resolution}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class ObjectSegmenting:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting):
        # self.llm = OpenAI(temperature=0)
        self.grounding = Text2Box
        self.sam = Segmenting

    @prompts(name="Segment the given object",
             description="useful when you only want to segment the certain objects in the picture"
                         "according to the given text"
                         "like: segment the cat,"
                         "or can you segment an obeject for me"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, det_prompt)
        updated_image_path = self.sam.segment_image_with_boxes(image_pil, image_path, boxes_filt, pred_phrases)
        print(
            f"\nProcessed ObejectSegmenting, Input Image: {image_path}, Object to be Segment {det_prompt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class BingMap:
    def __init__(self, device):
        print(f"Initializing BingMap to {device}")

    @prompts(name="Fetch Map From Internet",
             description="Useful when you want to fetch a specific satellite image of a map from internet according to the given input. "
                         "like: 'Fetch a satellite image of a certain location', 'Can you fetch an aerial map for the location '47.610,-122.107'?"
                         "The input to this tool should be a comma-separated string of three values, representing the center point coordinates "
                         "latitude and longitude, followed by the zoom level. Example: 47.610,-122.107,15. "
                         "When the zoom level is not specified, it defaults to 15. "
                         "When the location is not specified, it defaults to 47.610,-122.107.")
    def inference(self, inputs):

        """
        Fetch a static satellite image of a map from the internet using Bing Maps API.

        Args:
            - inputs (str): The input string representing the center point and the zoom level, separated by a comma.
        Returns:
            - image_filename (str): The filename of the saved image.
        """
        center_point, zoom_level = self.parse_inputs(inputs)
        image_data = bingapi.get_static_map_image(center_point=center_point, zoom_level=zoom_level)
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        image_data.save(image_filename)
        print(
            f"\nProcessed BingMap, Input Text: {inputs}, Output Image: {image_filename}")
        return image_filename

    def parse_inputs(self, inputs):
        """
        Parse the input string to extract the center point and zoom level.

        Args:
            - inputs (str): The input string representing the center point and the zoom level, separated by a comma.

        Returns:
            - center_point (str): The center point of the map in the format "latitude,longitude".
            - zoom_level (int): The zoom level of the map.
        """
        inputs = inputs.split(",")
        if len(inputs) != 3:
            raise ValueError("Invalid input format. Expected 'latitude,longitude,zoom_level'.")
        center_point = inputs[0] + "," + inputs[1]
        zoom_level = int(inputs[2])
        if zoom_level < 1 or zoom_level > 20:
            raise ValueError("Invalid zoom level. Expected a value between 1 and 20.")
        return center_point, zoom_level


class Text2RSImage:
    def __init__(self, device):
        print(f"Initializing Text2RSImage to {device}")
        self.model_path = "./checkpoints/Text2RSImage-1-5"
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.unet.load_attn_procs(self.model_path)
        self.pipe.to(device)
        self.class_prompt = 'remote sensing, birds_eye_view'
        # self.quality_prompt = 'High resolution, HDR, 4K, best quality, highly detailed, sharp focus'
        # self.n_quality_prompt = ''
        self.n_quality_prompt = ' '
        self.quality_prompt = 'High resolution, best quality, sharp focus'

        # self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

    @prompts(name="Generate Remote Sensing Image From User Input Text",
             description="useful when you want to generate an remote sensing image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print(f"\n Basci Prompt : {text}")

        global PROMPT_AGENT_STRADEGY
        mode = PROMPT_AGENT_STRADEGY
        global QUALITY_PROMPT_INSTRUCT
        self.quality_prompt = QUALITY_PROMPT_INSTRUCT
        print(f"\n Quality prompt: {self.quality_prompt}")
        print(f"\n PROMPT_AGENT_STRADEGY: {mode}")
        if mode == 'Input-output':
            positive_prompt, negative_prompt = self.input_output_prompt(text)
        elif mode == 'Chain-of-thought':
            positive_prompt, negative_prompt = self.chain_of_thought_prompt(text)
        elif mode == 'Self-consistency with CoT':
            positive_prompt, negative_prompt = self.self_consistency_prompts(text)
        elif mode == 'Brain-storming':
            positive_prompt, negative_prompt = self.brain_storming_prompts(text)
        else:
            positive_prompt, negative_prompt = self.input_output_prompt(text)

        global DYNAMIC_LORA_STRADEGY
        if DYNAMIC_LORA_STRADEGY == 'Dynamic':
            lora = LoraAgent(positive_prompt)
            self.model_path = "./checkpoints/" + lora
            self.pipe.unet.load_attn_procs(self.model_path)
            print(f"\n Load Lora: {lora}")

        positive_prompt = self.class_prompt + ', ' + positive_prompt + ', ' + self.quality_prompt

        global NEGATIVE_AGENT_STRADEGY
        if negative_prompt == None or NEGATIVE_AGENT_STRADEGY == False:
            negative_prompt = self.n_quality_prompt
        else:
            negative_prompt = negative_prompt + ', ' + self.n_quality_prompt

        print(f"\n Positive prompt: {positive_prompt}")
        print(f"\n Negative prompt: {negative_prompt}")
        image = self.pipe(positive_prompt, negative_prompt=negative_prompt, num_inference_steps=25, guidance_scale=7.5,
                          cross_attention_kwargs={"scale": 0.5}).images[0]
        # image = self.pipe(prompt_embeds=self.compel_proc(positive_prompt), negative_prompt=negative_prompt, 
        #                   num_inference_steps=25, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]

        image.save(image_filename)
        print(
            f"\nProcessed Text2RSImage, Input Text: {text}, Output Image: {image_filename}")
        return image_filename

    def extract_prompts(self, text):
        positive_prompts = ""
        negative_prompts = ""

        lines = text.split('\n')
        for line in lines:
            if line.startswith("Positive Prompts:"):
                positive_prompts = line.replace("Positive Prompts:", "").strip()
            elif line.startswith("Negative Prompts:"):
                negative_prompts = line.replace("Negative Prompts:", "").strip()

        return positive_prompts, negative_prompts

    def improve_diffusion_prompt(self, text):
        openai.api_key = openai_api_key
        openai_history = []
        input_text = 'Crafting Perfect Stable Diffusion Prompts for Remote Sensing Images with basic condition:' + text
        openai_history.append({"role": "system", "content": prompt_template.DIFFUSION_CHATGPT_PROMPT})
        openai_history.append({"role": "user", "content": input_text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        df_prompt = response['choices'][0]['message']['content']
        # print(f"\nGPT Text: {df_prompt}")
        positive_prompt, negative_prompt = self.extract_prompts(df_prompt)
        return positive_prompt, negative_prompt

    def lora_agent(self, text):
        input_text = "The scene description: " + text
        openai_history = []
        openai_history.append({"role": "system", "content": prompt_template.LORA_AGENT})
        openai_history.append({"role": "user", "content": input_text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        chat_message = response['choices'][0]['message']['content']
        print(f"\n select lora : {chat_message}")
        lora_name = "./checkpoints/" + chat_message
        if os.path.isdir(lora_name):
            return lora_name
        else:
            lora_name = "./checkpoints/RSB"
        self.pipe.unet.load_attn_procs(lora_name)
        return lora_name

    def input_output_prompt(self, text):
        openai.api_key = openai_api_key
        functions = [
            {
                "name": "get_prompts",
                "description": "Get the positive and negative prompt from user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positive_prompt": {
                            "type": "string",
                            "description": "Describe the desire information of image, e.g. rive, boats",
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Describe unrealistic states of image , e.g. Twisted streets, Chaotic house structure",
                        }
                    }
                }
            }
        ]
        input_text = "Generate a remote sensing image: " + text
        messages = [{"role": "user", "content": input_text}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )
        response_message = response["choices"][0]["message"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        positive_prompt = function_args.get('positive_prompt')
        negative_prompt = function_args.get('negative_prompt')
        return positive_prompt, negative_prompt

    def crafting_negative_prompts(self, text):
        openai.api_key = openai_api_key
        openai_history = []
        openai_history.append({"role": "system", "content": prompt_template.DIFFUSION_NEGATIVE_CHATGPT_PROMPT})
        openai_history.append({"role": "user", "content": text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        chat_message = response['choices'][0]['message']['content']
        print(f"\n crafting negative prompts : {chat_message}")
        openai_history.append({"role": "user", "content": "Negative Prompt of the previous step: "})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        negative_prompts = response['choices'][0]['message']['content']
        return negative_prompts

    def chain_of_thought_prompt(self, text):
        openai.api_key = openai_api_key
        openai_history = []
        input_text = 'Crafting Stable Diffusion Prompts for Remote Sensing Images with basic condition:' + text
        openai_history.append({"role": "system", "content": prompt_template.DIFFUSION_CoT_PROMPT})
        openai_history.append({"role": "user", "content": input_text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        chat_message = response['choices'][0]['message']['content']
        print(f"\nChain of thought : {chat_message}")
        openai_history.append({"role": "user", "content": "Output only the Prompt of the previous step:"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        positive_prompt = response['choices'][0]['message']['content']

        negative_prompt = self.crafting_negative_prompts(positive_prompt)
        return positive_prompt, negative_prompt

    def self_consistency_prompts(self, text):
        openai.api_key = openai_api_key
        openai_history = []
        openai_history.append({"role": "system", "content": prompt_template.DIFFUSION_SC_PROMPT})
        openai_history.append({"role": "user", "content": text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        chat_message = response['choices'][0]['message']['content']
        print(f"\nBrain Storming: {chat_message}")
        openai_history.append({"role": "user", "content": "Output only the final Prompt of the previous step:"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        positive_prompt = response['choices'][0]['message']['content']
        negative_prompt = self.crafting_negative_prompts(positive_prompt)
        return positive_prompt, negative_prompt

    def brain_storming_prompts(self, text):
        openai.api_key = openai_api_key
        openai_history = []
        openai_history.append({"role": "system", "content": prompt_template.DIFFUSION_BRAIN_STORMING_PROMPT})
        openai_history.append({"role": "user", "content": text})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        chat_message = response['choices'][0]['message']['content']
        print(f"\nBrain Storming: {chat_message}")
        openai_history.append({"role": "user", "content": "Output only the final Prompt of the previous step:"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_history
        )
        positive_prompt = response['choices'][0]['message']['content']
        negative_prompt = self.crafting_negative_prompts(positive_prompt)
        return positive_prompt, negative_prompt


class ImageEditing:
    template_model = True

    def __init__(self, Text2Box: Text2Box, Segmenting: Segmenting, Inpainting: Inpainting):
        print(f"Initializing ImageEditing")
        self.sam = Segmenting
        self.grounding = Text2Box
        self.inpaint = Inpainting

    def pad_edge(self, mask, padding):
        # mask Tensor [H,W]
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        # new_mask
        return new_mask

    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the object need to be removed. ")
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        return self.inference_replace_sam(f"{image_path},{to_be_removed_txt},background")

    @prompts(name="Replace Something From The Photo",
             description="useful when you want to replace an object from the object description or "
                         "location with another object from its description. "
                         "The input to this tool should be a comma separated string of three, "
                         "representing the image_path, the object to be replaced, the object to be replaced with ")
    def inference_replace_sam(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")

        print(f"image_path={image_path}, to_be_replaced_txt={to_be_replaced_txt}")
        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, to_be_replaced_txt)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()  # tensor

        mask = self.pad_edge(mask, padding=20)  # numpy
        mask_image = Image.fromarray(mask)

        updated_image = self.inpaint(prompt=replace_with_txt, image=image_pil,
                                     mask_image=mask_image)
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image = updated_image.resize(image_pil.size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:0',...}
        print(f"Initializing Remote Sensing GPT, load_dict={load_dict}")
        # if 'ImageCaptioning' not in load_dict:
        #     raise ValueError("You have to load ImageCaptioning as a basic function")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(temperature=0)

        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self):
        self.memory.clear()  # clear previous history
        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = prompt_template.REMOTE_SENSING_AGENT_PREFIX, prompt_template.REMOTE_SENSING_AGENT_FORMAT_INSTRUCTIONS, prompt_template.REMOTE_SENSING_AGENT_SUFFIX
        place = "Enter text and press enter, or upload an image"
        label_clear = "Clear"

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible=True), gr.update(visible=False), gr.update(placeholder=place), gr.update(
            value=label_clear)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)

        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '


def change_prompt_select(text):
    prompt_select.update(value=text)
    global PROMPT_AGENT_STRADEGY
    PROMPT_AGENT_STRADEGY = text
    return


def change_lora_select(text):
    prompt_select.update(value=text)
    global DYNAMIC_LORA_STRADEGY
    DYNAMIC_LORA_STRADEGY = text
    return


def change_quality_text(text):
    # quality_prompt_text.update(value=text)
    global QUALITY_PROMPT_INSTRUCT
    QUALITY_PROMPT_INSTRUCT = text
    return


def change_negative_select(use_negative):
    global NEGATIVE_AGENT_STRADEGY
    NEGATIVE_AGENT_STRADEGY = use_negative
    return


def save_mask(image):
    mask = image["mask"].convert("RGB").resize((512, 512))
    mask.save("./image/mask.png")


if __name__ == '__main__':
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="ImageCaptioning_cuda:0, Text2RSImage_cuda:0, "
                                                    "Text2Box_cuda:0, Segmenting_cuda:0, Inpainting_cuda:0,"
                                                    "Image2Canny_cpu, CannyText2Image_cuda:0,"
                                                    "BingMap_cpu")
    # ImageCaptioning_cuda:0,
    # "Image2Hed_cpu, HedText2Image_cuda:0"
    # "Image2Canny_cpu, CannyText2Image_cuda:0"
    # "Image2Scribble_cpu, ScribbleText2Image_cuda:0"
    # "Image2Depth_cpu,DepthText2Image_cuda:0"
    # "Image2Line_cpu,LineText2Image_cuda:0
    # "Image2Normal_cpu,NormalText2Image_cuda:0"
    # "InstructPix2Pix_cuda:0 " ,
    # Text2Box_cpu, Segmenting_cuda:0, Inpainting_cuda:0
    # "ImageCaptioning_cuda:0, Inpainting_cuda:0, VisualQuestionAnswering_cuda:0"

    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    global PROMPT_AGENT_STRADEGY
    PROMPT_AGENT_STRADEGY = "Input-output"
    global NEGATIVE_AGENT_STRADEGY
    NEGATIVE_AGENT_STRADEGY = True
    global DYNAMIC_LORA_STRADEGY
    DYNAMIC_LORA_STRADEGY = "Dynamic"
    bot = ConversationBot(load_dict=load_dict)
    bot.init_agent()
    global QUALITY_PROMPT_INSTRUCT
    QUALITY_PROMPT_INSTRUCT = "High resolution, HDR, 4K"

    with gr.Blocks(theme="soft") as demo:
        state = gr.State([])
        prompt_select = gr.Radio(
            ["Input-output", "Chain-of-thought", "Self-consistency with CoT", "Brain-storming"],
            label="Diffusion Prompt Bot",
            info="Choose a prompting stradegy for image generation model", value="Input-output",
            interactive=True)
        prompt_select.change(fn=change_prompt_select, inputs=prompt_select)
        with gr.Row():
            lora_select = gr.Radio(["Dyanmic", "Stable"],
                                   label="Diffusion LoRA Bot",
                                   info="Choose a LoRA stradegy for image generation model", value="Dyanmic",
                                   interactive=True)
            lora_select.change(fn=change_lora_select, inputs=lora_select)

            negative_select = gr.Checkbox(value=False, label="Using negative prompt",
                                          info="Generate negative prompt by Agent")
            negative_select.change(fn=change_negative_select, inputs=negative_select)

        quality_prompt_text = gr.Textbox(label="Quality Prompt", show_label=True,
                                         value="High resolution, HDR, best quality"
                                         , placeholder="Enter text and press enter")
        quality_prompt_text.change(fn=change_quality_text, inputs=quality_prompt_text)

        chatbot = gr.Chatbot(elem_id="chatbot", label="Remote Sensing GPT")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image",container=False)
        with gr.Row(visible=True) as input_raws:
            with gr.Column():
                clear = gr.Button("Clear")
            with gr.Column():
                btn = gr.UploadButton(label="🖼️", file_types=["image"])

        gr.Examples(["What is inside the photo of ",
                     "Generate an remote sensing image: Mountains",
                     "Fetch a Map of a certain location: '47,-122'",
                     "Detect the {edges} of this image",
                     "Generate a new remote sensing image of {city} from this canny image",
                     "Detect the Give Object in the picture: ",
                     "Segment the car in this image",
                     "Remove Airplane From The Photo"
                     ], txt)
        with gr.Row():
            image_loader = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil",
                                    label="Upload",height=300)
            image_paint = gr.Paint(height=300)
        save_btn = gr.Button("Save Mask")
        save_btn.click(fn=save_mask, inputs=[image_loader])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="127.0.0.1", server_port=7861)
