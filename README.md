# RSAgent


## Quick Start

```
#  prepare the basic environments
pip install -r requirements.txt
pip install  git+https://github.com/IDEA-Research/GroundingDINO.git
pip install  git+https://github.com/facebookresearch/segment-anything.git

# prepare your private OpenAI key (for Linux)
export OPENAI_API_KEY={Your_Private_Openai_Key}

# prepare your private OpenAI key (for Windows)
set OPENAI_API_KEY={Your_Private_Openai_Key}

# Start!
# You can specify the GPU/CPU assignment by "--load", the parameter indicates which 
# Visual Foundation Model to use and where it will be loaded to
# The model and device are separated by underline '_', the different models are separated by comma ','
# The available Visual Foundation Models can be found in the following table
# For example, if you want to load ImageCaptioning to cpu and Text2Image to cuda:0
# You can use: "ImageCaptioning_cpu,Text2Image_cuda:0"
                        
python RSAgent.py --load "Text2Box_cuda:0,Segmenting_cuda:0,
    Inpainting_cuda:0,ImageCaptioning_cuda:0,Text2RSImage_cuda:0,Image2Canny_cpu,CannyText2Image_cuda:0
    

```

## GPU memory usage
Here we list the GPU memory usage of each visual foundation model, you can specify which one you like:

| Foundation Model          | GPU Memory (MB)   |
|---------------------------|-------------------|
| ImageEditing              | 3981              |
| InstructPix2Pix           | 2827              |
| Text2RSImage              | 3385              |
| ImageCaptioning           | 1209              |
| Image2Canny               | 0                 |
| CannyText2Image           | 3531              |
| Image2Line                | 0                 |
| LineText2Image            | 3529              |
| Image2Hed                 | 0                 |
| HedText2Image             | 3529              |
| Image2Scribble            | 0                 |
| ScribbleText2Image        | 3531              |
| Image2Pose                | 0                 |
| Image2Seg                 | 919               |
| SegText2Image             | 3529              |
| Image2Depth               | 0                 |
| DepthText2Image           | 3531              |
| Image2Normal              | 0                 |
| NormalText2Image          | 3529              |
| VisualQuestionAnswering   | 1495              |
| BingMap                   | 0                 |
| ------------------------- | ----------------- |
## Acknowledgement
We appreciate the open source of the following projects:
[Task Matrix](https://github.com/microsoft/TaskMatrix.git) &#8194;
[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
[ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
[CLIPSeg](https://github.com/timojl/clipseg) &#8194;
[BLIP](https://github.com/salesforce/BLIP) &#8194;
