import torch
import numpy as np
from flask import Flask, request
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoFrameSync, VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth

from diffusers import StableDiffusionPipeline
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import pil2tensor
from streamdiffusion.image_utils import postprocess_image
from utils.wrapper import StreamDiffusionWrapper

from PIL import Image

# pip
# diffusers 0.24.0
# huggingface-hub 0.25.2
# tokenizers 0.13.3
# torch 2.1.0+cu121
# cyndilib 5.1.1.5 - pip(x)

# OSC 및 NDI 설정
OSC_PORT = 8000
NDI_INPUT_NAME = 'TouchDesigner_Input'
NDI_OUTPUT_NAME = 'StreamDiffusion_Output'

model_path = "KBlueLeaf/kohaku-v2.1"
current_prompt = 'dog'
negative_prompt = "low quality, bad quality, blurry"
num_inference_steps = 50

stream = StreamDiffusionWrapper(
    model_id_or_path=model_path,
    lora_dict=None,
    t_index_list=[32, 45],
    frame_buffer_size=1,
    width=640,
    height=360,
    warmup=10,
    acceleration="xformers",
    do_add_noise=False,
    enable_similar_image_filter=True,
    similar_image_filter_threshold=0.99,
    similar_image_filter_max_skip_frame=10,
    mode="img2img",
    use_denoising_batch=True,
    cfg_type="self",
    seed=2,
)

stream.prepare(
    prompt=current_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=1.2,
    delta=0.5
)

app = Flask(__name__)

# pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
#     device=torch.device("cuda"),
#     dtype=torch.float16,
# )

# 값 업데이트
def update_prompt(unused_addr, *args):
    global current_prompt
    current_prompt = args[0]
    stream.stream.update_prompt(current_prompt)
    print(f"Received Prompt: {current_prompt}")
def update_step(unused_addr, *args):
    global num_inference_steps
    num_inference_steps = int(args[0])
    stream.stream.guidance_scale = num_inference_steps
    print(f"Received Denoise Step: {num_inference_steps}")

# TD OSC 수신 후 값 업데이트
dispatcher = Dispatcher()
dispatcher.map("/prompt", update_prompt)
dispatcher.map("/num_inference_steps", update_step)

# OSC 서버 실행
def start_osc_server():
    server = ThreadingOSCUDPServer(("127.0.0.1", OSC_PORT), dispatcher)
    print(f"OSC Server running on port {OSC_PORT}")
    server.serve_forever()

# TD NDI 찾기
def find_specific_ndi_source():
    finder = Finder()
    finder.open()

    elapsed_time = 0
    interval = 1 

    print(f"Waiting for NDI source with name '{NDI_INPUT_NAME}'...")

    # TD NDI 찾을때까지
    while True:
        finder.update_sources()
        sources = finder.get_source_names()
        print(f"NDI Sources Found: {sources}")

        for source_name in sources:
            print(source_name)
            if NDI_INPUT_NAME in source_name:
                return finder.get_source(source_name)

        import time
        time.sleep(interval)

# NDI 송수신 초기화
def initialize_ndi_receiver():
    source = find_specific_ndi_source()

    receiver = Receiver(
        color_format=RecvColorFormat.RGBX_RGBA,
        bandwidth=RecvBandwidth.highest,
    )
    
    receiver.set_source(source)
    
    print(f"Receiver connected to source: {source.name}")
    
    return receiver

def initialize_ndi_sender():
    sender = Sender(ndi_name=NDI_OUTPUT_NAME, clock_video=True, clock_audio=True)
    
    video_send_frame = VideoSendFrame()
    
    video_send_frame.set_resolution(640, 360)
    video_send_frame.set_frame_rate(30)
    video_send_frame.set_fourcc(FourCC.RGBA)

    sender.set_video_frame(video_send_frame)
    
    sender.open() 
    print(f"NDI Sender initialized with name: {NDI_OUTPUT_NAME}")
    
    return sender

# NDI 수신(TD -> SD)
def receive_image_from_ndi(receiver):
    video_frame = VideoFrameSync()
    receiver.frame_sync.set_video_frame(video_frame)

    if receiver.is_connected():
        receiver.frame_sync.capture_video()
        if video_frame.xres > 0 and video_frame.yres > 0:  # 유효한 해상도 확인
            frame_data = video_frame.get_array()
            frame_data = frame_data.reshape(video_frame.yres, video_frame.xres, 4)
            return Image.fromarray(frame_data[:, :, :3])
    
    return None

# NDI 전송 (SD -> TD)
def send_image_to_ndi(sender,image):
    if image is None:
        print("Received None image, skipping...")
        return
    try:
        image_resized = image.resize((640, 360)).convert("RGBA")
        
        image_np = np.array(image_resized)
        
        writable_data = image_np.flatten()  # 데이터를 flatten()으로 1차원 배열 생성
        
        #print(f"Sending data size: {writable_data.size}")

        sender.write_video(writable_data)  # write_video를 사용하여 데이터 전송
    except AttributeError:
        print("Error resizing image, possibly None")

def preprocess_input_image(input_image):
    """
    입력 이미지를 StreamDiffusion에 맞게 전처리.
    Args:
    input_image (PIL.Image): 입력 이미지.
    Returns:
    torch.Tensor: (C, H, W) 순서로 정규화된 텐서 이미지.

    Todo : 
    """
    image_np = np.array(input_image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    return image_tensor

# 이미지 변환
def transform_image_with_streamdiffusion(input_image):
    global current_prompt, negative_prompt
    try:
        input_tensor = preprocess_input_image(input_image)
        # Todo : pil2tensor 사용
        #input_tensor = pil2tensor(input_image)

        for _ in range(stream.batch_size - 1):
            stream(image=input_tensor)
        
        output = stream(image=input_tensor)
        if isinstance(output, torch.Tensor):
            return postprocess_image(output, output_type="pil")
        elif isinstance(output, Image.Image):
            return output
        else:
            print(f"Unexpected output type: {type(output)}")
            return None
    except Exception as e:
        print(f"Error in transform_image_with_streamdiffusion: {str(e)}")
        return None

def process_images(receiver, sender):
    while True:
        input_image = receive_image_from_ndi(receiver)

        if input_image is None:
            print("Received None image from NDI.")
            continue
        
        output_image_pil = transform_image_with_streamdiffusion(input_image)

        if output_image_pil == None:
            continue

        output_image_pil_rgba = output_image_pil.convert("RGBA")
        
        if output_image_pil_rgba is not None:
            send_image_to_ndi(sender, output_image_pil_rgba)
        else:
            print("Failed to process image, skipping...")



if __name__ == "__main__":
    from threading import Thread
    
    # OSC
    osc_thread = Thread(target=start_osc_server)
    osc_thread.start()

    # NDI
    ndi_receiver_instance = initialize_ndi_receiver()
    ndi_sender_instance = initialize_ndi_sender()

    # diffusion
    process_images(ndi_receiver_instance, ndi_sender_instance)