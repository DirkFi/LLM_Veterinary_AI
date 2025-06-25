import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    """加载图片，支持URL和本地文件"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        try:
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None
    else:
        try:
            if not os.path.exists(image_file):
                print(f"Image file not found: {image_file}")
                return None
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            print(f"Error loading image file: {e}")
            return None
    return image


def get_user_input_with_image():
    """获取用户输入，包括可选的图片"""
    print("\n--- 新的对话轮次 ---")
    
    # 询问是否要上传图片
    while True:
        img_choice = input("是否要上传图片？(y/n): ").strip().lower()
        if img_choice in ['y', 'yes', 'n', 'no']:
            break
        print("请输入 y 或 n")
    
    image = None
    if img_choice in ['y', 'yes']:
        while True:
            image_path = input("请输入图片路径或URL (按Enter跳过): ").strip()
            if not image_path:
                print("跳过图片上传")
                break
            
            image = load_image(image_path)
            if image is not None:
                print(f"图片加载成功: {image.size}")
                break
            else:
                retry = input("图片加载失败，是否重试？(y/n): ").strip().lower()
                if retry not in ['y', 'yes']:
                    break
    
    # 获取文本输入
    text_input = input("请输入你的问题: ").strip()
    
    return text_input, image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(f"加载模型: {model_name}")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, device=args.device
    )

    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] 自动推断的对话模式是 {}, 但指定的是 {}, 使用 {}'.format(
            conv_mode, args.conv_mode, args.conv_mode))
        conv_mode = args.conv_mode
    else:
        args.conv_mode = conv_mode

    print(f"使用对话模式: {conv_mode}")
    print("=" * 50)
    print("LLaVA 交互式对话启动!")
    print("输入 'quit', 'exit' 或 Ctrl+C 退出")
    print("每轮对话可选择是否上传图片")
    print("=" * 50)

    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = ('user', 'assistant')  # 简化角色定义

    while True:
        try:
            # 获取用户输入和可选图片
            text_input, image = get_user_input_with_image()
            
            if not text_input or text_input.lower() in ['quit', 'exit']:
                print("退出对话...")
                break
                
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话...")
            break

        print(f"\n{roles[1]}: ", end="", flush=True)

        # 每轮对话重新初始化对话模板，避免图片索引冲突
        conv = conv_templates[args.conv_mode].copy()
        
        # 处理输入文本，如果有图片则添加图片token
        inp = text_input
        image_tensor = None
        image_size = None
        
        if image is not None:
            image_size = image.size
            # 处理图片
            image_tensor = process_images([image], image_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            # 添加图片token到输入
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        
        # 添加到对话历史
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 生成回复
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor if image_tensor is not None else None,
                image_sizes=[image_size] if image_size is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True
            )

        # 解码输出
        outputs = tokenizer.decode(output_ids[0]).strip()
        
        # 清理输出，移除输入部分
        if prompt in outputs:
            outputs = outputs.replace(prompt, "").strip()

        if args.debug:
            print(f"\n[DEBUG] Prompt: {prompt}")
            print(f"[DEBUG] Outputs: {outputs}")

        print()  # 换行，准备下一轮对话


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA 交互式CLI")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="模型路径")
    parser.add_argument("--model-base", type=str, default=None,
                       help="基础模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--conv-mode", type=str, default=None,
                       help="对话模式")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="生成温度")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="最大新token数")
    parser.add_argument("--load-8bit", action="store_true",
                       help="使用8bit量化")
    parser.add_argument("--load-4bit", action="store_true", 
                       help="使用4bit量化")
    parser.add_argument("--debug", action="store_true",
                       help="调试模式")
    
    args = parser.parse_args()
    main(args)
