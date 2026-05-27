import argparse
import json
import time
import os
import psutil
import onnxruntime_genai as og

from ryzenai_lora_inferface import (
    set_active_adapter,
    set_active_adapter_from_buffer
)

import onnx
import onnxruntime as ort
print(f"onnxruntime version: {ort.__version__} ort version:{onnx.__version__}")

def get_peak_memory_usage(label):
    process = psutil.Process(os.getpid())
    cur_mem = process.memory_info().rss
    peak_memory = process.memory_info().peak_wset
    print(label + " cur mem:", cur_mem/1024/1024/1024, "GB")
    print(label + " peak mem:", peak_memory/1024/1024/1024, "GB")

def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for Qwen3-VL-4B"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="cpu_and_mobile/models",
        help="Path to the model directory containing genai_config.json and ONNX models"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Text prompt file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default="base",
        help="Lora name"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="max context cache length"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()
    get_peak_memory_usage("after load model")

    if args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream, args)
    elif args.prompt:
        generate_response(model, processor, tokenizer, tokenizer_stream, args.prompt, args.image, args)
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            prompt = file.read()
        generate_response(model, processor, tokenizer, tokenizer_stream, prompt, args.image, args)
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


def generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path, args):
    # Build messages for chat template
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        # Message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        # Text-only message
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

    # Apply chat template (requires JSON string)
    full_prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)

    print(f"\nPrompt: {prompt}")
    if image_path:
        print(f"Image: {image_path}")
    print("\nGenerating response...")

    # Process inputs
    print("full_prompt:", len(full_prompt))
    inputs = processor(full_prompt, images=images)
    print("input_ids len:", inputs['input_ids'].shape()[1])

    # Set up generation parameters
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=args.max_length)

    # Generate
    generator = og.Generator(model, params)
    set_active_adapter(args.lora, dll_name="onnxruntime_providers_ryzenai.dll")
    get_peak_memory_usage("after generator")

    first_token = False
    start_time = time.time()
    generator.set_inputs(inputs)
    tokens = 0

    print("\nResponse: ", end="", flush=True)

    while not generator.is_done():
        generator.generate_next_token()
        tokens += 1
        new_token = generator.get_next_tokens()[0]
        if not first_token:
            TTFT = time.time() - start_time
            # print(f"First token generated at: {time.time()}")
            first_token = True
        print(tokenizer_stream.decode(new_token), end="", flush=True)
    print()
    end_time = time.time()
    print("\nTTFT: {:.2f} seconds".format(TTFT))
    print("TPS: {:.2f} tokens/second".format(tokens / (end_time - start_time - TTFT)))
    print("\n")
    get_peak_memory_usage("before del generator")
    del generator
    get_peak_memory_usage("after del generator")


def interactive_mode(model, processor, tokenizer, tokenizer_stream, args):
    """Run in interactive mode."""
    print("\n" + "="*50)
    print("Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("To include an image, type: image:/path/to/image.jpg")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            print("Please enter a prompt.")
            continue

        # Check for image path
        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]  # Remove "image:" prefix
            prompt = parts[1] if len(parts) > 1 else "Describe this image"

        try:
            generate_response(
                model, processor, tokenizer, tokenizer_stream,
                prompt, image_path
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-"*50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
