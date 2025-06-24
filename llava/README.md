# LLaVA Interactive CLI

This directory provides a command-line interface (CLI) for interactive conversations with LLaVA running locally, a vision-enabled large language model. You can chat with the model and optionally upload images each turn.


## Installation

> **Note:** If you are not using Linux, please refer to [macOS](#macos) or [Windows](#windows) instructions below before proceeding.

1. **Clone the repository**

    ```bash
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    ```

2. **Create and activate a Conda environment**

    ```bash
    conda create -n llava python=3.10 -y
    conda activate llava
    pip install --upgrade pip
    ```

3. **Install the package**

    ```bash
    pip install -e .
    pip install protobuf
    ```

Usage

Run the interactive CLI with your model path and options:

```
python example.py \
  --model-path /path/to/your/model \
  [--model-base /path/to/base/model] \
  [--device cuda|cpu] \
  [--conv-mode llava_v1|llama-2|mpt|...] \
  [--temperature 0.0-1.0] \
  [--max-new-tokens 128-1024] \
  [--load-8bit] [--load-4bit] [--debug]
```
After launching, you can:
	•	Type your question and optionally upload an image each turn
	•	Enter quit or exit (or press Ctrl+C) to terminate

Command-Line Arguments

Flag	Type	Default	Description
--model-path	string	REQUIRED	Path to the pretrained LLaVA model
--model-base	string	None	Path to a base model (for adapters or fine-tuning)
--device	string	cuda	Runtime device: cuda for GPU, cpu for CPU inference
--conv-mode	string	auto	Conversation template: choose from supported modes
--temperature	float	0.2	Sampling temperature (0 for greedy)
--max-new-tokens	int	512	Maximum number of tokens to generate per response
--load-8bit	boolean	False	Enable 8-bit quantized model loading
--load-4bit	boolean	False	Enable 4-bit quantized model loading
--debug	boolean	False	Print debug information including prompts and raw outputs

Examples

Launch with GPU and default settings:

python example.py --model-path ./checkpoints/llava_model

Launch on CPU with higher temperature and shorter responses:

python example.py --model-path ./checkpoints/llava_model --device cpu --temperature 0.7 --max-new-tokens 256

