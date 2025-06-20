import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import argparse
import datetime
from openai import OpenAI
from datasets import load_dataset, Image as HfImage, DatasetDict
import pandas as pd
from PIL import Image
import numpy as np


# 初始化 DeepSeek 客户端
client = OpenAI(api_key="",
                base_url="https://api.deepseek.com")


class DeepSeekRecapAgent:
    def __init__(self):
        self.prompt_template = """You are an expert poster prompt designer. Your task is to rewrite a user's short poster prompt into a detailed and vivid long-format prompt. Follow these steps carefully:

    **Step 1: Analyze the Core Requirements**
    Identify the key elements in the user's prompt. Do not miss any details.
    - **Subject:** What is the main subject? (e.g., a person, an object, a scene)
    - **Style:** What is the visual style? (e.g., photorealistic, cartoon, vintage, minimalist)
    - **Text:** Is there any text, like a title or slogan?
    - **Color Palette:** Are there specific colors mentioned?
    - **Composition:** Are there any layout instructions?

    **Step 2: Expand and Add Detail**
    Elaborate on each core requirement to create a rich description.
    - **Do Not Omit:** You must include every piece of information from the original prompt.
    - **Enrich with Specifics:** Add professional and descriptive details.
        - **Example:** If the user says "a woman with a bow", you could describe her as "a young woman with a determined expression, holding a finely crafted wooden longbow, with an arrow nocked and ready to fire."
    - **Fill in the Gaps:** If the original prompt is simple (e.g., "a poster for a coffee shop"), use your creativity to add fitting details. You might add "The poster features a top-down view of a steaming latte with delicate art on its foam, placed on a rustic wooden table next to a few scattered coffee beans."

    **Step 3: Handle Text Precisely**
    - **Identify All Text Elements:** Carefully look for any text mentioned in the prompt. This includes:
        - **Explicit Text:** Subtitles, slogans, or any text in quotes.
        - **Implicit Titles:** The name of an event, movie, or product is often the main title. For example, if the prompt is "generate a 'Inception' poster ...", the title is "Inception".
    - **Rules for Text:**
        - **If Text Exists:**
            - You must use the exact text identified from the prompt.
            - Do NOT add new text or delete existing text.
            - Describe each text's appearance (font, style, color, position). Example: `The title 'Inception' is written in a bold, sans-serif font, integrated into the cityscape.`
        - **If No Text Exists:**
            - Add some simple brief texts on it.
    - Every posters must have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

    **Step 4: Final Output Rules**
    - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
    - ***The Output must starts with description about title and text in the picture, and the character description behind them.*
    - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
    - **Keep it concise.** The final prompt should be under 300 words.
    - **Most Important** Only One line as brief_description, don't use multiple rows to be too complex.

    ---
    **User Prompt:**
    {brief_description}"""

    def recap_prompt(self, original_prompt):
        full_prompt = self.prompt_template.format(brief_description=original_prompt)
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": full_prompt},
                ],
                stream=False
            )
            final_answer = response.choices[0].message.content.strip()
            if final_answer:
                return final_answer
            print("DeepSeek returned an empty answer. Using original prompt.")
            return original_prompt
        except Exception as e:
            print(f"DeepSeek API call failed: {e}. Using original prompt.")
            return original_prompt


class PosterGenerator:
    def __init__(self,
                 enable_recap=True,
                 device="cuda:0"):

        self.device = torch.device(device) if isinstance(device, str) else device
        # 使用 DeepSeek 替代 Qwen 模型
        self.qwen_agent = DeepSeekRecapAgent() if enable_recap else None

    def generate_dummy_image(self, width=832, height=1216):
        """Generate a dummy image with gradient for testing purposes"""
        # Create a gradient image
        array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Vertical gradient from black to blue
        for y in range(height):
            blue_value = int(255 * (y / height))
            array[y, :, 2] = blue_value  # Blue channel
            
        # Horizontal gradient from black to red
        for x in range(width):
            red_value = int(255 * (x / width))
            array[:, x, 0] = red_value  # Red channel
            
        return Image.fromarray(array)

    def generate(self,
                 prompt,
                 enable_recap=True,
                 width=832,
                 height=1216,
                 seed=None):

        # Prompt rewriting
        if enable_recap and self.qwen_agent:
            final_prompt = self.qwen_agent.recap_prompt(prompt)
        else:
            final_prompt = prompt

        # Generate dummy image instead of using Flux
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        print("final_prompt:")
        print(final_prompt)

        image = self.generate_dummy_image(width, height)

        return image, final_prompt, seed


def process_dataset(dataset_name, output_dir="generated_posters_tsf"):
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    dataset = load_dataset(dataset_name)['train']

    # 定义 gen_prompt 函数
    def gen_prompt(exp):
        return "Poster about " + exp["product_category"] + " with a young Asian male singer with fair skin and black, slightly messy hair."

    # 添加 poster_prompt 列
    def add_prompt_column(example):
        example['poster_prompt'] = gen_prompt(example)
        return example

    dataset = dataset.map(add_prompt_column)

    # 初始化 PosterGenerator
    generator = PosterGenerator(enable_recap=True, device="cuda:0")

    # 生成图片并添加到数据集中
    def generate_image_and_update(example):
        prompt = example['poster_prompt']
        image, final_prompt, seed = generator.generate(prompt=prompt)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{example['product_category']}_poster_{timestamp}_{seed}.png")
        image.save(output_path)
        example["final_prompt"] = final_prompt
        #example['poster_image'] = output_path
        return example

    dataset = dataset.map(generate_image_and_update, num_proc=1)

    # 将 poster_image 转换为 HfImage 类型
    #dataset = dataset.cast_column("poster_image", HfImage())

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate posters from dataset')
    parser.add_argument('--dataset', type=str, default="svjack/Product_Posters_DESC", help='HuggingFace dataset name')
    args = parser.parse_args()

    processed_dataset = process_dataset(args.dataset)
    processed_dataset.save_to_disk("Product_Posters_Singer_DESC")
    print("Processed dataset saved to disk.")
