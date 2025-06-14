import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
import random
import argparse
import datetime
from openai import OpenAI

# 初始化 DeepSeek 客户端
client = OpenAI(api_key="sk-fe823f250986489f916901aa0efd11c8",
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
            - Do not add any text elements. The poster must be purely visual.
    - Most posters have titles. When a title exists, you must extend the title's description. Only when you are absolutely sure that there is no text to render, you can allow the extended prompt not to render text.

    **Step 4: Final Output Rules**
    - **Output ONLY the rewritten prompt.** No introductions, no explanations, no "Here is the prompt:".
    - **Use a descriptive and confident tone.** Write as if you are describing a finished, beautiful poster.
    - **Keep it concise.** The final prompt should be under 300 words.

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
                 pipeline_path="black-forest-labs/FLUX.1-dev",
                 custom_transformer_path=None,
                 enable_recap = True,
                 device="cuda:0"):

        self.device = torch.device(device) if isinstance(device, str) else device

        # 使用 DeepSeek 替代 Qwen 模型
        self.qwen_agent = DeepSeekRecapAgent() if enable_recap else None

        # Load Flux pipeline
        print(f"Loading Flux pipeline from: {pipeline_path}")
        self.pipeline = FluxPipeline.from_pretrained(pipeline_path, torch_dtype=torch.bfloat16)

        # Load custom transformer if provided
        if custom_transformer_path:
            print(f"Loading custom transformer from: {custom_transformer_path}")
            self.pipeline.transformer = FluxTransformer2DModel.from_pretrained(
                custom_transformer_path,
                torch_dtype=torch.bfloat16
            )

        self.pipeline.load_lora_weights("../Flux_Varesa_Lora_Early/my_first_flux_lora_v1_000000750.safetensors")
        self.pipeline.fuse_lora(lora_scale=1.5)

        if self.device.type != 'cpu':
            print("Enabling model CPU offload for FLUX pipeline.")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    def generate(self,
                 prompt,
                 enable_recap=True,
                 width=832,
                 height=1216,
                 num_inference_steps=28,
                 guidance_scale=3.5,
                 seed=None):

        # Prompt rewriting
        if enable_recap and self.qwen_agent:
            final_prompt = self.qwen_agent.recap_prompt(prompt)
        else:
            final_prompt = prompt

        # Poster generation
        if seed is None:
            seed = random.randint(1, 2**32 - 1)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("final_prompt :")
        print(final_prompt)

        with torch.inference_mode():
            image = self.pipeline(
                prompt=final_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]

        return image, final_prompt, seed


def main():
    parser = argparse.ArgumentParser(description='Generate poster')
    parser.add_argument('--prompt', type=str, required=True, help='Input poster description prompt')
    parser.add_argument('--enable_recap', action='store_true', default=True, help='Enable prompt rewriting (default: True)')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--pipeline_path', type=str, default="black-forest-labs/FLUX.1-dev", help='Flux pipeline path')
    parser.add_argument('--custom_transformer_path', type=str, default="PosterCraft/PosterCraft-v1_RL", help='Custom transformer path')
    args = parser.parse_args()

    generator = PosterGenerator(
        pipeline_path=args.pipeline_path,
        custom_transformer_path=args.custom_transformer_path,
        enable_recap = args.enable_recap,
        device="cuda:0"
    )

    image, final_prompt, seed = generator.generate(
        prompt=args.prompt,
        enable_recap=args.enable_recap,
        width=832,
        height=1216,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_poster_{timestamp}_{seed}.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    print(f"Final prompt used: {final_prompt}")


if __name__ == "__main__":
    main()
