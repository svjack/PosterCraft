#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import time
import logging
import traceback


# --- Basic Logging Setup ---
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Qwen Recap Agent ---
class QwenRecapAgent:
    def __init__(self, model_path="/hpc2hdd/home/jlai218/Siggraph2025/hf_test_2/Qwen3-8B", max_retries=3, retry_delay=2, device_map="auto"):
        """Initializes the Qwen recap agent."""
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        model_kwargs = {"torch_dtype": "auto"}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if self.device != "auto":
             self.model.to(self.device)
        
#         self.system_prompt = """"You are a Poster Prompt Rewriter. When the user provides a poster prompt—whether short or long—you must transform it into a detailed, coherent, and vividly descriptive long-format prompt matching the style used during model training. Follow these rules:

# 1. Opening Sentence  
#    Begin with:  
#    > This [adjective] poster for "<Title>" presents …  
#    Strict:Only include text explicitly mentioned in the user's prompt. If the user does not provide a title, subtitle, tagline, or any on-poster copy, do not generate any.

# 2. Scene & Composition  
#    In one or two sentences, describe the environment, main subject(s), their poses or actions, and the overall mood or narrative.

# 3. Artistic Style & Atmosphere  
#    In one or two sentences, specify the visual style, color palette, lighting, and emotional tone.

# 4. Typography & Layout  
#    Only if the original prompt specifies on-poster text, detail its placement, font style, size, color, and contrast.
#    If no on-poster text is given, omit this section entirely.

# 5. Preserve Original Content  
#    - Do not introduce any text sections, taglines, dates, venues, or copy not explicitlymentioned. 
#    - Only describe typography or copy the user has specified.

# 6. Preserve Typography Aesthetics  
#    - Retain any special text effects (metallic sheen, ripple effect, pixel style).  
#    - Keep original capitalization, punctuation, and decorative descriptors.

# 7. Impact Summary  
#    Add a closing sentence reinforcing the poster's emotional or visual impact using only original or permitted placeholder content.

# 8. No Extraneous Content  
#    Return _only_ the fully a complete rewritten prompt. Do not include analysis, commentary, salutations, or stray characters.

# Few-Shot Example:

# (short prompt)
# This poster for the 'Vintage Wheels Classic Car Show' presents a beautifully illustrated retro automobile with chrome details, set against a nostalgic backdrop, in a polished, classic Americana style.

# Assistant:
# This nostalgic poster for "Vintage Wheels Classic Car Show" presents a beautifully illustrated retro automobile, its gleaming chrome details catching the light. The car is positioned front and center, framed by a softly blurred backdrop evoking mid-century Americana, with hints of vintage signage and pastel tones. Rendered in a polished, classic style, the composition features a warm color palette dominated by creamy whites, soft reds, and subdued blues, complemented by smooth gradients and subtle shadows that enhance the retro aesthetic. The title "Vintage Wheels Classic Car Show" appears prominently at the top in bold, serif lettering with a polished metallic finish, reminiscent of classic car emblems, perfectly complementing the nostalgic theme. The overall design exudes elegance and celebrates the timeless allure of vintage automobiles
# """
        self.system_prompt = """"You are a Poster Prompt Rewriter. When the user provides a poster prompt—whether short or long—you must transform it into a detailed, coherent, and vividly descriptive long-format prompt matching the style used during model training. Follow these rules:

        1. Opening Sentence  
        Begin with:  
        > This [adjective] poster for "<Title>" presents …  
        Strict:Only include text explicitly mentioned in the user's prompt. If the user does not provide a title, subtitle, tagline, or any on-poster copy, do not generate any.

        2. Scene & Composition  
        In one or two sentences, describe the environment, main subject(s), their poses or actions, and the overall mood or narrative.

        3. Artistic Style & Atmosphere  
        In one or two sentences, specify the visual style, color palette, lighting, and emotional tone.

        4. Typography & On-Poster Text
        If the user's prompt includes on-poster text, add this section. Otherwise, omit it entirely. Carefully analyze the user's prompt to avoid missing any text.
        Systematically describe each piece of specified text (e.g., title, tagline, date). For each element, concisely detail:
            - Content: Its exact wording, preserving original capitalization and punctuation.
            - Placement: Its location on the poster (e.g., "at the top," "centered below the image").
            - Style: Its typographic attributes (e.g., "bold serif font," "glowing neon effect," "subtle metallic sheen").
            - Strict: Do not add or invent any text not present in the original prompt. Describe only what is provided.
        
        5. Concluding Statement
        Conclude with a single, concise sentence that synthesizes the key visual and textual elements to underscore the poster's central theme or purpose. Avoid repeating descriptions from previous sections.

        6. Preserve Typography Aesthetics  
        - Retain any special text effects (metallic sheen, ripple effect, pixel style).  
        - Keep original capitalization, punctuation, and decorative descriptors.

        7. Impact Summary  
        Add a closing sentence reinforcing the poster's emotional or visual impact using only original or permitted placeholder content.

        8. No Extraneous Content  
        Return _only_ the fully a complete rewritten prompt. Do not include analysis, commentary, salutations, or stray characters.

        Few-Shot Example:

        (short prompt)
        This poster for the 'Vintage Wheels Classic Car Show' presents a beautifully illustrated retro automobile with chrome details, set against a nostalgic backdrop, in a polished, classic Americana style.

        Assistant:
        This nostalgic poster for "Vintage Wheels Classic Car Show" presents a beautifully illustrated retro automobile, its gleaming chrome details catching the light. The car is positioned front and center, framed by a softly blurred backdrop evoking mid-century Americana, with hints of vintage signage and pastel tones. Rendered in a polished, classic style, the composition features a warm color palette dominated by creamy whites, soft reds, and subdued blues, complemented by smooth gradients and subtle shadows that enhance the retro aesthetic. The title "Vintage Wheels Classic Car Show" appears prominently at the top in bold, serif lettering with a polished metallic finish, reminiscent of classic car emblems, perfectly complementing the nostalgic theme. The overall design exudes elegance and celebrates the timeless allure of vintage automobiles
        """
    
    def recap_prompt(self, original_prompt, user_id='poster_recap'):
        """Rewrites the prompt using the Qwen model."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": original_prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=2048, temperature=0.1, do_sample=True,
                        top_p=0.95, top_k=20, min_p=0.0, pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                full_response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                final_answer = self._extract_final_answer(full_response)
                
                if final_answer:
                    return final_answer.strip()
                else:
                    logger.warning("Qwen returned an empty final answer.")
            except Exception as e:
                logger.error(f"Qwen recap request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"Qwen recap failed after all retries. Using original prompt.")
        return original_prompt

    def _extract_final_answer(self, full_response):
        try:
            if "</think>" in full_response:
                final_answer = full_response.split("</think>")[-1].strip()
                return '\n'.join(line.strip() for line in final_answer.split('\n') if line.strip())
            if "<think>" not in full_response:
                return full_response.strip()
            return None
        except Exception as e:
            logger.error(f"Error while extracting final answer: {e}")
            return None

# --- Global State and Device Configuration ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
else:
    device = torch.device("cpu")
    logger.warning("CUDA not available. Falling back to CPU.")

# --- Model Configuration ---
# DEFAULT_PIPELINE_PATH = "black-forest-labs/FLUX.1-dev"
DEFAULT_PIPELINE_PATH = "/hpc2hdd/home/jlai218/HF_checkpoint/flux1.0_dev"
# DEFAULT_QWEN_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"
DEFAULT_QWEN_MODEL_PATH = "/hpc2hdd/home/jlai218/Siggraph2025/hf_test_2/Qwen3-8B"
# DEFAULT_CUSTOM_WEIGHTS_PATH = "PosterCraft/PosterCraft-v1_RL" 
DEFAULT_CUSTOM_WEIGHTS_PATH = "/hpc2hdd/home/jlai218/Siggraph2025/hf_test_2/PosterCraft-v1_RL" 

# --- Poster Generator Class ---
class PosterGenerator:
    def __init__(self, pipeline_path, qwen_model_path, custom_weights_path, device):
        self.device = device
        self.qwen_agent = self._load_qwen_agent(qwen_model_path)
        self.pipeline = self._load_flux_pipeline(pipeline_path, custom_weights_path)
        
    def _load_qwen_agent(self, qwen_model_path):
        if not qwen_model_path:
            return None
        return QwenRecapAgent(model_path=qwen_model_path, device_map=str(self.device))

    def _load_flux_pipeline(self, pipeline_path, custom_weights_path):
        pipeline = FluxPipeline.from_pretrained(pipeline_path, torch_dtype=torch.bfloat16)
        
        if custom_weights_path and os.path.exists(custom_weights_path):
            logger.info(f"Loading custom Transformer from directory: {custom_weights_path}")
            transformer = FluxTransformer2DModel.from_pretrained(
                custom_weights_path, torch_dtype=torch.bfloat16
            )
            pipeline.transformer = transformer
        
        pipeline.to(self.device)
        return pipeline

    def generate(self, prompt, enable_recap, **kwargs):
        final_prompt = prompt
        if enable_recap:
            if not self.qwen_agent:
                raise gr.Error("Recap is enabled, but the recap model is not available. Check model path.")
            final_prompt = self.qwen_agent.recap_prompt(prompt)

        generator = torch.Generator(device=self.device).manual_seed(kwargs['seed'])
        
        with torch.inference_mode():
            image = self.pipeline(
                prompt=final_prompt,
                generator=generator,
                num_inference_steps=kwargs['num_inference_steps'],
                guidance_scale=kwargs['guidance_scale'],
                width=kwargs['width'],
                height=kwargs['height']
            ).images[0]
        
        return image, final_prompt

# --- Global Model Initialization ---
generator = PosterGenerator(
    pipeline_path=DEFAULT_PIPELINE_PATH,
    qwen_model_path=DEFAULT_QWEN_MODEL_PATH,
    custom_weights_path=DEFAULT_CUSTOM_WEIGHTS_PATH,
    device=device
)

# --- Gradio Interface Logic ---
def generate_image_interface(
    original_prompt, enable_recap, height, width, 
    num_inference_steps, guidance_scale, seed_input
):
    if not original_prompt or not original_prompt.strip():
        raise gr.Error("Prompt cannot be empty!")

    try:
        actual_seed = int(seed_input) if seed_input and seed_input > 0 else random.randint(1, 2**32 - 1)
        
        image, final_prompt = generator.generate(
            prompt=original_prompt,
            enable_recap=enable_recap,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=actual_seed
        )
        
        status_log = f"Seed: {actual_seed} | Generation complete."
        return image, final_prompt, status_log

    except Exception as e:
        logger.error(traceback.format_exc())
        raise gr.Error(f"An error occurred: {e}")


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# PosterCraft-v1.0")
    gr.Markdown(f"Running on: **{device}** | Base Pipeline: **{DEFAULT_PIPELINE_PATH}**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your creative prompt...")
            enable_recap_checkbox = gr.Checkbox(label="Enable Prompt Recap", value=True, info=f"Uses {DEFAULT_QWEN_MODEL_PATH} for rewriting.")
            
            with gr.Row():
                width_input = gr.Slider(label="Width", minimum=256, maximum=2048, value=832, step=64)
                height_input = gr.Slider(label="Height", minimum=256, maximum=2048, value=1216, step=64)
            gr.Markdown("Tip: Recommended size is 832x1216 for best results.")
            
            num_inference_steps_input = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=28, step=1)
            guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=0.0, maximum=20.0, value=3.5, step=0.1)
            seed_number_input = gr.Number(label="Seed", value=None, minimum=-1, step=1, info="Leave blank or set to -1 for a random seed.")
            generate_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Results")
            image_output = gr.Image(label="Generated Image", type="pil", show_download_button=True, height=512)
            recapped_prompt_output = gr.Textbox(label="Final Prompt Used", lines=5, interactive=False)
            status_output = gr.Textbox(label="Status Log", lines=4, interactive=False)

    inputs_list = [
        prompt_input, enable_recap_checkbox, height_input, width_input,
        num_inference_steps_input, guidance_scale_input, seed_number_input
    ]
    outputs_list = [image_output, recapped_prompt_output, status_output]
    
    generate_button.click(fn=generate_image_interface, inputs=inputs_list, outputs=outputs_list)

if __name__ == "__main__":
    logger.info("Starting Gradio application...")
    demo.queue().launch(server_name="0.0.0.0", server_port=8420, share=False)
