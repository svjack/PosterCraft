import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import random
import argparse
import datetime


class Qwen3RecapAgent:
    def __init__(self, model_path, device="auto"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model_kwargs = {"torch_dtype": "auto"}
        if device != "auto":
            model_kwargs["device_map"] = None
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if device != "auto":
            self.model.to(device)
        
        self.system_prompt = """You are a Poster Prompt Rewriter. When the user provides a poster prompt—whether short or long—you must transform it into a detailed, coherent, and vividly descriptive long-format prompt matching the style used during model training. Follow these rules:

1. Opening Sentence  
   Begin with: This [adjective] poster for "<Title>" presents …  
   Strict:Only include text explicitly mentioned in the user's prompt. If the user does not provide a title, subtitle, tagline, or any on-poster copy, do not generate any.

2. Scene & Composition  
   In one or two sentences, describe the environment, main subject(s), their poses or actions, and the overall mood or narrative.

3. Artistic Style & Atmosphere  
   In one or two sentences, specify the visual style, color palette, lighting, and emotional tone.

4. Typography & Layout  
   Only if the original prompt specifies on-poster text, detail its placement, font style, size, color, and contrast.
   If no on-poster text is given, omit this section entirely.

5. Preserve Original Content  
   - Do not introduce any text sections, taglines, dates, venues, or copy not explicitly mentioned. 
   - Only describe typography or copy the user has specified.

6. Preserve Typography Aesthetics  
   - Retain any special text effects (metallic sheen, ripple effect, pixel style).  
   - Keep original capitalization, punctuation, and decorative descriptors.

7. Impact Summary  
   Add a closing sentence reinforcing the poster's emotional or visual impact using only original or permitted placeholder content.

8. No Extraneous Content  
   Return _only_ the fully complete rewritten prompt. Do not include analysis, commentary, salutations, or stray characters."""
    
    def recap_prompt(self, original_prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": original_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.6,
                do_sample=True,
                top_p=0.95,
                top_k=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        full_response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return self._extract_final_answer(full_response) or original_prompt

    def _extract_final_answer(self, full_response):
        if "</think>" in full_response:
            think_end_pos = full_response.rfind("</think>")
            if think_end_pos != -1:
                final_answer = full_response[think_end_pos + 8:].strip()
                if final_answer:
                    return final_answer
        
        try:
            tokens = self.tokenizer.encode(full_response, add_special_tokens=False)
            think_end_token_id = 151668
            think_end_indices = [i for i, token_id in enumerate(tokens) if token_id == think_end_token_id]
            if think_end_indices:
                last_think_end_idx = think_end_indices[-1]
                answer_tokens = tokens[last_think_end_idx + 1:]
                if answer_tokens:
                    final_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
                    if final_answer:
                        return final_answer
        except:
            pass
        
        if "<think>" not in full_response:
            return full_response.strip()
        
        return None


class PosterGenerator:
    def __init__(self, 
                 pipeline_path="black-forest-labs/FLUX.1-dev",
                 custom_transformer_path=None,
                 qwen_model_path=None,
                 device="cuda:0"):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Load Qwen model for prompt rewriting
        if qwen_model_path and os.path.exists(qwen_model_path):
            print(f"Loading Qwen model from: {qwen_model_path}")
            self.qwen_agent = Qwen3RecapAgent(qwen_model_path, device=self.device)
        else:
            self.qwen_agent = None
        
        # Load Flux pipeline
        print(f"Loading Flux pipeline from: {pipeline_path}")
        self.pipeline = FluxPipeline.from_pretrained(pipeline_path, torch_dtype=torch.bfloat16)
        
        # Load custom transformer if provided
        if custom_transformer_path:
            print(f"Loading custom transformer from: {custom_transformer_path}")
            self.pipeline.transformer = FluxTransformer2DModel.from_pretrained(
                custom_transformer_path, 
                # subfolder="transformer", 
                torch_dtype=torch.bfloat16
            )
        
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
    parser.add_argument('--qwen_model_path', type=str, default="Qwen/Qwen3-8B", help='Qwen model path')
    args = parser.parse_args()

    generator = PosterGenerator(
        pipeline_path=args.pipeline_path,
        custom_transformer_path=args.custom_transformer_path,
        qwen_model_path=args.qwen_model_path,
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
