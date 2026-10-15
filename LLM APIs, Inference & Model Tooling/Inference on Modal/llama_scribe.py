import modal
from modal import Image, Volume

app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate", "bitsandbytes", "peft")
secrets=[modal.Secret.from_name("huggingface-secret")]
gpu = "T4"

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_MODEL = "damnigotpremium/LLAMA-Scribe"

CACHE_DIR = "/cache"
MIN_CONTAINERS = 0
HF_CACHE_VOLUME = Volume.from_name("hf-hub-cache", create_if_missing = True)


@app.cls(
    image = image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets = secrets,
    gpu = gpu,
    timeout = 1000,
    min_containers = MIN_CONTAINERS,
    volumes = {CACHE_DIR: HF_CACHE_VOLUME}
)
class LlamaScribe:

    @modal.enter()
    def setup(self):

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.bfloat16,
            bnb_4bit_quant_type = "nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map = "auto",
            quantization_config = self.quantization_config,
            dtype = torch.bfloat16
        )

        self.base_model.config.pad_token = self.tokenizer.pad_token_id
        self.base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model,
            ADAPTER_MODEL
        )

        self.finetuned_model.eval()
        print("Model loaded successfully!")

    @modal.method()
    def llama_scribe(self, prompt: str):
        
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt"
        ).to(self.finetuned_model.device)

        with torch.no_grad():
            outputs = self.finetuned_model.generate(
                **inputs,
                max_new_tokens = 200,
                pad_token_id = self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens = True
        )

        return response