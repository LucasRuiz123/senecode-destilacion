"""
Destilación (SFT con LoRA) de un modelo pequeño para responder ejercicios de Senecode.

Entrada:  data/train.jsonl y data/eval.jsonl, una línea por ejemplo:
  {"messages": [
     {"role": "system", "content": "Eres un tutor de programación..."},
     {"role": "user", "content": "Enunciado: ...\nCódigo: ...\nTests: 3/5 pasan (falla n=0)\nAST: usa 'for' en línea 4"},
     {"role": "assistant", "content": "{\"correctness\": false, ... , \"feedback\": \"...\"}"}
  ]}
  (la respuesta del assistant es la del modelo grande, ya filtrada con los tests)

Salida:
  checkpoints/          -> se guardan cada SAVE_STEPS (sobrevive al reinicio nocturno)
  salida/lora/          -> solo el adaptador (decenas a cientos de MB)
  salida/merged_16bit/  -> modelo fusionado para vLLM / transformers
  salida/gguf/          -> GGUF Q4_K_M para llamafile / Ollama / llama.cpp

Requisitos (en WSL2, PC con RTX 4090):
  sudo apt install build-essential
  python -m venv venv && source venv/bin/activate
  pip install unsloth
"""

import argparse
import glob
import os

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

# ---------------- Configuración ----------------
# Prueba de humo:  python destilar_senecode.py --modelo Qwen/Qwen3.5-2B --epocas 3 --prueba
# Corrida real:    python destilar_senecode.py
parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="Qwen/Qwen3.5-4B")
parser.add_argument("--epocas", type=float, default=2)
parser.add_argument("--prueba", action="store_true",
                    help="dataset diminuto: registra la pérdida en cada paso")
cli = parser.parse_args()

MODELO_BASE = cli.modelo
MAX_SEQ_LEN = 2048                 # súbelo si enunciado + código + respuesta no caben
LORA_R = 16
EPOCHS = cli.epocas
SAVE_STEPS = 5 if cli.prueba else 100   # ~cada 15-30 min en corridas reales
LOG_STEPS = 1 if cli.prueba else 10
CHECKPOINTS = "checkpoints"
SALIDA = "salida"

# ---------------- Modelo + LoRA ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODELO_BASE,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,   # bf16 LoRA: el 4B usa ~10 GB de VRAM
    load_in_16bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_R,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---------------- Datos ----------------
def a_texto(ejemplo):
    # Usa EXACTAMENTE la plantilla de chat del modelo: la misma debe usarse al servirlo.
    texto = tokenizer.apply_chat_template(
        ejemplo["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,   # respuestas directas, sin bloque de razonamiento
    )
    return {"text": texto}

ds = load_dataset("json", data_files={"train": "data/train.jsonl",
                                      "eval": "data/eval.jsonl"})
ds = ds.map(a_texto, remove_columns=ds["train"].column_names)

print("---- Ejemplo formateado (revisa los marcadores de rol) ----")
print(ds["train"][0]["text"][:1500])

# ---------------- Entrenamiento ----------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    args=SFTConfig(
        dataset_text_field="text",
        output_dir=CHECKPOINTS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=LOG_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        seed=3407,
        report_to="none",
    ),
)

# Solo aprende de la respuesta del assistant, no del enunciado ni del código.
# Marcadores de la plantilla ChatML de Qwen; verifícalos con el ejemplo impreso arriba.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Reanuda solo si ya hay un checkpoint (p. ej. tras el reinicio nocturno).
hay_checkpoint = bool(glob.glob(os.path.join(CHECKPOINTS, "checkpoint-*")))
trainer.train(resume_from_checkpoint=hay_checkpoint)

# ---------------- Exportar ----------------
model.save_pretrained(f"{SALIDA}/lora")
tokenizer.save_pretrained(f"{SALIDA}/lora")

model.save_pretrained_merged(f"{SALIDA}/merged_16bit", tokenizer,
                             save_method="merged_16bit")

model.save_pretrained_gguf(f"{SALIDA}/gguf", tokenizer,
                           quantization_method="q4_k_m")

print("Listo. Adaptador, modelo fusionado y GGUF en", SALIDA)
