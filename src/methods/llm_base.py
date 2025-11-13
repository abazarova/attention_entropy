"""Module with class for extraction of internal states of LLM."""

import os
from dataclasses import dataclass
from typing import Any, Generator, Literal

import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import STOP_SEQUENCES, StoppingCriteriaSub, postprocess_generation

load_dotenv()


def move_causal_lm_output_to_cpu(
    output: CausalLMOutputWithPast,
) -> CausalLMOutputWithPast:
    if output.past_key_values is not None:
        output.past_key_values = tuple(
            (past_state.cpu(), past_key.cpu())
            for past_state, past_key in output.past_key_values
        )

    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        output.hidden_states = tuple(hs.cpu() for hs in output.hidden_states)

    if hasattr(output, "attentions") and output.attentions is not None:
        output.attentions = tuple(attn.cpu() for attn in output.attentions)

    return output


@dataclass
class LLMBase:
    """Class for extracting internal states of the LLM."""

    model_name: Literal["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.1"]
    dtype: str = "float16"
    device: str = "cuda"
    llm: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None

    num_return_sequences: int = 20
    temperature: float = 1.0
    max_new_tokens: int = 512

    def get_model_id(self):
        login(os.getenv("HUGGING_FACE_API_KEY"))

        if self.model_name == "Llama-2-7b-chat-hf":
            model_id = "meta-llama/Llama-2-7b-chat-hf"
        elif self.model_name == "Mistral-7B-Instruct-v0.1":
            model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        elif self.model_name == "Llama-3.1-8B-Instruct":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
        elif self.model_name == "Llama-2-13b-chat-hf":
            model_id = "meta-llama/Llama-2-13b-chat-hf"
        elif self.model_name == "Qwen2.5-7B-Instruct":
            model_id = "Qwen/Qwen2.5-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model_id

    def instantiate_llm(self):
        """Load LLM and tokenizer."""
        logger.info(f"Instantiating {self.model_name}.")
        # Load API key based on the model
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)

        model_id = self.get_model_id()

        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        logger.info(f"{self.model_name} has been successfully instantiated.")

        return llm, tokenizer

    @torch.no_grad()
    def generate_llm_outputs(
        self,
        X: pd.DataFrame,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        move_to_cpu: bool = True,
    ) -> Generator[Any, torch.Tensor, torch.Tensor]:
        """Yield output, prompt_ids tensor and answer_ids tensor of the LLM on passed data."""
        if (self.llm is None) or (self.tokenizer is None):
            self.llm, self.tokenizer = self.instantiate_llm()

        for i in range(len(X)):
            prompt = X["prompt"].iloc[i]
            answer = X["response"].iloc[i]

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            answer_ids = self.tokenizer(
                answer,
                add_special_tokens=False,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = torch.cat(
                [prompt_ids["input_ids"], answer_ids["input_ids"]], axis=1
            ).to(self.device)

            # Yield the output of the model for the current example
            output = self.llm(
                input_ids,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            if move_to_cpu:
                output = move_causal_lm_output_to_cpu(output)
            yield output, prompt_ids, answer_ids

    @torch.no_grad()
    def generate_llm_responses(self, X: pd.DataFrame) -> Generator[Any, None, None]:
        if (self.llm is None) or (self.tokenizer is None):
            self.llm, self.tokenizer = self.instantiate_llm()

        dataset_name = X["name"].iloc[0]

        # add special token
        def spec_token(model_name):
            if model_name in [
                "Mistral-7B-Instruct-v0.1",
                "Llama-2-7b-chat-hf",
                "Llama-2-13b-chat-hf",
            ]:
                return "</s>"
            elif model_name in ["Llama-3.1-8B-Instruct"]:
                return "<|eot_id|>"
            else:
                return "<|endoftext|>"

        eos_token = spec_token(self.model_name)

        for i in trange(len(X)):
            prompt = X["prompt"].iloc[i]

            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)

            stopping_criteria = None
            if X["name"].iloc[0] == "coqa":
                stopping_criteria = StoppingCriteriaList(
                    [
                        StoppingCriteriaSub(
                            stops=STOP_SEQUENCES,
                            initial_length=len(prompt_ids[0]),
                            tokenizer=self.tokenizer,
                        )
                    ]
                )

            generated_answers = []
            for _ in range(self.num_return_sequences):
                generated_ids = self.llm.generate(
                    **prompt_ids,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )[0][len(prompt_ids[0]) :].cpu()

                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                if dataset_name == "coqa":
                    generated_text = postprocess_generation(generated_text)[0]

                generated_answers.append(f"{generated_text}{eos_token}")

            yield generated_answers
