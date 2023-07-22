# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import torch
import argparse
import openminers
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
import bittensor
import deepspeed
import os
from transformers import GenerationConfig
# from openbase.config import config

deployment_framework = "ds_inference"
deployment_framework = "accelerate"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n \n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

class LlamaMiner( openminers.BasePromptingMiner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--deployment_framework',  type=str, choices=['accelerate', 'deepspeed'], default="accelerate", help='Inference framework to use for multi-gpu inference')
        parser.add_argument( '--use_8_bit', action='store_true', default=False, help='Whether to use int8 quantization or not.' )
        parser.add_argument( '--use_4_bit',  action='store_true', default=False, help='Whether to use int4 quantization or not' )
        parser.add_argument('--llama.model_name',  type=str, default="huggyllama/llama-65b", help='Name/path of model to load')
        parser.add_argument('--llama.max_tokens', type=int, default=20, help="The maximum number of tokens to generate in the completion.")
        parser.add_argument('--llama.do_sample', type=bool, default=True, help='Description of do_sample')
        parser.add_argument('--llama.temperature', type=float, default=1.0, help='Description of temperature')
        parser.add_argument('--llama.top_p', type=float, default=0.95, help='Description of top_p')
        parser.add_argument('--llama.top_k', type=int, default=10, help='Description of top_k')
        parser.add_argument('--llama.stopping_criteria', type=str, default='stop', help='Description of stopping_criteria')

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser( description='llama Config' )
        cls.add_args( parser )
        return bittensor.config( parser )

    def __init__( self, *args, **kwargs):
        super( LlamaMiner, self ).__init__( *args, **kwargs )
        bittensor.logging.info( 'Loading ' + str( self.config.llama.model_name ) )
        # loading the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llama.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llama.model_name, 
            device_map="auto", 
            load_in_4bit=True,
        ).eval()
        self.get_config= GenerationConfig.from_pretrained(self.config.llama.model_name)
        self.get_config.max_new_tokens =500
        self.get_config.length_penalty =1.5
        self.get_config.max_time = 8
        self.get_config.early_stopping = True

    @staticmethod
    def _process_history( history: List[ Dict[str, str] ] ) -> str:
        if history[0]["role"] != "system":
            history = [{"role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                        }] + history
                
        history = [
            {
                "role": history[1]["role"],
                "content": B_SYS
                + history[0]["content"]
                + E_SYS
                + E_SYS
                + B_INST
                + history[1]["content"]
                + E_INST
            }
        ] + history[2:]
        return history[0]['content']

    def forward( self, messages: List[Dict[str, str]]  ) -> str: 
        with torch.no_grad():
            history = self._process_history(messages)
            bittensor.logging.debug( "Message: " + str( history ) )
            inputs = self.tokenizer(history, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs,generation_config=self.get_config)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")
            # Logging input and generation if debugging is active
            bittensor.logging.debug( "Generation: " + text )
        return text

if __name__ == "__main__":  
    miner = LlamaMiner()
    with miner:
        while True:
            print ('running...', time.time() )
            time.sleep( 12)