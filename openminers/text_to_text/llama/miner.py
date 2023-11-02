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
from threading import Lock
# from openbase.config import config

from prompting.baseminer.miner import Miner
from prompting.protocol import Prompting


deployment_framework = "ds_inference"
deployment_framework = "accelerate"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being short and quick to the point."
question_prompt = 'Ask a single relevant and insightful question about the preceding context and previous questions. Do not try to return an answer or a summary:'
answer_prompt = 'Answer the question step by step and explain your thoughts'


class LlamaMiner( Miner ):

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument('--deployment_framework',  type=str, choices=['accelerate', 'deepspeed'], default="accelerate", help='Inference framework to use for multi-gpu inference')
        parser.add_argument( '--use_8_bit', action='store_true', default=False, help='Whether to use int8 quantization or not.' )
        parser.add_argument( '--use_4_bit',  action='store_true', default=False, help='Whether to use int4 quantization or not' )
        parser.add_argument('--llama.model_name',  type=str, default="huggyllama/llama-65b", help='Name/path of model to load')
        parser.add_argument('--llama.model_name_2',  type=str, default="huggyllama/llama-65b", help='Name/path of model to load')

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
            device_map="cpu", 
            torch_dtype=torch.float16,
        )
        self.model = deepspeed.init_inference(self.model,
                                    mp_size=1,
                                    dtype=torch.half,
                                    max_out_tokens = 1024,
                                    replace_with_kernel_inject=True)

        self.model_quantized = AutoModelForCausalLM.from_pretrained(
                    self.config.llama.model_name_2, 
                    device_map="auto", 
                    load_in_4bit=True,
                ).to_bettertransformer()
        self.mutex = Lock()
        self.get_config= GenerationConfig.from_pretrained(self.config.llama.model_name)
        self.get_config.max_new_tokens =200
        self.get_config.temperature = 1.5
        self.get_config.max_time = 8.5

    def _process_history(self, role: List[str], message: List[str] ) -> str:

        history = [
            {
                "role": 'user',
                "content":
                + B_SYS
                + history[0]["content"]
                + E_SYS
                + message[0]
                
            }
        ] 
        return self.tokenizer.apply_chat_template(history, return_tensors="pt")

    def reprocess_message(self, message,name):
        if name =='summarize':
            return message
        elif name == 'answer':
            return message.split('Previous Question')[0] + message.split('Question:')[-1] 
        else:
            return message.split('Previous Question')[0] + question_prompt + E_INST

    def prompt( self, synapse: Prompting  ) -> str: 
        role, messages = synapse.roles, synapse.messages
        with torch.no_grad():
            start = time.time()
            history = self._process_history(role, messages)
            

            if 'Summarize the preceding context' in history :
                #history = self.reprocess_message(history, 'summarize')
                bittensor.logging.debug( "Message: " + str( history ) )
                inputs = self.tokenizer(history, return_tensors="pt").to("cuda")
                with self.mutex:
                    outputs = self.model.generate(**inputs,generation_config=self.get_config)
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")

            elif 'Answer the question ' in history :
                #history = self.reprocess_message(history, 'answer')
                bittensor.logging.debug( "Message: " + str( history ) )
                inputs = self.tokenizer(history, return_tensors="pt").to("cuda")
                with self.mutex:
                    outputs = self.model.generate(**inputs,generation_config=self.get_config)
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")

            else:
                #history = self.reprocess_message(history, 'question')
                bittensor.logging.debug( "Message: " + str( history ) )
                inputs = self.tokenizer(history, return_tensors="pt").to("cuda")
                outputs = self.model_quantized.generate(**inputs,generation_config=self.get_config)
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace( str( history ), "")


            # Logging input and generation if debugging is active
            bittensor.logging.debug( "Generation: " + str(time.time()-start) + text )
            synapse.completion = text
        return synapse

if __name__ == "__main__":  
    miner = LlamaMiner()
    with miner:
        while True:
            print ('running...', time.time() )
            time.sleep( 60)
            torch.cuda.empty_cache()