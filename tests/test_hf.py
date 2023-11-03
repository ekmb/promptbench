# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

class TestBatch:
    def test_batch(self):
        model = "google/flan-t5-large"
        tokenizer = T5Tokenizer.from_pretrained(model, device_map="cuda")
        model = T5ForConditionalGeneration.from_pretrained(model, device_map="cuda")

        prompts = ("Analyze the tone of this statement and respond with either 'positive' or 'negative': it 's a charming and often affecting journey .  Answer: ", 
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': unflinchingly bleak and desperate  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': it 's slow -- very , very slow .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': a sometimes tedious film .  Answer: ", "Analyze the tone of this statement and respond with either 'positive' or 'negative': or doing last year 's taxes with your ex-wife .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': you do n't have to know about music to appreciate the film 's easygoing blend of comedy and romance .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': in exactly 89 minutes , most of which passed as slowly as if i 'd been sitting naked on an igloo , formula 51 sank from quirky to jerky to utter turkey .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': the mesmerizing performances of the leads keep the film grounded and keep the audience riveted .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': it takes a strange kind of laziness to waste the talents of robert forster , anne meara , eugene levy , and reginald veljohnson all in the same movie .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': ... the film suffers from a lack of humor ( something needed to balance out the violence ) ...  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': we root for ( clara and paul ) , even like them , though perhaps it 's an emotion closer to pity .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': even horror fans will most likely not find what they 're seeking with trouble every day ; the movie lacks both thrills and humor .  Answer: ",
                   "Analyze the tone of this statement and respond with either 'positive' or 'negative': a gorgeous , high-spirited musical from india that exquisitely blends music , dance , song , and high drama .  Answer: ")
        
        gt = ['positive', 'negative', 'positive', 'positive', 'negative', 'positive', 'negative', 'negative', 'positive', 'negative', 'positive', 'negative', 'negative', 'positive', 'negative', 'positive']
        # get batch predictions
        batch_input_ids = tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt").input_ids.to("cuda")
        batch_output = model.generate(batch_input_ids, max_length=20, early_stopping=True)
        batch_preds = tokenizer.batch_decode(batch_output, skip_special_tokens=True)
        
        # get individual predictions
        individual_preds = []
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            output = model.generate(input_ids, max_length=20, early_stopping=True)
            individual_preds.append(tokenizer.decode(output[0], skip_special_tokens=True))
        
        assert batch_preds == individual_preds, "Batch predictions do not match individual predictions"
        assert batch_preds == gt, "Batch predictions do not match ground truth"


