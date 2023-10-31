from pytriton.client import ModelClient
import numpy as np
from typing import List

def query_llm(url, model_name, prompts: List[str], max_output_token: int=128, top_k: int=0, top_p: float=0.9, temperature: float=1.0,  init_timeout: float=600.0):
    # source /opt/NeMo/scripts/deploy/query.py
    str_ndarray = np.array(prompts)[..., np.newaxis]
    prompts = np.char.encode(str_ndarray, "utf-8")
    max_output_token = np.full(prompts.shape, max_output_token, dtype=np.int_)
    top_k = np.full(prompts.shape, top_k, dtype=np.int_)
    top_p = np.full(prompts.shape, top_p, dtype=np.single)
    temperature = np.full(prompts.shape, temperature, dtype=np.single)

    with ModelClient(url, model_name, init_timeout_s=init_timeout) as client:
        result_dict = client.infer_batch(
            prompts=prompts,
            max_output_token=max_output_token,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        output_type = client.model_config.outputs[0].dtype

    if output_type == np.bytes_:
        sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
        return sentences
    else:
        return result_dict["outputs"]

if __name__ == "__main__":
    template = "<extra_id_0>System\n\n<extra_id_1>User\n{prompt}\n<extra_id_1>Assistant\n"
    prompts=["What is true for a type-Ia supernova? #\nA. This type occurs in binary systems.\nB. This type occurs in young galaxies.\nC. This type produces gamma-ray bursts.\nD. This type produces high amounts of X-rays.\nAnswer:",
             "How was your summer?"]

    for i in range(len(prompts)):
        prompts[i] = template.format(prompt=prompts[i])

    output = query_llm(url="localhost:8000",
            model_name="GPT-8B-SFT",
            prompts=prompts)

    for i in range(len(prompts)):
        print("prompt: ", prompts[i])
        print("output: ", output[i])
        print("=========================================")
