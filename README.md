# ID2223 Lab 2 - Fact Checker
Submission for ID2223 HT24 Lab 2 from Group 4

## Task 1
### Inference
During the first week of working on this assignment, we had spent considerable amount of time trying run inference sufficiently fast without GPU on free HuggingFace Spaces. We initially used a heavily quantized ([`iq2-xxs`](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L23))  [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) based fine tuned model, exported in [GGUF](https://huggingface.co/docs/hub/en/gguf) format and used [`llama-cpp-python`](https://llama-cpp-python.readthedocs.io/en/latest/) in the Spaces runtime for inference.

But since Jim mentioned last week that we can [demo the inference on Colab itself](https://canvas.kth.se/courses/50172/discussion_topics/432284), we scrapped those plans and restarted with Llama-3B-Instruct. We use [`unsloth`](https://docs.unsloth.ai/basics/inference) for the inference pipeline as well, levergaing the GPU.

### UI
The UI is built with [Gradio](https://www.gradio.app/docs/gradio/interface), presenting the user with a prompt input and output, with additional prefilled but modifiable inputs for the system prompt and some model tweaks.

## Task 2
### Initial idea

To improve our model's results compared to Task 1, we initially considered fine-tuning with a new dataset, specifically the [FEVER](https://huggingface.co/datasets/fever/fever) dataset. However, we realized the dataset is not in a chat template and we may not get enough time to write a proper converter to Llama chat template before the submission deadline.

Incorporating a conversation-based system prompt was essential for our task. Consequently, we shifted our focus to optimizing the model's hyperparameters.

### Model Selection

We observed during Task 1 that the Colab runtime's max GPU usage during training was around 4 GB out of ~12GB, indicating its capacity to train a larger model. Based on this, we upgraded to Llama 3.2 with 3B parameters (from 1B) while maintaining 4Q quantization.

### Hyperparameter Tuning

Guided by the [LORA](https://arxiv.org/abs/2106.09685) paper, we made two key adjustments to the fine tuning architecture:
- Lower value for `r`: We reduced `r`, [the rank of the decomposition matrix](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia#r), from 16 to 8, as the paper indicates lower values perform comparably to higher ones, reducing training time.
- [Target Modules](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia#target-modules): We trained only the *Wq* and *Wk* modules, which the research suggests is sufficient to achieve comparable results.

Additionally:
* we applied a dropout rate of 0.1 to mitigate overfitting
* increased `per_device_train_batch_size` to 64, utilizing 11 GB of GPU RAM.

Training was conducted over one epoch due to time constraints.

### Results
With the system prompt asking the model to be a strict fact checker and respond as consicely as possible, for the input *Berlin is a capital of Sweden*

* Task 1 model responded: *Berlin is not a capital of Sweden. Berlin is actually the capital of Germany.*
* Task 2 model responded: *Berlin is the capital of Germany.*

This shows that our model might be better at understanding quite complicated system prompts and that it performs better on our task. We are aware that one example is not very representative, but given our time and resources, we are happy with the results.
