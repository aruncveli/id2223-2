# id2223-2
Submission for ID2223 HT24 Lab 2

## Task 2
### Initial Approach
 To improve our model's results, we initially considered fine-tuning with a new dataset, specifically the FEVER dataset. However, we realized that incorporating a conversation-based system prompt was essential for our task. Consequently, we shifted our focus to optimizing the model's hyperparameters.

### Model Selection
 We observed that our GPU's maximum RAM usage was around 4 GB, indicating capacity for a larger model. Based on this, we upgraded to Llama 3.2 with 3B parameters (from 1B) while maintaining 4Q quantization.

### Hyperparameter Tuning 
Guided by the LORA paper, we made two key adjustments to the model architecture:
	- Lower r Value: We set r to 8, as the paper indicates lower values perform comparably to higher ones, reducing training time.
	- Target Modules: We trained only the Wq and Wk modules, which the research suggests is sufficient to achieve similar results

Additionally, we applied a dropout rate of 0.1 to mitigate overfitting and increased per_device_train_batch_size to 64, utilizing 11 GB of GPU RAM. Training was conducted over one epoch due to time constraints.

### Results
 After training, we tested the model with the input: "Berlin is a capital of Sweden". The original model responded: "Berlin is not a capital of Sweden. Berlin is actually the capital of Germany." Our model returned the more concise: "Berlin is the capital of Germany."

This shows that our model might be better at understanding quite complicated system prompts and that it performs better on our task. We are aware that one example is not very representative, but given our time and resources, we are happy with the results.
