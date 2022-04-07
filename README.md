## Downloading a text dataset to perform fine-tuning on

Download the raw character-level data of the `WikiText-2` dataset from 
[this](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
link.

Extract the data and place the folder containing the `.raw` files in the root
directory of this project.

## Fine-tuning RoBERTa

Run `lm_finetuning.py` with the following command:

```
--output_dir=lm_finetuning_roberta_output --model_type=roberta 
--model_name_or_path=roberta-base --do_lower_case --mlm 
--train_data_file=wikitext-2-raw/wiki.train.raw --do_train 
--per_gpu_train_batch_size=2 --eval_data_file=wikitext-2-raw/wiki.test.raw 
--do_eval --per_gpu_eval_batch_size=2 --save_steps=100 --save_total_limit=3
```

A directory with the name `lm_finetuning_roberta_output` will be generated.

If you want to fine-tune a different model, for example BERT, change 
`model_type`, `model_name_or_path` and `output_dir`, giving 
you a command like so:

```
--output_dir=lm_finetuning_bert_output --model_type=bert 
--model_name_or_path=bert-base-uncased --do_lower_case --mlm 
--train_data_file=wikitext-2-raw/wiki.train.raw --do_train 
--per_gpu_train_batch_size=2 --eval_data_file=wikitext-2-raw/wiki.test.raw 
--do_eval --per_gpu_eval_batch_size=2 --save_steps=100 --save_total_limit=3
```

At least one of `do_train` and `do_eval` must be provided. If you only want 
to run training, then just don't pass `do_eval` as an arg, and vice versa. The 
other eval-related args 
will be ignored.

Note that saving all checkpoints will take up a lot of disk space, 
hence the save limit of 3.

## Using the pretrained model

Run `run_roberta.py`. In the `get_all_predictions()` function, you can swap out
the model you fine-tuned with the default RoBERTa model by commenting out the
first two lines of that function body. More specifically, do this:

```
def get_all_predictions(text):
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()
    # roberta_path = 'lm_finetuning_roberta_output'
    # roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    # roberta_model = RobertaForMaskedLM.from_pretrained(roberta_path).eval()
    ...
```
