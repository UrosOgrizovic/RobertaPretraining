import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import string


def encode(tokenizer, text):
    text = text.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models don't predict punctuation.
    if tokenizer.mask_token == text.split()[-1]:
        text += ' .'

    # add_special_tokens is True so as to encode out-of-vocabulary tokens with ## subwords
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def decode(tokenizer, pred_idx):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return ', '.join(tokens)


def get_all_predictions(text):
    # roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()
    roberta_path = 'lm_finetuning_roberta_output'
    roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    roberta_model = RobertaForMaskedLM.from_pretrained(roberta_path).eval()
    num_top_words = 5
    input_ids, mask_idx = encode(roberta_tokenizer, text)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    bert = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(num_top_words).indices.tolist())
    return {'bert': bert}


if __name__ == '__main__':
    text = "For most people, <mask> is the most important holiday."
    json_obj = get_all_predictions(text)
    print(json_obj)
