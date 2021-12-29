
from transformers import AutoTokenizer
from ecco import util


class TestTokenizers:
    def test_gpt_tokenizer(self):
        tokenizers = ['gpt2', 'bert-base-uncased']
        model_name = 'distilgpt2'
        config = util.load_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizers[0])
        token_ids = tokenizer(' tokenization')['input_ids']
        is_partial_1 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[0]))
        is_partial_2 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[1]))
        assert not is_partial_1
        assert is_partial_2

    def test_bert_tokenizer(self):
        model_name = 'bert-base-uncased'
        config = util.load_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_ids = tokenizer(' tokenization')['input_ids']
        is_partial_1 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[1])) # skip CLS
        is_partial_2 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[2]))
        assert not is_partial_1
        assert is_partial_2

    def test_t5_tokenizer(self):
        model_name = 't5-small'
        config = util.load_config(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_ids = tokenizer(' tokenization')['input_ids']
        is_partial_1 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[0]))
        is_partial_2 = util.is_partial_token(config,
                                             tokenizer.convert_ids_to_tokens(token_ids[1]))
        assert not is_partial_1
        assert is_partial_2


