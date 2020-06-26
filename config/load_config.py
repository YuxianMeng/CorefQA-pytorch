#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# load config 


import copy
import json
import six


class Config(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def update_args(self, args_namespace):
        args_dict = args_namespace.__dict__
        print("Please notice that merge the args_dict and json_config ... ...")
        for args_key, args_value in args_dict.items():
            if args_key not in self.__dict__.keys():
                self.__dict__[args_key] = args_value 
            else:
                print("update the config from args input ... ...")
                self.__dict__[args_key] = args_value

    @classmethod 
    def from_dict(cls, json_object):
        config_instance = Config()
        for key, value in json_object.items():
            try:
                tmp_value = Config.from_json_dict(value)
                config_instance.__dict__[key] = tmp_value 
            except:
                config_instance.__dict__[key] = value 
        return config_instance

    @classmethod 
    def from_json_file(cls, json_file):
        with open(json_file, "r") as f:
            text = f.read()
        return Config.from_dict(json.loads(text))

    @classmethod 
    def from_json_dict(cls, json_str):
        return Config.from_dict(json_str)

    @classmethod 
    def from_json_str(cls, json_str):
        return Config.from_dict(json.loads(json_str))


    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output = {k: v.to_dict() if isinstance(v, Config) else v for k, v in output.items()}
        return output

    def print_config(self):
        model_config = self.to_dict()
        json_config = json.dumps(model_config, indent=2)
        print(json_config)
        return json_config

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent = 2, sort_keys=True) + "\n"


class BertConfig(object):
    """Configuration for `BertModel`."""
    def __init__(self,
              vocab_size,
              hidden_size=768,
              num_hidden_layers=12,
              num_attention_heads=12,
              intermediate_size=3072,
              hidden_act="gelu",
              hidden_dropout_prob=0.1,
              attention_probs_dropout_prob=0.1,
              max_position_embeddings=512,
              type_vocab_size=16,
              initializer_range=0.02):
      """Constructs BertConfig.

      Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
      self.vocab_size = vocab_size
      self.hidden_size = hidden_size
      self.num_hidden_layers = num_hidden_layers
      self.num_attention_heads = num_attention_heads
      self.hidden_act = hidden_act
      self.intermediate_size = intermediate_size
      self.hidden_dropout_prob = hidden_dropout_prob
      self.attention_probs_dropout_prob = attention_probs_dropout_prob
      self.max_position_embeddings = max_position_embeddings
      self.type_vocab_size = type_vocab_size
      self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
          config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
          text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
