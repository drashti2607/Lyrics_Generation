# generator/middleware.py


# # old middleware for model work on later
# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from django.conf import settings
# from keras.preprocessing.sequence import pad_sequences

# class LyricsGeneratorMiddleware:
#     model = None
#     tokenizer = None

#     def __init__(self, get_response):
#         self.get_response = get_response
#         model_path = os.path.join(settings.BASE_DIR, 'generator', 'lyrics_generation_model.h5')
#         tokenizer_path = os.path.join(settings.BASE_DIR, 'generator', 'tokenizer.pkl')
        
#         print(f"Loading model from: {model_path}")
#         self.__class__.model = tf.keras.models.load_model(model_path)
#         print(f"Loading tokenizer from: {tokenizer_path}")
#         with open(tokenizer_path, 'rb') as handle:
#             self.__class__.tokenizer = pickle.load(handle)

#     def __call__(self, request):
#         request.model = self.__class__.model
#         request.tokenizer = self.__class__.tokenizer
#         response = self.get_response(request)
#         return response

#     @classmethod
#     def generate_lyrics(cls, seed_text, next_words=50):
#         model = cls.model
#         tokenizer = cls.tokenizer

#         print(f"Generating lyrics for seed text: {seed_text}")
#         for _ in range(next_words):
#             token_list = tokenizer.texts_to_sequences([seed_text])[0]
#             token_list = pad_sequences([token_list], maxlen=100, padding='pre')
#             predicted = np.argmax(model.predict(token_list), axis=-1)
#             output_word = ""
#             for word, index in tokenizer.word_index.items():
#                 if index == predicted:
#                     output_word = word
#                     break
#             seed_text += " " + output_word
#         return seed_text


# # generator/middleware.py

# import os
# from django.conf import settings
# from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# class LyricsGeneratorMiddleware:
#     model = None
#     tokenizer = None

#         self.get_response = get_response
#         model_name = "gpt2"  
        
#         print(f"Loading model and tokenizer from: {model_name}")
#         self.__class__.model = TFGPT2LMHeadModel.from_pretrained(model_name)
#         self.__class__.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#     def __call__(self, request):
#         request.model = self.__class__.model
#         request.tokenizer = self.__class__.tokenizer
#         response = self.get_response(request)
#         return response

#     @classmethod
#     def generate_lyrics(cls, seed_text, next_words=50):
#         model = cls.model
#         tokenizer = cls.tokenizer
      

#         print(f"Generating lyrics for seed text: {seed_text}")
#         input_ids = tokenizer.encode(seed_text, return_tensors='tf')

#         # Generate text
#         generated_text_ids = model.generate(input_ids, max_length=next_words + len(input_ids[0]), num_return_sequences=1)
#         generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
        
#         return generated_text




# # old middleware for model work on later
# import os
# import pickle
# import numpy as np
# import tensorflow as tf
# from django.conf import settings
# from keras.preprocessing.sequence import pad_sequences

# class LyricsGeneratorMiddleware:
#     model = None
#     tokenizer = None

#     def __init__(self, get_response):
#         self.get_response = get_response
#         model_path = os.path.join(settings.BASE_DIR, 'generator', 'lyrics_generation_model.h5')
#         tokenizer_path = os.path.join(settings.BASE_DIR, 'generator', 'tokenizer.pkl')
        
#         print(f"Loading model from: {model_path}")
#         self.__class__.model = tf.keras.models.load_model(model_path)
#         print(f"Loading tokenizer from: {tokenizer_path}")
#         with open(tokenizer_path, 'rb') as handle:
#             self.__class__.tokenizer = pickle.load(handle)

#     def __call__(self, request):
#         request.model = self.__class__.model
#         request.tokenizer = self.__class__.tokenizer
#         response = self.get_response(request)
#         return response

#     @classmethod
#     def generate_lyrics(cls, seed_text, next_words=50):
#         model = cls.model
#         tokenizer = cls.tokenizer

#         print(f"Generating lyrics for seed text: {seed_text}")
#         for _ in range(next_words):
#             token_list = tokenizer.texts_to_sequences([seed_text])[0]
#             token_list = pad_sequences([token_list], maxlen=100, padding='pre')
#             predicted = np.argmax(model.predict(token_list), axis=-1)
#             output_word = ""
#             for word, index in tokenizer.word_index.items():
#                 if index == predicted:
#                     output_word = word
#                     break
#             seed_text += " " + output_word
#         return seed_text



# generator/middleware.py

import os
from django.conf import settings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LyricsGeneratorMiddleware:
    model = None
    tokenizer = None

    def __init__(self, get_response):
        self.get_response = get_response
        model_name = "gpt2"  
        
        print(f"Loading model and tokenizer from: {model_name}")
        self.__class__.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.__class__.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.__class__.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__class__.model.to(self.__class__.device)

    def __call__(self, request):
        request.model = self.__class__.model
        request.tokenizer = self.__class__.tokenizer
        request.device = self.__class__.device
        response = self.get_response(request)
        return response

    @classmethod
    def generate_lyrics(cls, seed_text, next_words=50):
        model = cls.model
        tokenizer = cls.tokenizer
        device = cls.device

        print(f"Generating lyrics for seed text: {seed_text}")
        input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)

        # Generate text with early stopping and adjusted parameters
        generated_text_ids = model.generate(
            input_ids, 
            max_length=next_words + len(input_ids[0]), 
            early_stopping=True,
            temperature=0.7,  
            top_p=0.9,        
            no_repeat_ngram_size=2  
        )
        generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
        
        # Post-process the generated text to format it like a song
        processed_text = cls.post_process_lyrics(generated_text)
        
        return processed_text

    @staticmethod
    def post_process_lyrics(text):
        lines = text.split(". ")
        formatted_lyrics = "\n".join(lines)
        return formatted_lyrics
