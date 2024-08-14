# generator/views.py


# # for old model work on later
# from django.shortcuts import render
# from django.http import JsonResponse
# from generator.middleware import LyricsGeneratorMiddleware

# def index(request):
#     return render(request, 'generator/index.html')

# def generate_lyrics(request):
#     seed_text = request.GET.get('seed_text')
#     num_words = int(request.GET.get('num_words', 50))
#     if not seed_text:
#         return JsonResponse({'error': 'No seed text provided'}, status=400)

#     try:
#         print(f"Received request to generate lyrics with seed text: {seed_text} and num_words: {num_words}")
#         lyrics = LyricsGeneratorMiddleware.generate_lyrics(seed_text, next_words=num_words)
#         return JsonResponse({'lyrics': lyrics})
#     except Exception as e:
#         print(f"Error generating lyrics: {str(e)}")
#         return JsonResponse({'error': str(e)}, status=500)


# generator/views.py

from django.shortcuts import render
from django.http import JsonResponse
from generator.middleware import LyricsGeneratorMiddleware

def index(request):
    return render(request, 'generator/index.html')

def generate_lyrics(request):
    seed_text = request.GET.get('seed_text')
    num_words = int(request.GET.get('num_words', 50))
    if not seed_text:
        return JsonResponse({'error': 'No seed text provided'}, status=400)

    try:
        print(f"Received request to generate lyrics with seed text: {seed_text} and num_words: {num_words}")
        lyrics = LyricsGeneratorMiddleware.generate_lyrics(seed_text, next_words=num_words)
        return JsonResponse({'lyrics': lyrics})
    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
