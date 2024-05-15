# from gradio_client import Client

# client = Client("https://facebook-seamless-m4t-v2-large.hf.space/--replicas/qmxoc/",upload_files=False)
# result = client.predict(
# 		"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",	# filepath  in 'Input speech' Audio component
# 		"Afrikaans",	# Literal[Afrikaans, Amharic, Armenian, Assamese, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Burmese, Cantonese, Catalan, Cebuano, Central Kurdish, Croatian, Czech, Danish, Dutch, Egyptian Arabic, English, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hebrew, Hindi, Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lao, Lithuanian, Luo, Macedonian, Maithili, Malayalam, Maltese, Mandarin Chinese, Marathi, Meitei, Modern Standard Arabic, Moroccan Arabic, Nepali, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Norwegian Nynorsk, Nyanja, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Latvian, Standard Malay, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, West Central Oromo, Western Persian, Yoruba, Zulu]  in 'Source language' Dropdown component
# 		"Bengali",	# Literal[Bengali, Catalan, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Maltese, Mandarin Chinese, Modern Standard Arabic, Northern Uzbek, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swahili, Swedish, Tagalog, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, Western Persian]  in 'Target language' Dropdown component
# 							api_name="/s2st",
# )
# print(result)
from gradio_client import Client
seamless_client = Client("facebook/seamless_m4t")

def process_speech(input_language, audio_input):
    """
    processing sound using seamless_m4t
    """
    if audio_input is None:
        return "no audio or audio did not save yet \nplease try again ! "
    print(f"audio : {audio_input}")
    print(f"audio type : {type(audio_input)}")
    # job = seamless_client.submit(5, "add", 4, api_name="/predict")
    #         job.status()
    #         job.result()  # blocking call
    job = seamless_client.submit(
        "S2TT",
        "file",
        None,
        audio_input,
        "",
        input_language,
        "English",
        api_name="/run",
    )
    job.status()
    job.result()  # blocking call
    out = out[1]  # get the text
    try:
        return f"{out}"
    except Exception as e:
        return f"{e}"
    
if __name__ == "__main__":
    tts = process_speech("Mandarin Chinese","/home/isayahc/projects/FanTonic/Alina_Chn.wav")