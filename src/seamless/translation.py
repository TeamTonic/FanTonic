from gradio_client import Client
import os

languages = {
    "English": "eng",
    "Modern Standard Arabic": "arb",
    "Bengali": "ben",
    "Catalan": "cat",
    "Czech": "ces",
    "Mandarin Chinese": "cmn",
    "Welsh": "cym",
    "Danish": "dan",
    "German": "deu",
    "Estonian": "est",
    "Finnish": "fin",
    "French": "fra",
    "Hindi": "hin",
    "Indonesian": "ind",
    "Italian": "ita",
    "Japanese": "jpn",
    "Korean": "kor",
    "Maltese": "mlt",
    "Dutch": "nld",
    "Western Persian": "pes",
    "Polish": "pol",
    "Portuguese": "por",
    "Romanian": "ron",
    "Russian": "rus",
    "Slovak": "slk",
    "Spanish": "spa",
    "Swedish": "swe",
    "Swahili": "swh",
    "Telugu": "tel",
    "Tagalog": "tgl",
    "Thai": "tha",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Northern Uzbek": "uzn",
    "Vietnamese": "vie"
}


client = Client("https://facebook-seamless-m4t-v2-large.hf.space/--replicas/qmxoc/")

# result = client.predict(
#         https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav,    # filepath  in 'Input speech' Audio component
#         Afrikaans,    # Literal[Afrikaans, Amharic, Armenian, Assamese, Basque, Belarusian, Bengali, Bosnian, Bulgarian, Burmese, Cantonese, Catalan, Cebuano, Central Kurdish, Croatian, Czech, Danish, Dutch, Egyptian Arabic, English, Estonian, Finnish, French, Galician, Ganda, Georgian, German, Greek, Gujarati, Halh Mongolian, Hebrew, Hindi, Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kyrgyz, Lao, Lithuanian, Luo, Macedonian, Maithili, Malayalam, Maltese, Mandarin Chinese, Marathi, Meitei, Modern Standard Arabic, Moroccan Arabic, Nepali, North Azerbaijani, Northern Uzbek, Norwegian Bokmål, Norwegian Nynorsk, Nyanja, Odia, Polish, Portuguese, Punjabi, Romanian, Russian, Serbian, Shona, Sindhi, Slovak, Slovenian, Somali, Southern Pashto, Spanish, Standard Latvian, Standard Malay, Swahili, Swedish, Tagalog, Tajik, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, West Central Oromo, Western Persian, Yoruba, Zulu]  in 'Source language' Dropdown component
#         Bengali,    # Literal[Bengali, Catalan, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Maltese, Mandarin Chinese, Modern Standard Arabic, Northern Uzbek, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swahili, Swedish, Tagalog, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh, Western Persian]  in 'Target language' Dropdown component
#                             api_name="/s2st"
# )
# print(result)

def translate_language(
    url_path:str,
    output_language:str,
    input_language:str,
    client = Client(
        "https://facebook-seamless-m4t-v2-large.hf.space/--replicas/qmxoc/",
        hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        upload_files=False
        ),
    ) -> str:
    
    result = client.predict(
        url_path,
        output_language,
        input_language,
        api_name="/s2st"
    )
    
    return result


if __name__ == "__main__":
    audio = "/home/isayahc/projects/FanTonic/Alina_Chn.wav"
    input_language = "Mandarin Chinese"
    output_language = "English"
    data = translate_language(audio,output_language,input_language)
    x=0
    