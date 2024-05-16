import aiohttp
import asyncio
import os
import wavio

async def fon_speech_to_text(filename):
    API_URL = "https://api-inference.huggingface.co/models/speechbrain/asr-wav2vec2-dvoice-fongbe"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            data = f.read()
        async with session.post(API_URL, headers=headers, data=data, ) as response:
            return await response.json()


async def fon_speech_to_english(filename):
    API_URL = "https://api-inference.huggingface.co/models/OctaSpace/wav2vec2-bert-fongbe"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            data = f.read()
        async with session.post(API_URL, headers=headers, data=data) as response:
            return await response.json()
        
# async def fon_text_to_french(text:str):
#     API_URL = "https://api-inference.huggingface.co/models/masakhane/m2m100_418M_fon_fr_rel_news"
#     headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
#     async with aiohttp.ClientSession() as session:
#         # with open(filename, "rb") as f:
#         #     data = f.read()
#         async with session.post(API_URL, headers=headers, data=text) as response:
#             return await response.json()

async def fon_text_to_french(text: str):
    API_URL = "https://api-inference.huggingface.co/models/masakhane/m2m100_418M_fon_fr_rel_news"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.post(API_URL, headers=headers, data=text) as response:
                result = await response.json()
                if 'error' not in result or result.get('error') != 'Model masakhane/m2m100_418M_fon_fr_rel_news is currently loading':
                    break  # Exit loop if response is not in loading state or does not contain error
                await asyncio.sleep(5)  # Wait for 5 seconds before checking again

    return result


async def english_text_to_fon_speech(text: str, save_file_name:str) -> bytes:
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-fon"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    
    async with aiohttp.ClientSession() as session:
        payload = {"inputs": text}
        while True:
            async with session.post(API_URL, headers=headers, json=payload) as response:
                result = await response.read()
                if isinstance(result, bytes) and result:
                    break  # Exit loop if response is received
                await asyncio.sleep(5)  # Wait for 5 seconds before checking again
                
    if isinstance(result, bytes):
        with open(save_file_name, 'wb') as f:
            f.write(result)
    else:
        print("Error: Audio bytes not received or not in correct format.")


    # return result

# output = query("sample1.flac")


async def main():
    output = await fon_speech_to_text("/home/isayahc/projects/FanTonic/edba837d-d79b-4945-8159-ecaa3677b12e.flac")
    print(output)
    
    # alt_output = await fon_speech_to_english("/home/isayahc/projects/FanTonic/edba837d-d79b-4945-8159-ecaa3677b12e.flac")
    # print(alt_output)
    
    alt_alt_output = await fon_text_to_french(output['text'])
    print(alt_alt_output)
    
    dd  = await english_text_to_fon_speech("Hi my name is Isayah am i am trying a thing to live life","sample.wav")
    x=0

if __name__ == '__main__':
    asyncio.run(main())
