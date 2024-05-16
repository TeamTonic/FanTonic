import aiohttp
import asyncio
import os


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
        
async def fon_text_to_french(text:str):
    API_URL = "https://api-inference.huggingface.co/models/masakhane/m2m100_418M_fon_fr_rel_news"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    async with aiohttp.ClientSession() as session:
        # with open(filename, "rb") as f:
        #     data = f.read()
        async with session.post(API_URL, headers=headers, data=text) as response:
            return await response.json()

# output = query("sample1.flac")


async def main():
    output = await fon_speech_to_text("/home/isayahc/projects/FanTonic/edba837d-d79b-4945-8159-ecaa3677b12e.flac")
    print(output)
    
    # alt_output = await fon_speech_to_english("/home/isayahc/projects/FanTonic/edba837d-d79b-4945-8159-ecaa3677b12e.flac")
    # print(alt_output)
    
    alt_alt_output = await fon_text_to_french(output['text'])
    print(alt_alt_output)
    x=0

if __name__ == '__main__':
    asyncio.run(main())
