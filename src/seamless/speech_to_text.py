import aiohttp
import asyncio
import os


async def fon_speech_to_text(filename):
    API_URL = "https://api-inference.huggingface.co/models/speechbrain/asr-wav2vec2-dvoice-fongbe"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    async with aiohttp.ClientSession() as session:
        with open(filename, "rb") as f:
            data = f.read()
        async with session.post(API_URL, headers=headers, data=data) as response:
            return await response.json()

async def main():
    output = await fon_speech_to_text("/home/isayahc/projects/FanTonic/edba837d-d79b-4945-8159-ecaa3677b12e.flac")
    print(output)

if __name__ == '__main__':
    asyncio.run(main())
