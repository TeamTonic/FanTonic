# FanTonic

![Fan-Tonic](https://git.tonic-ai.com/Tonic-AI/FanTonic/FanTonic/-/raw/main/docs/Fan-Tonic.png?ref_type=heads)

## How FanTonic Works

FanTonic is an innovative live voice application geared towards enhancing communication for speakers of the Fan language from Benin by providing immediate vocal responses in the same language. Below is a detailed explanation and architecture of how the application functions.

### System Architecture Overview:



1. **Microphone Input Capture:**
   - The application begins its process by capturing live audio input from the user through a microphone. This audio is expected to be in the Fan language.

2. **Voice to Text Conversion:**
   - The captured audio data is processed through a Fon Automatic Speech Recognition (ASR) system specifically tailored to recognize and transcribe the Fan language. This ASR technology efficiently converts spoken words into written text.

3. **Text Processing:**
   - The transcribed text is sent to the LlamaIndex message processing system. Here, the processed messages are further refined for clarity and context.
   - Parallelly, the text interacts with a self-hosted Aya model, which ensures responses are not only accurate but also contextually relevant based on previous interactions. This is achieved through the integration with CosmosDB, which conducts memory and reference attribute generation storage (RAG) to provide a personalized communication experience.

4. **Chat Memory Storage:**
   - To enhance user experience by making interactions more personalized and context-aware, the chat history is stored securely in CosmosDB. This storage allows the system to learn and adapt from previous conversations, making each interaction increasingly precise and user-specific.

5. **Response Generation:**
   - Once the message is processed and a suitable response is formulated, this text is converted back into the Fan language using Facebook’s Massively Multilingual Speech (MMS) Text-to-Speech Models. This ensures that the spoken output is natural and fluid.

6. **Multimedia Integration:**
   - In addition to providing voice responses, FanTonic integrates multimedia capabilities. Using Azure OpenAI DALL-E and Azure embeddings, the application can generate related images and immersive content to accompany the text responses. This enhances the engagement and overall user experience.

7. **Azure OpenAI Utilization:**
   - Finally, to complete the multimedia experience and refine the application’s capabilities, FanTonic uses Azure OpenAI services. This integration supports various functionalities including advanced image generation and embedding computations, contributing to a robust, multi-faceted user interaction.

### Technical Specifications:
  
- **Platforms Used:**
  - Fon Automatic Speech Recognition (ASR) for voice to text conversion.
  - LlamaIndex and Aya model for text processing.
  - CosmosDB for data storage and retrieval.
  - Facebook MMS for text to speech conversion.
  - Azure OpenAI DALL-E and Azure embeddings for image and multimedia content generation.

By employing a combination of advanced speech recognition, contextual processing models, and dynamic storage solutions, FanTonic not only facilitates but also enriches communication for the Fan-speaking community, encompassing a richer conversational experience using state-of-the-art technology.