# ./src/utilities.py

# from llama_index.vector_stores.azurecosmosmongo.base import VectorStoreQuery
from llama_index.core.memory.chat_summary_memory_buffer import ChatSummaryMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.utils import get_tokenizer
from typing import List, Dict, Union, NoReturn, Tuple
from llama_index.core.vector_stores.types import VectorStoreQueryResult
import os
import json
import uuid

# from src.transcriber import Transcriber
import dotenv
import sys
import os
import pandas as pd
import numpy as np
import tiktoken
from llama_index.core.settings import _Settings
# from llama_index.llms.azure_openai.base import OpenAI
import openai
# from langchain_openai import AzureOpenAI
from llama_index.llms.azure_openai.base import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI


class AzureAIManager:
    def __init__(self):
        dotenv.load_dotenv()  # Load environment variables

        # Retrieve environment variables
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_api_version = os.getenv('AZURE_OPENAI_VERSION')

        # Hardcoded or retrieved from environment
        self.engine = os.getenv('OPENAI_ENGINE')
        self.model = os.getenv('OPENAI_MODEL')  
        # self.model = os.getenv('OPENAI_MODEL')  
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '1.0')) 

        # Initialize the Azure OpenAI API
        self.llm = AzureOpenAI(
            engine=self.engine,
            model=self.model,
            temperature=self.temperature,
            api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.azure_api_version,
        )

    # def get_llm(self) -> Union[OpenAI, AzureOpenAI]:
    def get_llm(self) -> AzureOpenAI:
        return self.llm
    
if __name__ == '__main__':
    ai_manager = AzureAIManager()
    llm = ai_manager.get_llm()