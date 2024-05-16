from flask import Flask, request, jsonify
from src.gen_ai.chat_bot import conversation
from src.seamless.speech_to_text import (
    english_tts,
    french_tts,
    fon_speech_to_text,
    fon_text_to_french
)
from flask_cors import CORS 
import requests
import tempfile    
import asyncio


app = Flask(__name__)
CORS(app)

@app.route('/tts', methods=['POST'])
async def query_tts():
    """
    
    query string
    
    Keyword arguments:
    argument -- description
    Return: return_description
    ### Example
    `curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"blobUrl": "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav" "target_language":"english" }' \
    http://127.0.0.1:5000/tts`
    """
    
    
    data = request.get_json()  # Get JSON data from POST request
    
    blob_url = data['blobUrl']
    
    response = requests.get(blob_url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            # print("WAV file downloaded successfully as temporary file:", temp_file_path)
            # return temp_file_path
    
    
        if not data or 'target_language' not in data:
            return jsonify({'error': 'Invalid request. Please provide a "query_string" in the JSON data.'}), 400
        
        elif data['target_language'].lower() == "english":
            ai_input_message = await english_tts(temp_file_path)
        
        elif data['target_language'].lower() == "french":
            ai_input_message = await french_tts(temp_file_path)
            
        elif data['target_language'].lower() == "fongbe":
            ai_input_message = await fon_speech_to_text(temp_file_path)
            ai_input_message = await fon_text_to_french(ai_input_message)
        
        else:
            return jsonify({"error":"Invalid language"}), 400
            
        
        
        ai_output = ai_input_message['text']
        
        sample = conversation.invoke(
        {"question": ai_input_message['text']},
        return_only_outputs=True,
        )
        
        return_answer = sample['text']
    
    return jsonify({"question":ai_output, "answer":return_answer }), 200


@app.route('/query', methods=['POST'])
def query():
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    
    ### Example
    `curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"query_string": "Your query string here"}' \
    http://127.0.0.1:5000/query`
    """
    
    data = request.get_json()  # Get JSON data from POST request
    if not data or 'query_string' not in data:
        return jsonify({'error': 'Invalid request. Please provide a "query_string" in the JSON data.'}), 400
    
    sample = conversation.invoke(
    {"question": data['query_string']},
    return_only_outputs=True,
    )

    # Process the query string
    # query_string = data['query_string']
    return sample

if __name__ == '__main__':
    # app.run(debug=True)
    asyncio.run(app.run(debug=True))