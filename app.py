from flask import Flask, request, jsonify
# from src.server.structured_tools import agent_executor
# from src.server.cite_recognition_chain import cite_chain
# from alt_structured_agent import agent_executor
from src.gen_ai.chat_bot import conversation

app = Flask(__name__)

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
    app.run(debug=True)