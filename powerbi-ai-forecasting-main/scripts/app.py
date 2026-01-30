from flask import Flask, request, jsonify
from rag_chain import get_qa_chain

app = Flask(__name__)
qa_chain = get_qa_chain()

@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question")
    answer = qa_chain.run(user_question)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
