from flask import Flask, request, jsonify
import os
from predict import separate

app = Flask(__name__)

@app.route("/separate", methods=["POST"])
def sep():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    filepath = os.path.join("stem_sep", "temp.wav")
    f.save(filepath)

    separate(filepath, "stem_sep/vocals_out.wav", "stem_sep/karaoke_out.wav")
    return jsonify({"vocal_path": "stem_sep/vocals_out.wav", "karaoke_path": "stem_sep/karaoke_out.wav"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)