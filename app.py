from flask import Flask, render_template, jsonify, request
from neural_network import NeuralNetwork, MODEL_PATH

app = Flask(__name__)

nn = NeuralNetwork()

if nn.model_exists():
    nn.load()

@app.route("/train", methods=["POST"])
def train():
    data = nn.train_one_epoch()
    return jsonify(data)

@app.route("/fast_train", methods=["POST"])
def fast_train():
    epochs = int(request.json.get("epochs", 5000))
    for _ in range(epochs):
        nn.train_one_epoch()
    nn.save()                      
    return jsonify(nn.get_current_state())

@app.route("/reset", methods=["POST"])
def reset():
    nn.reset()
    return jsonify(nn.get_current_state())

@app.route("/state")
def state():
    return jsonify(nn.get_current_state())


@app.route("/save", methods=["POST"])
def save_model():
    nn.save()
    return jsonify({"ok": True, "epoch": nn.epoch, "path": MODEL_PATH})

@app.route("/load", methods=["POST"])
def load_model():
    if not nn.model_exists():
        return jsonify({"error": "Nenhum modelo salvo encontrado."}), 404
    nn.load()
    return jsonify(nn.get_current_state())


@app.route("/predict", methods=["POST"])
def predict():

    body = request.json

    features = [
        float(body.get("moradores",    2)),
        float(body.get("tvs",          1)),
        float(body.get("ventiladores", 1)),
        float(body.get("geladeiras",   1)),
        float(body.get("horas_ar",     0)),
        float(body.get("lampadas",     6)),
        float(body.get("area",        60)),
    ]

    resultado = nn.predict_custom(features)

    TARIFA = 0.90

    return jsonify({
        "kwh": resultado["kwh"],
        "reais": round(resultado["kwh"] * TARIFA, 2),
        "level": resultado["level"],
        "message": resultado["message"]
    })


@app.route("/corrupt", methods=["POST"])
def corrupt():
    nn.corrupt_memory()
    return jsonify(nn.get_current_state())


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)