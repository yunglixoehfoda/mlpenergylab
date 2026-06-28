import numpy as np
import json
import os

MODEL_PATH = "model.json"

class NeuralNetwork:

    def __init__(self):
        self.reset()


    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def predict_custom(self, features: list):

        x = np.array(features, dtype=float)
        
        if np.all(x == 67):
            
            return {
                "kwh": 67,
                "level": "67",
                "message": "67 detectado. a profecia foi cumprida."
            }

        if x[6] <= 0:
            return {
                "kwh": 0,
                "level": "inexistente",
                "message": "casa inexistente, sem consumo de energia"
            }

        if np.any(x < 0):
            return {
                "kwh": 0,
                "level": "inválido",
                "message": "dados inválidos (valores negativos)"
            }

        x = x.reshape(1, -1)
        x_norm = (x - self.X_mean) / self.X_std

        Z1 = x_norm.dot(self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2

        kwh = max(0, float(self.denormalize_output(Z2).item()))

        if kwh < 100:
            level = "baixo"
            msg = "consumo leve, casa econômica"

        elif kwh < 400:
            level = "médio"
            msg = "consumo moderado, uso equilibrado"

        else:
            level = "alto"
            msg = "consumo elevado, atenção no gasto"

        return {
            "kwh": round(kwh, 1),
            "level": level,
            "message": msg
        }

       
    def get_energy_data(self):
        import numpy as np

        np.random.seed(42)

        X = []
        y = []

        for _ in range(150):

            moradores = np.random.randint(1, 7)
            tvs = np.random.randint(0, 5)
            ventiladores = np.random.randint(0, 6)
            geladeiras = np.random.randint(1, 3)
            horas_ar = np.random.randint(0, 10)
            lampadas = np.random.randint(1, 12)
            area_m2 = np.random.randint(30, 220)

            consumo = (
                60
                + moradores * 22
                + tvs * 18
                + ventiladores * 6
                + geladeiras * 45
                + horas_ar * 20
                + lampadas * 4
                + area_m2 * 0.25
            )

            consumo += np.random.normal(0, 18)
            consumo = max(30, consumo)

            X.append([
                moradores, tvs, ventiladores,
                geladeiras, horas_ar, lampadas, area_m2
            ])

            y.append([consumo])

        return np.array(X, dtype=float), np.array(y, dtype=float)


    def normalize(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_std  = float(y.std()) + 1e-8
        return (X - self.X_mean) / self.X_std, (y - self.y_mean) / self.y_std

    def denormalize_output(self, y_norm):
        return y_norm * self.y_std + self.y_mean


    def reset(self):
        self.X_raw, self.y_raw = self.get_energy_data()
        self.X, self.y = self.normalize(self.X_raw, self.y_raw)

        n_in, n_h = 7, 8

        scale_ih = np.sqrt(2.0 / n_in)
        scale_ho = np.sqrt(2.0 / n_h)

        self.W1 = np.random.randn(n_in, n_h) * scale_ih  
        self.b1 = np.zeros((1, n_h))                       
        self.W2 = np.random.randn(n_h, 1)   * scale_ho    
        self.b2 = np.zeros((1, 1))                          

        self.epoch         = 0
        self.learning_rate = 0.01
        self.error_history = []


    def _forward(self, X=None):
        if X is None:
            X = self.X
        self.Z1 = X.dot(self.W1) + self.b1   
        self.A1 = self.relu(self.Z1)           
        self.Z2 = self.A1.dot(self.W2) + self.b2  
        self.A2 = self.Z2                     
        return self.A1, self.A2


    def _backward(self, A1, A2):
        n   = self.X.shape[0]
        lr  = self.learning_rate

        # dL/dZ2 = (2/n) * (A2 - y)   
        dZ2 = (2.0 / n) * (A2 - self.y)   

        dW2 = A1.T.dot(dZ2)               
        db2 = dZ2.sum(axis=0, keepdims=True)  

    
        dA1 = dZ2.dot(self.W2.T)                     
        dZ1 = dA1 * self.relu_derivative(self.Z1)     

        dW1 = self.X.T.dot(dZ1)                       
        db1 = dZ1.sum(axis=0, keepdims=True)           

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


    def _build_state(self, A1, A2, mse):
        predictions_kwh = self.denormalize_output(A2)
        return {
            "epoch":         self.epoch,
            "error":         float(mse),
            "predictions":   predictions_kwh.flatten().tolist(),
            "expected":      self.y_raw.flatten().tolist(),
            "inputs":        self.X_raw.tolist(),
            "hidden_output": A1.tolist(),
            "weights_ih":    self.W1.tolist(),
            "bias_h":        self.b1.tolist(),
            "weights_ho":    self.W2.tolist(),
            "bias_o":        self.b2.tolist(),
            "history":       self.error_history,
            "feature_names": [
                "Moradores","TVs","Ventiladores",
                "Geladeiras","Horas de Ar","Lâmpadas","Área (m²)"
            ]
        }

    def get_current_state(self):
        A1, A2 = self._forward()
        mse = float(np.mean((self.y - A2) ** 2))
        return self._build_state(A1, A2, mse)

    def train_one_epoch(self):
        A1, A2 = self._forward()

        mse = float(np.mean((self.y - A2) ** 2))
        self.error_history.append(mse)

        self._backward(A1, A2)

        self.epoch += 1
        return self._build_state(A1, A2, mse)

    


    def save(self, path=MODEL_PATH):
        data = {
            "epoch":         self.epoch,
            "learning_rate": self.learning_rate,
            "error_history": self.error_history,
            "W1":  self.W1.tolist(),
            "b1":  self.b1.tolist(),
            "W2":  self.W2.tolist(),
            "b2":  self.b2.tolist(),
            "X_mean": self.X_mean.tolist(),
            "X_std":  self.X_std.tolist(),
            "y_mean": self.y_mean,
            "y_std":  self.y_std,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path=MODEL_PATH):
        with open(path) as f:
            d = json.load(f)
        self.epoch         = d["epoch"]
        self.learning_rate = d["learning_rate"]
        self.error_history = d["error_history"]
        self.W1     = np.array(d["W1"])
        self.b1     = np.array(d["b1"])
        self.W2     = np.array(d["W2"])
        self.b2     = np.array(d["b2"])
        self.X_mean = np.array(d["X_mean"])
        self.X_std  = np.array(d["X_std"])
        self.y_mean = d["y_mean"]
        self.y_std  = d["y_std"]
        self.X_raw, self.y_raw = self.get_energy_data()
        self.X, self.y = (self.X_raw - self.X_mean) / self.X_std, \
                         (self.y_raw - self.y_mean) / self.y_std

    def model_exists(self, path=MODEL_PATH):
        return os.path.exists(path)


    def corrupt_memory(self):
        self.W2 += np.random.uniform(-0.7, 0.7, self.W2.shape)
        self.W1 += np.random.uniform(-0.7, 0.7, self.W1.shape)
        self.b2 += np.random.uniform(-0.3, 0.3, self.b2.shape)
        self.b1 += np.random.uniform(-0.3, 0.3, self.b1.shape)



    
