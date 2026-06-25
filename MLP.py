import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


np.random.seed(1)

weights_input_hidden = np.random.uniform(-1, 1, (2,4))

weights_hidden_output = np.random.uniform(-1, 1, (4,1))

learning_rate = 0.5

epochs = 5000


for epoch in range(epochs):

   

    hidden_input = np.dot(
        X,
        weights_input_hidden
    )

    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(
        hidden_output,
        weights_hidden_output
    )

    final_output = sigmoid(final_input)


    error = y - final_output

    mse = np.mean(np.square(error))

    print(
        f'Epoca: {epoch + 1} '
        f'| Erro: {mse:.6f}'
    )


    d_output = (
        error *
        sigmoid_derivative(final_output)
    )

    hidden_error = d_output.dot(
        weights_hidden_output.T
    )

    d_hidden = (
        hidden_error *
        sigmoid_derivative(hidden_output)
    )

    weights_hidden_output += (
        hidden_output.T.dot(d_output)
        * learning_rate
    )

    weights_input_hidden += (
        X.T.dot(d_hidden)
        * learning_rate
    )

print('\nRESULTADO FINAL:\n')

predictions = (
    final_output > 0.5
).astype(int)

for i in range(len(X)):

    print(
        f'Entrada: {X[i]}'
        f'| Esperado: {y[i][0]} '
        f'| Previsto: {predictions[i][0]}'
    )

print(f'\nTotal de epocas: {epochs}')
print(f'Erro final: {mse:.6f}')