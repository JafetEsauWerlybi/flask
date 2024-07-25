from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados y el escalador
gb_model = joblib.load('gbProyecto.pkl')
rf_model = joblib.load('randomForestPry.pkl')
scaler = joblib.load('scalerPry.pkl')

app.logger.debug('Modelos y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('proyecto.html')  # Cambia 'titanic.html' por 'index.html' si es necesario

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request en formato JSON
        data = request.get_json()

        # Extraer las características
        calle = data['Calle']
        colonia = data['Colonia']
        numero_interior = data['NumeroInterior']
        numero_exterior = data['NumeroExterior']
        estado = data['Estado']
        ciudad = data['Ciudad']
        usuario_id = data['UsuarioID']
        lat = data['Lat']
        long = data['Long']

        # Crear un DataFrame con las características necesarias
        input_data = pd.DataFrame({
            'Calle': [0],
            'Colonia': [0],
            'NumeroInterior': [0],
            'NumeroExterior': [0],
            'Estado': [0],
            'Ciudad': [0],
            'UsuarioID': [0],
            'Lat': [lat],
            'Long': [long]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        scaled_data_for_prediction = scaled_data[:, [-1,-2,0]]  # Asegúrate de que estos índices son correctos
        scaled_data_for_predictionCla = scaled_data[:, [-1,-2]]  # Asegúrate de que estos índices son correctos

        # Realizar las predicciones con los datos escalados
        gb_prediction = gb_model.predict(scaled_data_for_predictionCla)
        rf_prediction = rf_model.predict(scaled_data_for_prediction)

        # Devolver las predicciones como JSON
        return jsonify({
            'GradientBoostingPrediction': gb_prediction.tolist(),
            'RandomForestPrediction': rf_prediction.tolist()
        })

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
