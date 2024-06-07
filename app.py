from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
#model = joblib.load('modelo.pkl')
model = joblib.load('decision_tree_model.pkl')

app.logger.debug('Modelo cargado correctamente.')


@app.route('/')
def home():
    return render_template('Pasajeros.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Obtener los datos enviados en el request
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
    
    # Asegúrate de que el DataFrame tenga las columnas en el mismo orden que el modelo espera
        df = df[['FrequentFlyer_Yes', 'Age', 'BookedHotelOrNot_Yes', 'ServicesOpted', 'AnnualIncomeClass']]

    # Hacer predicción
        prediction = model.predict(df)
    
    # Devolver la predicción como JSON
        predictionS =int(prediction[0])
        prediccionStr = ""
        if predictionS == 0:
            prediccionStr = "El cliente no abandonará"
        else:
            prediccionStr = "El cliente si abandonará"

        return jsonify({'prediction': prediccionStr})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
