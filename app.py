from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('md2.pkl')
scaler = joblib.load('s3f.pkl')

app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('PrecioDeVenta.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request en formato JSON
        data = request.get_json() 

        Age = float(data['Age'])
        Present_Price = float(data['Present_Price'])
        Kms_Driven = float(data['Kms_Driven'])
        Fuel_Type_Diesel = float(data['Fuel_Type_Diesel'])

        # Crear un DataFrame con todas las características necesarias
        input_data = pd.DataFrame({
            'Age': [Age],
            'Present_Price': [Present_Price],
            'Kms_Driven': [Kms_Driven],
            'Owner': [0],
            'Fuel_Type_CNG': [0],
            'Fuel_Type_Diesel': [Fuel_Type_Diesel],
            'Fuel_Type_Petrol': [0],
            'Seller_Type_Dealer': [0],
            'Seller_Type_Individual': [0],
            'Transmission_Automatic': [0],
            'Transmission_Manual': [0],
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [0, 1, 2, 5]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = float(prediccion[0])  # Convertir a float si es necesario
        return jsonify({'prediction': prediction_value})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
