# import pickle
# import gradio as gr
# import numpy as np
# import pandas as pd
# import joblib

# with open('model/model.pkl', 'rb') as f:
#     model = pickle.load(f)# Load the model from the file

# # Define a function for prediction
# def predict(seat_comfort, cabin_service, food_bev, entertainment, ground_service, value_for_money, traveller_type, cabin_class):
#     # Prepare the input data as dataframe
#     input_data= pd.DataFrame({
#         'seat_comfort': [seat_comfort],
#         'cabin_service': [cabin_service],
#         'food_bev': [food_bev],
#         'entertainment': [entertainment],
#         'ground_service': [ground_service],
#         'value_for_money': [value_for_money],
#         'traveller_type_Business': [1 if traveller_type== 'Business' else 0],
#         'traveller_type_Couple Leisure': [1 if traveller_type== 'Couple Leisure' else 0],
#         'traveller_type_Family Leisure': [1 if traveller_type== 'Family Leisure' else 0],
#         'traveller_type_Solo Leisure': [1 if traveller_type== 'Solo Leisure' else 0],
#         'cabin_Business Class': [1 if cabin_class== 'Business Class' else 0],
#         'cabin_Economy Class': [1 if cabin_class== 'Economy Class' else 0],
#         'cabin_First Class': [1 if cabin_class== 'First Class' else 0],
#         'cabin_Premium Economy': [1 if cabin_class== 'Premium Economy' else 0],

#     })

#     # Make predictions using the model
#     prediction= model.predict(input_data)

#     # Convert numerical output to labels
#     output_label = "Recommend" if prediction == 1 else "Not Recommend"

#     return output_label

# # Define the Gradio interface inputs and outputs
# inputs= [
#     gr.Slider(0,5, step=1, label= 'Seat Comfort'),
#     gr.Slider(0,5, step=1, label= 'Cabin Service'),
#     gr.Slider(0,5, step=1, label= 'Food and Beverage'),
#     gr.Slider(0,5, step=1, label= 'Entertainment'),
#     gr.Slider(0,5, step=1, label= 'Ground Service'),
#     gr.Slider(0,5, step=1, label= 'Value for Money'),
#     gr.Dropdown(['Business', 'Couple Leisure', 'Family Leisure', 'Solo Leisure'], label= "Traveller Type"),
#     gr.Dropdown(['Business Class', 'Economy Class', 'First Class', 'Premium Economy'], label="Cabin Class")
# ]

# # Gradio Output
# output = gr.Textbox(label="Prediction")

# # Create the fradio interface
# interface= gr.Interface(fn=predict, inputs=inputs, outputs= output, title="Airline Satisfaction Prediction")

# # Launch the gradio app
# interface.launch()


import pickle
import gradio as gr
import pandas as pd
import os

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function for prediction
def predict(seat_comfort, cabin_service, food_bev, entertainment, ground_service, value_for_money, traveller_type, cabin_class):
    input_data = pd.DataFrame({
        'seat_comfort': [seat_comfort],
        'cabin_service': [cabin_service],
        'food_bev': [food_bev],
        'entertainment': [entertainment],
        'ground_service': [ground_service],
        'value_for_money': [value_for_money],
        'traveller_type_Business': [1 if traveller_type == 'Business' else 0],
        'traveller_type_Couple Leisure': [1 if traveller_type == 'Couple Leisure' else 0],
        'traveller_type_Family Leisure': [1 if traveller_type == 'Family Leisure' else 0],
        'traveller_type_Solo Leisure': [1 if traveller_type == 'Solo Leisure' else 0],
        'cabin_Business Class': [1 if cabin_class == 'Business Class' else 0],
        'cabin_Economy Class': [1 if cabin_class == 'Economy Class' else 0],
        'cabin_First Class': [1 if cabin_class == 'First Class' else 0],
        'cabin_Premium Economy': [1 if cabin_class == 'Premium Economy' else 0],
    })

    # Make predictions
    prediction = model.predict(input_data)

    # Convert numerical output to labels
    output_label = "Recommend" if prediction[0] == 1 else "Not Recommend"
    return output_label

# Define the inputs
inputs = [
    gr.Slider(0, 5, step=1, label="Seat Comfort", info="Rate seat comfort (0-5)"),
    gr.Slider(0, 5, step=1, label="Cabin Service", info="Rate cabin service (0-5)"),
    gr.Slider(0, 5, step=1, label="Food and Beverage", info="Rate food & beverage (0-5)"),
    gr.Slider(0, 5, step=1, label="Entertainment", info="Rate entertainment (0-5)"),
    gr.Slider(0, 5, step=1, label="Ground Service", info="Rate ground service (0-5)"),
    gr.Slider(0, 5, step=1, label="Value for Money", info="Rate value for money (0-5)"),
    gr.Dropdown(
        ['Business', 'Couple Leisure', 'Family Leisure', 'Solo Leisure'],
        label="Traveller Type",
        info="Select traveller type",
    ),
    gr.Dropdown(
        ['Business Class', 'Economy Class', 'First Class', 'Premium Economy'],
        label="Cabin Class",
        info="Select cabin class",
    )
]

# Define the output
output = gr.Textbox(label="Prediction", lines=2, placeholder="Result will appear here")

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="Airline Satisfaction Prediction",
    description="Provide your travel experience ratings and get a recommendation prediction.",
    theme="default",  # You can use themes like `default`, `huggingface`, or `compact`.
    live=False
)

# Launch the Gradio app
interface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
