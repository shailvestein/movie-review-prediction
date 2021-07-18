import streamlit as st
import pickle
import tensorflow as tf

#############################################################################################
#############################################################################################

# Loading tfidf trained vectorizer to convert taken review in numbers

with open("vect.pkl", 'rb') as v:
    vect = pickle.load(v)

# with open("4th_model_0_77_accuracy.h5", 'rb') as m:
#     model = pickle.load(m)
# Defining and loading model weights for prediction of taken review

input_layer = tf.keras.Input(shape=(2662,))
layer_1 = tf.keras.layers.Dense(1000, activation='relu')(input_layer)
layer_2 = tf.keras.layers.Dense(500, activation='relu')(layer_1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(layer_2)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='IMBD')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights("4th_model_0_77_accuracy.h5")
# model = pickle.load(open("4th_model_0_77_accuracy.h5", 'rb'))
##############################################################################################
##############################################################################################

# Below we are defining a streamlit webpage which will take user input and predict polarity of taken review

st.title("Movie review polarity prediction")
st.write("Here we will predict polarity of your review for a movie whether or not it is positive.")
# st.text_input(label="Enter your review below")


with st.form("my_form"):
    # st.write("Enter your review below")
    review = st.text_input(label='Please enter your review below')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Predict")


if submitted:
    if review == '':
        st.text(f"Please enter your review and press predict")
    else:
        input_review = vect.transform([review])
        input_review = input_review.toarray()

        predicted_score = model.predict(input_review)

        if predicted_score < 0.5:
            st.text(f"You've given negative review")
        else:
            st.text(f"You've given positive review")
