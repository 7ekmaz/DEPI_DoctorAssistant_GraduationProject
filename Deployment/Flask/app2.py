import os
from werkzeug.utils import secure_filename
from email.policy import default
from wsgiref.validate import validator
from flask import Flask, flash, render_template, redirect, request, url_for, session,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import tensorflow as tf
from keras.models import load_model
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from PIL import Image
import azure.cognitiveservices.speech as speechsdk
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential
from pyngrok import conf, ngrok


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png',
                      'jpg', 'jpeg', 'gif', 'zip', 'csv', 'png'}

app = Flask(__name__)
app.config['SECRET_KEY'] = "Your_secret_string"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Azure services setup
speech_key = "dbd409c216334bd9914dc82fcc104c7b"
service_region = "eastasia"
qa_endpoint = "https://qabraintumor.cognitiveservices.azure.com/"
qa_key = "ebb5337031664837892317f393dbc60e"

qa_client = QuestionAnsweringClient(qa_endpoint, AzureKeyCredential(qa_key))

class Signup(db.Model):

    # sno, name, email, password, date

    # sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(40), nullable=False)
    email = db.Column(db.String(30), nullable=False, primary_key=True)
    password = db.Column(db.String(20), nullable=False)
    date = db.Column(db.String(12), nullable=True)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.name}"


# class Contact(db.Model):

#     # sno, name, email, message, date

#     sno = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(40), nullable=False)
#     email = db.Column(db.String(30), nullable=False)
#     message = db.Column(db.String(120), nullable=False)
#     date = db.Column(db.String(12), nullable=True)


@app.route("/", methods=['GET', 'POST'])
def home():

    # if(request.method == 'POST'):
    #     name = request.form.get('name')
    #     email = request.form.get('email')
    #     message = request.form.get('message')

    #     entry1 = Contact(name=name, email=email,
    #                      message=message, date=datetime.now())
    #     db.session.add(entry1)
    #     db.session.commit()

    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if(request.method == 'POST'):
        # Add entry to the data base
        email = request.form['email']
        password = request.form['password']
        print(f"Logging in with {email}, {password}")  # Debug print

        user = Signup.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            print("Login successful")  # Debug print
            return render_template("dashboard.html")
        else:
            print("Login failed")  # Debug print
            # If no user zis found or password does not match
            flash('Invalid email or password')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if(request.method == 'POST'):
        # Add entry to the data base
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        existing_user = Signup.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.')
            return redirect(url_for('signup'))
            
        password_hash = generate_password_hash(password)
        new_user = Signup(name=name, email=email, password=password_hash, date=datetime.now())
        #new_user = Signup(name=name, email=email, password=password, date=datetime.now())
        db.session.add(new_user)
        db.session.commit()
        return redirect("/")

    return render_template('signup.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/prediction_form')
def prediction_form():
    return render_template('pred_form.html')

@app.route('/brain_prediction_form')
def brain_prediction_form():
    return render_template('brain_pred_form.html')


@app.route('/uploads/file/<filename>')
def uploaded_file(filename):
    """Serve files directly from the 'uploads' directory."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/uploads/Ocular/<name>')
def download_file(name):
    """Render a template that includes a file from the 'uploads' directory."""
    return render_template("success.html", name=name)

@app.route('/uploads/Brain/<name>')
def download_brain_file(name):
    """Render a template that includes a file from the 'uploads' directory."""
    return render_template("brain_success.html", name=name)

@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    else:
        # Handle GET request or simply inform the user that this route is not for GET requests
        #flash('This route is only for file submissions.')
        return redirect(url_for('prediction_form'))  # Assuming 'home' is a valid endpoint
    
@app.route('/brain_success', methods=['POST', 'GET'])
def brain_success():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_brain_file', name=filename))
    else:
        # Handle GET request or simply inform the user that this route is not for GET requests
        #flash('This route is only for file submissions.')
        return redirect(url_for('brain_prediction_form'))  # Assuming 'home' is a valid endpoint

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

brain_model = tf.keras.models.load_model('brain_tumour_model.h5')

# Load individual models for different conditions
models = {
    'Diabetic': tf.keras.models.load_model('vgg19_all_datasets_full_df_diabetic_aug.h5'),
    'Glaucoma': tf.keras.models.load_model('vgg19_all_datasets_full_df_glaucoma_aug.h5'),
    'Cataract': tf.keras.models.load_model('vgg19_all_datasets_full_df_cataract_aug.h5'),
    'Macular Degeneration': tf.keras.models.load_model('vgg19_all_datasets_full_df_macular_degeneration_aug.h5'),
    'Hypertensive': tf.keras.models.load_model('vgg19_all_datasets_full_df_hypertensive_aug.h5'),
    'Myopia': tf.keras.models.load_model('vgg19_all_datasets_full_df_myopia_aug.h5'),
    'Other': tf.keras.models.load_model('vgg19_all_datasets_full_df_other_aug.h5')
}



def speak_prediction_lively(prediction_text):
    # Configure Azure Speech SDK
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_output = speechsdk.audio.AudioOutputConfig(filename="static/prediction_audio.wav")  # Save as a .wav file

    # Create a synthesizer
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # Synthesize speech
    result = synthesizer.speak_text_async(prediction_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")

def speak_answer_lively(answer_text):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_output = speechsdk.audio.AudioOutputConfig(filename="static/answer_audio.wav")  # Save as a .wav file

    # Create a synthesizer
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # Synthesize speech
    result = synthesizer.speak_text_async(answer_text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Answer speech synthesized successfully.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Answer speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")



@app.route("/predict_image/<filename>", methods=['GET', 'POST'])
def predict_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or the path is incorrect")
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0)

    # Generate predictions
    predictions = {name: model.predict(img_array) for name, model in models.items()}
    results = {name: 'Yes' if pred > 0.5 else 'No' for name, pred in predictions.items()}
    confidences = {name: round(float(pred) * 100, 2) for name, pred in predictions.items()}

    # Prepare the prediction result text for speech
    prediction_text = "The results are: "
    for condition, result in results.items():
        prediction_text += f"{condition}: {result}, with a confidence of {confidences[condition]} percent. "

    # Speak the prediction
    speak_prediction_lively(prediction_text)

    if request.method == 'POST':
        # Get the question from the form
        question = request.form.get('question')

        if question:
            # Get the answer from Azure QnA
            answer = ask_question_to_azure(question)
            speak_answer_lively(answer)

            # Pass the question, answer, predictions, and confidences back to the template
            return render_template('output.html', question=question, answer=answer, filename=filename, results=results, confidences=confidences)

    # Handle the initial GET request (loading the page)
    return render_template('output.html', results=results, confidences=confidences, filename=filename)



@app.route("/brain_prediction/<filename>", methods=['GET', 'POST'])
def brain_prediction(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or the path is incorrect")
    
    # Preprocess the image
    img = cv2.resize(img, (299, 299))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0

    # Make prediction using the brain model
    predictions = brain_model.predict(img_array, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)

    # Class mapping
    class_labels = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
    prediction = class_labels[predicted_class[0]]
    
    # Get the confidence scores for all classes
    confidences = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))

    # Speak the prediction using Azure Text-to-Speech
    speak_prediction_lively('You have ' + prediction)

    # Handle question submission
    question = None
    answer = None
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = ask_question_to_azure(question)
            speak_answer_lively(answer)  # Optionally speak the answer

    return render_template('brain_output.html', predictions=sorted_confidences, 
                           filename=filename, question=question, answer=answer)

def ask_question_to_azure(question):
    project_name = "BrainTumorQnA"
    deployment_name = "production"

    response = qa_client.get_answers(
        question=question,
        top=1,
        project_name=project_name,
        deployment_name=deployment_name
    )

    if response.answers:
        return response.answers[0].answer
    return "Sorry, I couldn't find an answer to that question."

@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer = ask_question_to_azure(question)
            # Optionally, speak the answer using Azure Text-to-Speech
            speak_prediction_lively(answer)
            return render_template('answer.html', question=question, answer=answer)
    
    return render_template('ask_question.html')



with app.app_context():
    db.create_all()
"""    
# Start ngrok
    public_url = ngrok.connect(5000)
    print(f" * Tunnel URL: {public_url}")
"""

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True)
