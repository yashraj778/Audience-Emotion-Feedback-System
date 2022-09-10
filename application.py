from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# We are using the relative path so you dont need to put this model_folder object.
# models_folder = '/home/xyz/Downloads/concert_feedback_classification/'

app = Flask(__name__)

class ReviewForm(Form):
    feedback = TextAreaField('',[validators.DataRequired(),validators.length(min=4)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('/reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():

        # reading the text feedback from the user
        review = request.form['feedback']
        print(review)

        # reading the questions response from the user
        q1 = int(request.form["q1"])
        q2 = int(request.form["q2"])
        q3 = int(request.form["q3"])
        q4 = int(request.form["q4"])
        q5 = int(request.form["q5"])

        print('Received responses:', q1, q2, q3, q4, q5)

        # creating binary features from the user responses 
        
        
        features = [q1 >=3, q2 >=3, q3 >=3, q4 >=3, q5 >= 3 ]
        print('features are:', features)

        # updating the database with the user feedback data
        path = os.path.join(os.getcwd(), 'feedback_data.csv')
        if not os.path.exists(path):
            df = pd.DataFrame( {'Q1': [q1], 'Q2':[q2], 'Q3': [q3], 'Q4':[q4], 'Review': [review]})
            df.to_csv(path, index = False)
        else:
            data = pd.read_csv(path)
            data = data.append({'Q1': q1, 'Q2':q2, 'Q3': q3, 'Q4':q4, 'Review': review}, ignore_index = True)
            data.to_csv(path, index = False)


        # prediction using QA logistic model
       

        qa_model_file = './logistic_reg_qa.pkl'
        logistic_model_loaded = pickle.load(open(qa_model_file, "rb"))
        inp_qa = np.array([features]) * 1
        y = logistic_model_loaded.predict(inp_qa)[0]
        
        if y == 0:
        	pred_qa = 'Negative'
        	y_qa_proba = logistic_model_loaded.predict_proba(inp_qa)[0,0]
        else:
        	pred_qa = 'Positive'
        	y_qa_proba = logistic_model_loaded.predict_proba(inp_qa)[0,1]


        # prediction using text feedback based SGD classifier model 
        text_model_file ='./finalized_model.sav'
        loaded_model = pickle.load(open(text_model_file, 'rb'))

        # load vectorizer for creating features from the text data
        vector_data_file = './vector_data.pkl'
        vectorizer = pickle.load(open( vector_data_file, 'rb' ))
        inp = vectorizer.transform([review])
        y_text = loaded_model.predict(inp)[0]
        
        if y_text == -1:
        	pred_text = 'Negative'
        	y_text_proba = loaded_model.predict_proba(inp)[0, 0]
        else:
        	pred_text = 'Positive'
        	y_text_proba = loaded_model.predict_proba(inp)[0, 1]


       	# overall feedback prediction using combination of the QA and text data
       	if y == 0 and y_text == -1:
       		pred_final = 'Negative'
       	elif y == 1 and y_text == 1:
       		pred_final = 'Positive'
       	else:
       		pred_final = 'Neutral'

        return render_template('results.html',content=review,prediction=pred_qa,probability=round(y_qa_proba*100, 2), prediction_t=pred_text,probability_t=round(y_text_proba*100, 2), prediction_final = pred_final)
    return render_template('reviewform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
