from flask import Flask, render_template, request
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    cv = joblib.load('cv_Vocabulary.pkl', 'rb')
    nb_spam_model = open('NB_spam_model.pkl', 'rb')
    clf = joblib.load(nb_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)