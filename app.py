from flask import Flask, render_template, request, redirect, url_for
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index(): 
    if request.method == "POST":
        # Get the data from the form
        name = request.form['name']
        job_pos = request.form['job']
        resume_file = request.files['resume']
        reader = PyPDF2.PdfReader(resume_file)
        resume_text=""
        for page in reader.pages:
            resume_text+=page.extract_text() 
        texts = [resume_text, job_pos]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_score = round(score * 100, 2)
        print(f"Match Score: {match_score}%")

        print(f"Name: {name}")
        print(f"Job: {job_pos}")
        print(f"Resume File: {resume_file.filename}")
        print(resume_text)
        print(f"Match Score: {match_score}%")
        return render_template("index.html", result=match_score)
        pass
    return render_template('index.html', result=None)

if __name__ == "__main__":
    app.run(debug=True)