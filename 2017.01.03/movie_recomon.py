import pandas as pd
import numpy as np
from flask import Flask, request, flash, redirect, url_for, render_template
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html", comment="Hello World!")

@app.route("/recomon", method=['GET', 'POST'])
def recomon():
    data = pd.read_csv('train_recomon.csv')

    if (request.form['category'] == 1): #itembased
        M = data.pivot_table(index=['user'], columns=['movie'], values='rating')
    elif (request.form['category'] == 2): #userbased
        M = data.pivot_table(index=['movie'], columns=['user'], values='rating')
    else:
        flash("추천 방법을 다시 선택해주세요.")
        return redirect(url_for('hello'))

    return render_template("movie_recomon.html", recommon_movie = get_recs(request.form['movie_name'], M, 1))


def pearson(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s1_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))


def get_recs(movie_name, M, num):
    reviews = []
    for title in M.columns:
        if title == movie_name:
            continue
        cor = pearson(M[movie_name], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
            print

    reviews.sort(key=lambda tup: tup[1], reverse=True)
    return reviews[:num]


if __name__ == "__main__":
    app.run()

