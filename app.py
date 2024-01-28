import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import sys

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rid(inf):
    try:
        d = pd.read_csv(inf)
        return d
    except FileNotFoundError:
        return None

def vid(d):
    if len(d.columns) < 3:
        return False

    for c in d.columns[1:]:
        if not pd.api.types.is_numeric_dtype(d[c]):
            return False

    return True

def nm(m):
    nm = m / np.linalg.norm(m, axis=0)
    return nm

def wnm(nm, w):
    wnm = nm * w
    return wnm

def ibw(wnm, i):
    ib = np.max(wnm, axis=0)
    iw = np.min(wnm, axis=0)

    ib = ib * i
    iw = iw * i

    return ib, iw

def cd(wnm, ib, iw):
    db = np.linalg.norm(wnm - ib, axis=1)
    dw = np.linalg.norm(wnm - iw, axis=1)

    return db, dw

def ts(db, dw):
    ts = dw / (db + dw)
    return ts

def tr(ts):
    r = np.argsort(ts)[::-1] + 1
    return r

def sr(d, rf):
    d.to_csv(rf, index=False)

def pt(d, w, i):
    m = d.iloc[:, 1:].values
    n = nm(m)
    wn = wnm(n, w)
    ib, iw = ibw(wn, i)
    db, dw = cd(wn, ib, iw)
    s = ts(db, dw)
    d['TS'] = s
    d['R'] = tr(s)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            inf_file = request.files['inf']
            w = np.array(list(map(float, request.form['w'].split(','))))
            i = np.array([1 if j == '+' else -1 for j in request.form['i'].split(',')])
            rf = request.form['rf']

            if inf_file and allowed_file(inf_file.filename):
                inf_filename = os.path.join(app.config['UPLOAD_FOLDER'], inf_file.filename)
                inf_file.save(inf_filename)

                d = rid(inf_filename)

                if d is None:
                    return render_error("File not found.")

                if not vid(d):
                    return render_error("Input file must contain three or more columns, and numeric values only.")

                if len(w) != len(i) or len(w) != len(d.columns) - 1:
                    return render_error("Number of weights, impacts, and columns must be the same.")

                for j in i:
                    if j not in [1, -1]:
                        return render_error("Impacts must be either +ve or -ve.")

                pt(d, w, i)
                sr(d, rf)

                return render_success(d)

            return render_error("Invalid file format. Please upload a CSV file.")

        except Exception as e:
            return render_error(str(e))

    return render_template('index.html')

def render_error(error_message):
    return render_template('error.html', error_message=error_message)

def render_success(result_df):
    return render_template('success.html', result_table=result_df.to_html(index=False))

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
