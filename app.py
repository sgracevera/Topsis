import pandas as pd
import numpy as np
import sys

def cp():
    if len(sys.argv) != 5:
        print("Usage: python <prog.py> <InFile> <W> <I> <RFile>")
        sys.exit(1)

def rid(inf):
    try:
        d = pd.read_csv(inf)
        return d
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)

def vid(d):
    if len(d.columns) < 3:
        print("Input file must contain three or more columns.")
        sys.exit(1)

    for c in d.columns[1:]:
        if not pd.api.types.is_numeric_dtype(d[c]):
            print("Columns from 2nd to last must contain numeric values only.")
            sys.exit(1)

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

def m():
    cp()

    inf = sys.argv[1]
    w = np.array(list(map(float, sys.argv[2].split(','))))
    i = np.array([1 if j == '+' else -1 for j in sys.argv[3].split(',')])
    rf = sys.argv[4]

    d = rid(inf)
    vid(d)

    if len(w) != len(i) or len(w) != len(d.columns) - 1:
        print("Number of weights, impacts, and columns must be the same.")
        sys.exit(1)

    for j in i:
        if j not in [1, -1]:
            print("Impacts must be either +ve or -ve.")
            sys.exit(1)

    pt(d, w, i)
    sr(d, rf)

if __name__ == "__main__":
    m()
