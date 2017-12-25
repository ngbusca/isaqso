#!/usr/bin/env python

from os.path import dirname
import argparse
import numpy as np
import fitsio
from IPython.display import SVG
import scipy.misc
from . import models
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.models import load_model

parser = argparse.ArgumentParser()

def read_sdrq(sdrq):
    sdrq = fitsio.FITS(sdrq)
    qso_thids = sdrq[1]["THING_ID"][:]
    class_person = sdrq[1]["CLASS_PERSON"][:]
    z_conf = sdrq[1]["Z_CONF_PERSON"][:]
    z_vi = sdrq[1]["Z_VI"][:]
    z_vi = sdrq[1]["Z_VI"][:]

    sdrq = {t:(c,zc,z) for t,c,zc,z in zip(qso_thids, class_person, z_conf, z_vi)}

    return sdrq

def read_data(fi, sdrq):
    h=fitsio.FITS(fi)
    tids = h[1]["TARGETID"][:]
    Xtmp = h[0].read()
    assert Xtmp.shape[0]==tids.shape[0]

    mdata = np.average(Xtmp[:,:443], weights = Xtmp[:,443:], axis=1)
    sdata = np.average((Xtmp[:,:443]-mdata[:,None])**2, weights = Xtmp[:,443:], axis=1)
    sdata=np.sqrt(sdata)
    Xtmp=Xtmp[:,:443]-mdata[:,None]
    Xtmp/=sdata[:,None]
    X = np.zeros(Xtmp.shape)
    X[:] = Xtmp
    del Xtmp

    ## classes: 0 = STAR, 1=GALAXY, 2=QSO_LZ, 3=QSO_HZ, 4=BAD (zconf != 3)
    classes = 5
    Y = np.zeros((X.shape[0],classes))
    z = np.zeros(X.shape[0])
    bal = np.zeros(X.shape[0])
    notfound=0

    for i,t in enumerate(tids):
        if t not in sdrq:
            notfound+=1
            continue
        c_aux,zc_aux,z_aux = sdrq[t]
        j=4
        if zc_aux==3:
            if c_aux==1:
                j=0
            elif c_aux==3 or c_aux==30:
                z[i]=z_aux
                if c_aux==30:
                    bal[i]=1

                if z_aux>2.1:
                    j=3
                else:
                    j=2
            elif c_aux==4:
                j=1
            else:
                print("can't find class", zc_aux)
                stop

        Y[i,j] = 1
    print("not found: ",notfound)
    w = Y.sum(axis=1)==1
    Y=Y[w,:]
    X=X[w,:]
    z = z[w]
    bal = bal[w]
    tids=tids[w]

    return tids,X,Y,z,bal

def read_desi_truth(fin):
    h=fitsio.FITS(fin)
    truth = {}
    for t,c,z in zip(h[1]["TARGETID"][:], h[1]["TRUESPECTYPE"][:], h[1]["TRUEZ"][:]):
        c = c.strip()
        if c=="QSO":
            c=3
        elif c=="GALAXY":
            c=4
        elif c=="STAR":
            c=1
        assert isinstance(c,int)
        truth[t] = (c,3,z)
    return truth

