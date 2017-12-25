from __future__ import print_function
import fitsio
import numpy as np
from numpy import random
import glob
from os.path import dirname

from desitarget import desi_mask

llmin = np.log10(3600)
llmax = np.log10(10000)
dll = 1e-3

nbins = int((llmax-llmin)/dll)
nmasked_max = nbins/10
wave = 10**(llmin + np.arange(nbins)*dll)

def read_spcframe(b_spcframe,r_spcframe,pf2tid):
    data = []
    tids = []

    hb = fitsio.FITS(b_spcframe)
    hr = fitsio.FITS(r_spcframe)
    target_bits = hb[5]["BOSS_TARGET1"][:]
    w = np.zeros(len(target_bits),dtype=bool)
    mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
    for i in mask:
        w = w | (target_bits & 2**i)
    w = w>0
    print("INFO: found {} quasars in file {}".format(w.sum(),b_spcframe))

    plate = hb[0].read_header()["PLATEID"]
    fid = hb[5]["FIBERID"][:]
    fl = np.hstack((hb[0].read(),hr[0].read()))
    iv = np.hstack((hb[1].read()*(hb[2].read()==0),hr[1].read()*(hr[2].read()==0)))
    ll = np.hstack((hb[3].read(),hr[3].read()))

    fid = fid[w]
    fl = fl[w,:]
    iv = iv[w,:]
    ll = ll[w,:]

    for i in range(fl.shape[0]):
        if (plate,fid[i]) in pf2tid:
            t = pf2tid[(plate,fid[i])]
        else:
            print("DEBUG: ({},{}) not found in spall".format(plate,fid[i]))
            continue

        fl_aux = np.zeros(nbins)
        iv_aux = np.zeros(nbins)
        bins = ((ll[i]-llmin)/dll).astype(int)
        wbin = (bins>=0) & (bins<nbins) & (iv[i]>0)
        bins=bins[wbin]
        c = np.bincount(bins,weights=fl[i,wbin]*iv[i,wbin])
        fl_aux[:len(c)]=+c
        c = np.bincount(bins,weights=iv[i,wbin])
        iv_aux[:len(c)]=+c
        nmasked = (iv_aux==0).sum()
        if nmasked >= nmasked_max :
            print("INFO: skipping specrum {} with too many masked pixels {}".format(t,nmasked))
            continue
        data.append(np.hstack((fl_aux,iv_aux)))
        tids.append(t)

        assert ~np.isnan(fl_aux,iv_aux).any()

    if len(data)==0:
        return
    data = np.vstack(data).T
    assert ~np.isnan(data).any()
    ## now normalize coadded fluxes
    norm = data[nbins:,:]*1.
    w = norm==0
    norm[w] = 1.
    data[:nbins,:]/=norm

    assert ~np.isnan(data).any()

    return tids,data

def read_spall(spall):
    spall = fitsio.FITS(spall)
    plate=spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fid = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:]
    specprim=spall[1]["SPECPRIMARY"][:]

    #pf2tid = {(p,m,f):t for p,m,f,t,s in zip(plate,mjd,fid,tid,specprim) if s==1}
    pf2tid = {(p,m,f):t for p,m,f,t,s in zip(plate,mjd,fid,tid,specprim)}
    spall.close()
    return pf2tid

def read_drq_superset(drq_sup,high_z = 2.1):
    ##from https://arxiv.org/pdf/1311.4870.pdf
    ##      only return targets with z_conf_person == 3:
    ##      class person: 1 (Star), 3 (QSO), 4 (Galaxy), 30 (QSO_BAL)
    ##      my class_person: 1 (Star), 3 (QSO and z < hz), 4 (Galaxy), 5 (QSO and z>=hz), 30 (QSO_BAL)

    drq = fitsio.FITS(drq_sup)
    qso_thids = drq[1]["THING_ID"][:]
    class_person = drq[1]["CLASS_PERSON"][:]
    z_conf = drq[1]["Z_CONF_PERSON"][:]
    z = drq[1]["Z_VI"][:]

    ## select objects with good classification
    w = (qso_thids > 0) & (z_conf==3)
    qso_thids = qso_thids[w]
    class_person = class_person[w]
    my_class_person = class_person*1
    z = z[w]

    ## STARS
    w = class_person == 1
    my_class_person[w] = 0

    ## GALAXIES
    w = class_person == 4
    my_class_person[w] = 1

    ## QSO_LOWZ, include BAL
    w = ((class_person==3) | (class_person == 30)) & (z<high_z)
    my_class_person[w] = 2

    ## QSO_HIGHZ, include BAL
    w = ((class_person==3) | (class_person == 30)) & (z>=high_z)
    my_class_person[w] = 3

    drq_classes = ["STAR","GALAXY","QSO_LOWZ","QSO_HIGHZ","BAL"]
    Y = np.zeros((len(class_person),len(drq_classes)))
    for i in range(Y.shape[0]):
        Y[i,my_class_person[i]]=1

    ## add BAL flag
    w = class_person == 30
    Y[w,drq_classes.index("BAL")]=1

    ## 
    target_class = {tid:y for tid,y in zip(qso_thids,Y)}
    target_z = {tid:y for tid,y in zip(qso_thids,z)}
    
    drq.close()

    return target_class, target_z

def read_plates(plates,pf2tid,nplates=None):
    data = []
    read_plates = 0
    tids = []

    for p in plates:
        h=fitsio.FITS(p)
        head = h[0].read_header()
        exps=[]

        ## read b,r exposures
        try:
            nexp_b = head["NEXP_B1"]+head["NEXP_B2"]
        except:
            continue
        if nexp_b>99:
            nexp_b=99
        for exp in range(nexp_b):
            str_exp = str(exp+1)
            if exp<9:
                str_exp = '0'+str_exp
            exp_b = head["EXPID{}".format(str_exp)][:11]
            exp_r = exp_b.replace("b", "r")
            exps.append((exp_b, exp_r))
         
        for exp_b, exp_r in exps:
            spcframe_b = dirname(p)+"/spCFrame-{}.fits".format(exp_b)
            spcframe_r = dirname(p)+"/spCFrame-{}.fits".format(exp_r)
            res = read_spcframe(spcframe_b, spcframe_r, pf2tid)
            if res is not None:
                plate_tid,plate_data = res
                data.append(plate_data)
                tids = tids + plate_tid

        if nplates is not None:
            if len(data)//2==nplates:
                break

    data = np.hstack(data)

    return tids, data

def export(fout,tids,data):
    h = fitsio.FITS(fout,"rw",clobber=True)
    h.write(data,extname="DATA")
    tids = np.array(tids)
    h.write([tids],names=["TARGETID"],extname="METADATA")
    h.close()


def read_desi_spectra(fin):
    h=fitsio.FITS(fin)
    nbins = int((llmax-llmin)/dll)
    tids = h[1]["TARGETID"][:]
    nspec = len(tids)
    fl = np.zeros((nspec, nbins))
    iv = np.zeros((nspec, nbins))
    if nspec == 0: return None
    for band in ["B", "R", "Z"]:
        wave = h["{}_WAVELENGTH".format(band)].read()
        w = (np.log10(wave)>llmin) & (np.log10(wave)<llmax)
        wave = wave[w]
        bins = np.floor((np.log10(wave)-llmin)/dll).astype(int)
        fl_aux = h["{}_FLUX".format(band)].read()[:,w]
        iv_aux = h["{}_IVAR".format(band)].read()[:,w]
        for i in range(nspec):
            c = np.bincount(bins, weights=fl_aux[i]*iv_aux[i])
            fl[i,:len(c)] += c
            c = np.bincount(bins, weights = iv_aux[i])
            iv[i,:len(c)]+=c

    w = iv>0
    fl[w]/=iv[w]
    fl = np.hstack((fl,iv))

    ## select QSO targets:
    wqso = h[1]['DESI_TARGET'][:] & desi_mask.mask('QSO')
    wqso = wqso>0
    print("INFO: founds {} qso targets".format(wqso.sum()))
    fl = fl[wqso,:]
    tids = tids[wqso]
    return tids, fl


def read_spplate(fin, pf2tid):
    h=fitsio.FITS(fin)
    head = h[0].read_header()
    c0 = head["COEFF0"]
    c1 = head["COEFF1"]
    p = head["PLATEID"]
    m = head["MJD"]
    
    fids = h[5]["FIBERID"][:]
    wqso = np.zeros(len(fids), dtype=bool)
    mask = [10,11,12,13,14,15,16,17,18,19,40,41,42,43,44]
    target_bits = h[5]["BOSS_TARGET1"][:]
    for i in mask:
        wqso = wqso | (target_bits & 2**i)

    ## SEQUELS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET0"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass

    ## EBOSS
    try:
        mask = [10, 11 ,12 ,13, 14, 15, 16, 17, 18]
        target_bits = h[5]["EBOSS_TARGET1"][:]
        for i in mask:
            wqso = wqso | (target_bits & 2**i)
    except:
        pass
    wqso = wqso>0

    print("INFO: found {} quasars in file {}".format(wqso.sum(), fin))
    if wqso.sum()==None:
        return

    fids = fids[wqso]
    try:
        tids = np.array([pf2tid[(p, m, f)] for f in fids])
    except:
        return None

    nspec = len(tids)
    nbins = int((llmax-llmin)/dll)
    fl = np.zeros((nspec, nbins)) 
    iv = np.zeros((nspec, nbins))
    nbins = fl.shape[1]

    fl_aux = h[0].read()[wqso,:]
    iv_aux = h[1].read()[wqso,:]
    wave = 10**(c0 + c1*np.arange(fl_aux.shape[1]))
    bins = np.floor((np.log10(wave)-llmin)/dll).astype(int)
    w = (bins>=0) & (bins<nbins)
    bins = bins[w]

    fl_aux=fl_aux[:,w]
    iv_aux=iv_aux[:,w]
    for i in range(nspec):
        c = np.bincount(bins, weights=fl_aux[i]*iv_aux[i])
        fl[i,:len(c)] += c
        c = np.bincount(bins, weights = iv_aux[i])
        iv[i,:len(c)]+=c

    w = iv>0
    fl[w]/=iv[w]
    fl = np.hstack((fl,iv))
    print(fl.shape)
    wbad = iv==0
    w=wbad.sum(axis=1)>10
    fl=fl[~w,:]
    tids=tids[~w]
    return tids, fl


