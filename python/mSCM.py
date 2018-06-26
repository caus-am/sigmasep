# Copyright (c) 2018  Patrick Forré, Joris M. Mooij
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
from os.path import basename, dirname, splitext
import re
import time
import numpy as np
import scipy.stats
import pandas as pd
import random
from numpy.random import exponential
from numpy.random import choice
from numpy.random import binomial
from numpy.random import normal
from numpy.random import uniform
from numpy.linalg import norm
from itertools import product
import subprocess

#from scipy.stats import norm as qnorm
#from scipy.stats import rankdata
#import tigramite as tg
#from tigramite.independence_tests import ParCorr, CMIknn #, CMIsymb

#import matplotlib.pyplot as plt
#import networkx as nx
##%matplotlib inline

def add_independent_noise_nodes(A,s=-1):
    d = A.shape[0]
    if s<=0:
        s=d
    s = min(d,s)
    dg = [1]*s+[0]*(d-s)
    B = np.concatenate((A,np.diag(dg)),axis=1)
    return B

# d = number of observed variables, k = number of latent confounder
def sample_adjacency_matrix(d=6,k=4,p=0.3,add_indep_noise=False):
    a = binomial(n=1,p=p,size=(d,d+k))
    if add_indep_noise==False:
        v = np.zeros(d+k)
        v[0] = 1
        for s in range(d):
            c = a[s,:]
            while np.count_nonzero(c) < 1:
                np.random.shuffle(v)
                c= c + v
                c = np.minimum(c,1)
            a[s,:] = c
    v = np.zeros(d)
    v[0] = 1
    for s in range(d,d+k):
        c = a[:,s]
        while np.count_nonzero(c) < 2:
            np.random.shuffle(v)
            c= c + v
            c = np.minimum(c,1)
        a[:,s] = c
    if add_indep_noise==True:
        a = add_independent_noise_nodes(a)
    np.fill_diagonal(a,0)
    return a

def extract_edges(A=np.array([[0,1,1],[0,0,1]])):
    d = A.shape[0]
    return A[:,:d]

def extract_confs(A=np.array([[0,1,1],[0,0,1]])):
    d = A.shape[0]
    k = A.shape[1]
    B = np.zeros(shape=(d,d))
    for i in range(d,k):
        c = A[:,i]
        for j in range(d):
            for k in range(d):
                if c[j]==1 and c[k]==1:
                    B[j,k]=1
    np.fill_diagonal(B,0)
    return B


# From here assume d <= k
# m = about the number of hidden units to add per parent node (excluding rounding error, see below)

def num_hidden_units_to_add(A,m=2):
    return (2*np.ceil(0.5*m*A.sum(axis=1))).astype(int)


# add the hidden units of a neural network to the graph in between every node and its parents
def extend_adjacency_matrix(A=np.array([[0,1],[0,0]]),n_hid_units=np.array([]),add_indep_noise=True):
    d = A.shape[0]
    k = A.shape[1]
    c = n_hid_units
    if n_hid_units.size == 0:
        c = np.zeros(d)
    l = c.sum().astype(int)
    E00 = np.zeros((d,k))
    E10 = np.empty(shape=(0,k))
    for i in range(d):
        v = A[i,].reshape((1,k))
        for j in range(c[i].astype(int)):
            E10 = np.concatenate((E10,v),axis=0)
    EE0 = np.concatenate((E00,E10),axis=0)
    E11 = np.zeros((l,l))
    E01 = np.empty(shape=(d,0))
    for i in range(d):
        v = np.zeros((d,1))
        v[i,0] = 1
        for j in range(c[i].astype(int)):
            E01 = np.concatenate((E01,v),axis=1)
    EE1 = np.concatenate((E01,E11),axis=0)
    EE = np.concatenate((EE0[:,:d],EE1,EE0[:,d:k]),axis=1)
    if add_indep_noise==True:
        EE = add_independent_noise_nodes(EE,d)
    return EE


# nx can only use square adjacency matrices and encodes adjacency matrix transpose to our convention:

def make_square_pad_zeros(A):
    d = A.shape[0]
    k = A.shape[1]
    dk = max(d,k)
    return np.pad(A, [(0, dk-d), (0, dk-k)], mode='constant', constant_values=0)

def make_square_pad_eye(A):
    d = A.shape[0]
    k = A.shape[1]
    dk = max(d,k)
    B = A - np.eye(d,k)
    C = np.pad(B, [(0, dk-d), (0, dk-k)], mode='constant', constant_values=0)
    return C+np.eye(dk,dk)

#def to_nx(A):
#    return make_square_pad_zeros(A).transpose()
#
#def draw_graphs(A,kk=0.15):
#    d = A.shape[0]
#    dk = A.shape[1]
#    k = dk - d
#    V = extract_edges(A)
#    H = extract_confs(A)
#    GA = nx.from_numpy_matrix(to_nx(A),create_using=nx.MultiDiGraph())
#    GV = nx.from_numpy_matrix(to_nx(V),create_using=nx.MultiDiGraph())
#    GH = nx.from_numpy_matrix(to_nx(H),create_using=nx.MultiDiGraph())
#    posA = nx.spring_layout(GA,k=kk)
#    posObs = {t: posA[t] for t in list(posA)[0:d]}
#    posExt = {t: posA[t] for t in list(posA)[d:dk]}
#    colObs = ['yellow']*d
#    colExt = ['cyan']*k
#    colA = colObs+colExt
#    q = plt.subplot(1, 2, 1)
#    plt.axis('off')
#    nx.draw_networkx(GA,pos=posA, node_color=colA,edge_color='blue')
#    plt.subplot(1, 2, 2, sharex=q, sharey=q)
#    plt.axis('off')
#    nx.draw_networkx(GV,pos=posObs, node_color=colObs,edge_color='blue')  #blue: directed edges (blue bidirected edges = directed edges in both direction)
#    nx.draw_networkx(GH,pos=posObs, node_color=colObs,edge_color='red')   #red: bidirected edges
#    return posA


#def draw_graphs_extended(A,kk=0.15, m=2):
#    d = A.shape[0]
#    k = A.shape[1]
#    V = extract_edges(A)
#    H = extract_confs(A)
#    c = num_hidden_units_to_add(A,m)
#    l = c.sum().astype(int)
#    B = extend_adjacency_matrix(A,c)
#    GA = nx.from_numpy_matrix(to_nx(A),create_using=nx.MultiDiGraph())
#   GV = nx.from_numpy_matrix(to_nx(V),create_using=nx.MultiDiGraph())
#    GH = nx.from_numpy_matrix(to_nx(H),create_using=nx.MultiDiGraph())
#    GB = nx.from_numpy_matrix(to_nx(B),create_using=nx.MultiDiGraph())
#    posB = nx.spring_layout(GB,k=kk)
#    #posB = nx.spectral_layout(GB)
#    posObs = {t: posB[t] for t in list(posB)[0:d]}
#    posHid = {t: posB[t] for t in list(posB)[d:l+d]}
#    posExt = {t: posB[t] for t in list(posB)[l+d:l+d+k]}
#    posA = {**posObs, **posExt}
#    mapping = {GA.nodes()[s]:list(posA)[s] for s in range(len(list(posA)))}
#    GA=nx.relabel_nodes(GA,mapping,copy=False)
#    colObs = ['yellow']*d
#    colHid = ['pink']*l
#    colExt = ['cyan']*k
#    colB = colObs+colHid+colExt
#    colA = colObs+colExt
#    q = plt.subplot(1, 3, 1)
#    plt.axis('off')
#    nx.draw_networkx(GB,pos=posB, node_color=colB,edge_color='blue')
#    q = plt.subplot(1, 3, 2, sharex=q, sharey=q)
#    plt.axis('off')
#    nx.draw_networkx(GA,pos=posA, node_color=colA,edge_color='blue')
#    plt.subplot(1, 3, 3, sharex=q, sharey=q)
#    plt.axis('off')
#    nx.draw_networkx(GV,pos=posObs, node_color=colObs,edge_color='blue')  #blue: directed edges (blue bidirected edges = directed edges in both direction)
#    nx.draw_networkx(GH,pos=posObs, node_color=colObs,edge_color='red')   #red: bidirected edges
#    return A

#def draw_graphs_from_dir(filepath,kk=0.15):
#    V0 = np.loadtxt(filepath+"edges_true.csv",delimiter=',')
#    H0 = np.loadtxt(filepath+"confs_true.csv",delimiter=',')
#    G0 = nx.from_numpy_matrix(to_nx(V0),create_using=nx.MultiDiGraph())
#    C0 = nx.from_numpy_matrix(to_nx(H0),create_using=nx.MultiDiGraph())
#    pos0 = nx.spring_layout(G0,k=kk)
#    Gs =[G0]
#    Cs =[C0]
#    files = [file for file in os.listdir(filepath) if re.match('^edges_pred_.*\.csv$',file)]
#    for file in files:
#        V = np.loadtxt(filepath+file,delimiter=',')
#        G = nx.from_numpy_matrix(to_nx(V),create_using=nx.MultiDiGraph())
#        Gs += [G]
#        conf_file = 'confs'+file[5:]
#        H = np.loadtxt(filepath+conf_file,delimiter=',')
#        C = nx.from_numpy_matrix(to_nx(H),create_using=nx.MultiDiGraph())
#        Cs += [C]
#    npl = len(Gs)
#    q = plt.subplot(1, npl, 1)
#    for i in range(npl):
#        plt.subplot(1, npl, i+1, sharex=q, sharey=q)
#        plt.axis('off')
#        nx.draw_networkx(Gs[i],pos=pos0, node_color='yellow',edge_color='blue')  #blue: directed edges (blue bidirected edges = directed edges in both direction)
#        nx.draw_networkx(Cs[i],pos=pos0, node_color='yellow',edge_color='red')
#    return None

#def sample_and_draw_graphs(d=5,k=3,p=0.3,m=2,kk=0.15,add_indep_noise=False): #higher kk forces nodes to be more distant
#    A = sample_adjacency_matrix(d,k,p,add_indep_noise)
#    draw_graphs_extended(A=A,kk=kk,m=m)
#    return A


# d = number of observed nodes, dk = number of observed plus latent nodes, A = adjacency matrix d x dk

def sample_L1_uniform_weights(A=np.ones((2, 2)), include_latent=True):
    d = A.shape[0]
    dk = A.shape[1]
    if include_latent == True:
        dd = dk
    else:
        dd = d
    B = A[:d, :dd]
    C = A[:d, dd:dk]
    a = choice([-1, 1], size=(d, dd))
    b = exponential(scale=1, size=(d, dd)) * B
    c = exponential(scale=1, size=(d, 1)) + b.sum(axis=1, keepdims=True)
    D = (a * b) / c
    W = np.concatenate((D, C), axis=1)
    return W


def sample_weights_and_bias(A=np.array([[0, 1], [0, 0]]), n_hid_units=np.array([]), add_indep_noise=True,
                            include_latent=True):
    if sum(n_hid_units) == 0:
        G = A
    else:
        G = extend_adjacency_matrix(A, n_hid_units, add_indep_noise)
    W = sample_L1_uniform_weights(G, include_latent)
    d = W.shape[0]
    q = normal(-0.5, 0.2, size=(d, 1))
    return (W, q)


def load_weights_and_bias(filepath):
    W = np.loadtxt(filepath + "weights_bias.csv", delimiter=',')
    d = W.shape[0]
    k = W.shape[1]
    q = W[:, k - 1].reshape((d, 1))
    W = W[:, :k - 1].reshape((d, k - 1))
    return (W, q)


# activation function: scaled hyperbolic tangent
def tanh(x):
    return np.tanh(x)


def softplus(x):
    return np.log(1 + np.exp(x))


def silu(x):
    return 0.9 * x / (1 + np.exp(-x))


def lrelu(x):
    return np.maximum(x, (0.3 * x))


# x_ext = external conditional value,
# do_target = intervention target indices,
# do_values = corresponding interventon values
# e = acceptable error, sc = activation function scale
def iterate_mSCM(weights, bias, x_ext=np.array([]), do_targets=np.array([]), do_values=np.array([]),
                 actfct=np.tanh, sc=1, e=0.0000001):
    d = weights.shape[0]
    dk = weights.shape[1]
    k = dk - d
    # W = make_square_pad_zeros(weights)
    W_int = weights[:d, :d]
    W_ext = weights[:d, d:dk]
    # b = np.zeros(shape=(d,1))
    b = bias.reshape((d, 1))
    # t = np.ones(shape=(dk,1))
    non_do = np.ones(shape=(d, 1))
    for j in range(len(do_targets)):
        non_do[do_targets[j], :] = 0
    # non_do[d:dk,:] = np.zeros(shape=(k,1))
    # z = np.zeros(shape=(dk,1))
    do_x = np.zeros(shape=(d, 1))
    for j in range(len(do_targets)):
        do_x[do_targets[j], :] = do_values[j]
    # xd = np.zeros(shape(d,1))
    # for j in range(len(do_targets)):
    #    xd[do_targets[j],:] = do_values[j]
    # z[d:dk,:] = x_ext.reshape((k,1))
    x_ext = x_ext.reshape((k, 1))
    # do_x = (1-non_do)*do_x
    # x0 = normal(0,1,size=(dk,1)) #initialization
    x = normal(0, 1, size=(d, 1))  # initialization
    x = do_x + non_do * x
    y = do_x + non_do * (actfct(sc * (np.dot(W_int, x) + b)) / sc + np.dot(W_ext, x_ext))
    while norm(x - y, ord=np.inf) > e:
        x = y
        y = do_x + non_do * (actfct(sc * (np.dot(W_int, x) + b)) / sc + np.dot(W_ext, x_ext))
    return y


def sample_from_mSCM(weights, bias, do_targets=np.array([]), n=1, d=1, actfct=np.tanh, sc=1, sd=1, noise='normal',
                     e=0.0000001):  # ,do_values=np.array([])
    W = weights
    t = W.shape[0]
    tk = W.shape[1]
    S = np.empty(shape=(0, d))
    l = do_targets.size
    for j in range(n):
        if noise == 'uniform':
            x_ext = uniform(-1, 1, size=(tk - t, 1)) * np.sqrt(3) * sd
            do_values = uniform(-1, 1, size=l) * np.sqrt(3) * sd
        if noise == 'normal':
            x_ext = normal(0, sd, size=(tk - t, 1))
            do_values = normal(0, sd, size=l)
        x = iterate_mSCM(weights, bias, x_ext, do_targets, do_values, actfct, sc, e)
        S = np.concatenate((S, x[:d, :].transpose()), axis=0)
    return S


def name_suffix(actfctn='tanh',sc=1,noise='normal',sd=1,do_targets=np.array([]),nbr_do=0,max_do=0):
    #if do_all:
    #    t = 'all'
    #else:
    #    t = '['+';'.join([str(s) for s in do_targets])+']'
    #t = str(nbr_do)
    s = '_'+actfctn+'_sc_'+str(sc)+'_'+noise+'_sd_'+str(sd)+'_do_'+str(nbr_do)+'_'+str(max_do)+'_.csv'
    return s

def str_to_list(s):
    return list({int(v) for v in s[1:-1].split(';') if v is not ''})


# def param_from_name(filename):
#    s = filename.split('_')
#    actfctn = s[-9]
#    sc = int(s[-7])
#    noise = s[-6]
#    sd = int(s[-4])
#    do = s[-2]
#    return (actfctn, sc, noise, sd, do)


def save_samples_from_mSCM(weights, bias, filepath='', do_targets=np.array([]), n=1, d=1, actfct=np.tanh, sc=1, sd=1,
                           noise='normal', e=0.0000001):
    S = sample_from_mSCM(weights, bias, do_targets, n=n, d=d, actfct=actfct, sc=sc, sd=sd, noise=noise, e=e)
    # do = ""
    # for nb in do_targets:
    #    do += str(nb)
    filename = 'samples' + name_suffix(actfct.__name__, sc, noise, sd, do_targets)
    # _'+actfct.__name__+'_sc['+str(sc)+']_'+noise+'_sd['+str(sd)+']_do['+do+'].csv'
    np.savetxt(filepath + filename, S, delimiter=",")
    print('Saved samples to: ', filename)
    return filename


def powerlist(lst):
    pwlst = []
    d = len(lst)
    for i in range(2 ** d):
        subset = [x for j, x in enumerate(lst) if (i >> j) & 1]
        pwlst.append(subset)
    return pwlst


def save_samples_from_mSCM_all_interventions(weights, bias, filepath, do_targets=np.array([]),
                                             n=1, d=1, actfct=np.tanh, sc=1, sd=1, noise='normal', e=0.0000001):
    for do_targets in powerlist(range(d)):
        save_samples_from_mSCM(weights, bias, filepath, np.array(do_targets), n, d, actfct, sc, sd, noise, e)
    return None


def all_xyZ_partitions_list(lst):
    lst = set(lst)
    all_xyZ = []
    for i, x in enumerate(lst):
        for j, y in enumerate(lst):
            if j > i:
                ll = list(lst.difference({x, y}))
                for Z in powerlist(ll):
                    all_xyZ.append(([x], [y], Z))
    return list(all_xyZ)

#############

############# normal rank transform partial correlation test ##########################

def get_residuals(df, id_y, id_z):
    n = df.shape[0]
    Y = df[:, id_y]
    Y = Y - Y.mean(axis=0)  # .reshape(n,len(id_y))
    Y = Y / Y.std(axis=0)  # .reshape(n,len(id_y))
    Z = df[:, id_z]
    Z = Z - Z.mean(axis=0)  # .reshape(n,len(id_z))
    Z = Z / Z.std(axis=0)  # .reshape(n,len(id_z))
    if len(id_z) > 0:
        beta = np.linalg.lstsq(Z, Y)[0]
        residual = Y - np.dot(Z, beta)
    else:
        residual = Y
    return residual


def pcor(df, id_x, id_y, id_z):
    ex = get_residuals(df, id_x, id_z)
    ey = get_residuals(df, id_y, id_z)
    pcr, dd = scipy.stats.pearsonr(ex, ey)
    return pcr


def p_value(pcr, n_samples, dim_cond_set):
    degfr = max(n_samples - dim_cond_set - 2, 1)
    pcr2 = pcr ** 2
    if pcr2 == 1:
        p_val = 0
    # elif degfr < 1:
    #    p_val = np.nan
    else:
        value = np.sqrt((degfr * pcr2) / (1. - pcr2))
        p_val = scipy.stats.t.sf(value, degfr) * 2
    return p_val


def pcor_p(df, id_x, id_y, id_z):
    pcr = pcor(df, id_x, id_y, id_z)
    n = df.shape[0]
    d = len(id_z)
    p_val = p_value(pcr, n, d)
    return p_val


# def normalranktransform(df):
#     n = df.shape[0]
#     d = df.shape[1]
#     S = np.empty(shape=(n, 0))
#     for j in range(d):
#         v = scipy.stats.norm.ppf(scipy.stats.rankdata(df[:, j], method='ordinal') / (n + 1)).reshape((n, 1))
#         S = np.concatenate((S, v), axis=1)
#     return S

def normalranktransform(df):
    n = df.shape[0]
    d = df.shape[1]
    df = pd.DataFrame(df)
    S = df.rank(method='first').as_matrix()/(n+1)
    return scipy.stats.norm.ppf(S).reshape((n,d))

def nrt(df):
    return normalranktransform(df.transpose()).transpose()


def nrt_pcor_p(df, id_x, id_y, id_z):
    return pcor_p(normalranktransform(df), id_x, id_y, id_z)

############# END normal rank transform partial correlation test ##########################


############# tranform text table of CIT to ASP file ##########################


def list_to_number(lst):
    return sum({(2 ** f) for f in lst})


# def list_to_number_cmpl(lst1,lst2=range(4)):
#    diff = list(set(lst2).difference(set(lst1)))
#    return list_to_number(diff)

def str_to_number(s):
    return list_to_number(str_to_list(s))


def strs_to_cmpl_number(X, Y, Z, do='[]', d=4):
    lst = str_to_list(X) + str_to_list(Y) + str_to_list(Z) + str_to_list(do)
    set1 = set(range(d)).difference(lst)
    return list_to_number(list(set1))


def strs_to_asp_expr(X, Y, Z, do='[]', d=4, w=0, indep=True):
    if indep:
        ind = 'in'
    else:
        ind = ''
    expr = ind + 'dep( %d , %d , %d , %d , %d , %d ). \n' % (
    str_to_list(X)[0], str_to_list(Y)[0], str_to_number(Z), str_to_number(do), strs_to_cmpl_number(X, Y, Z, '[]', d),
    int(w))
    return expr


def transform_tests_file_to_asp_file(filepath,filename,al=0.01,mul=1000,infty=1000,max_do=100):
    #asp_filename = 'aspes'+ filename[5:-4]+'al_'+str(al)+'_mul_'+str(mul)+'_.ind'
    asp_filename = filename[:-4]+str(max_do)+'_al_'+str(al)+'_mul_'+str(mul)+'_.pl'
    df = pd.read_csv(filepath+filename)
    A = np.loadtxt(filepath+"graph_true.csv", delimiter=",")
    d = A.shape[0]
    node_str = "#const nrnodes = "+str(d)+". \n" ##const nrnodes = 4.
    with open(filepath+asp_filename, 'a') as ff:
            ff.write(node_str)
    for k in range(df.shape[0]):
        #np.seterr(divide='ignore')
        indep = (df['p-val'].iloc[k] >= al)
        if al != 0:
            odds = df['p-val'].iloc[k]/al
        else:
            odds = np.inf
        if odds <= 1e-316:
            score = -infty
        else:
            score = np.log(odds)
        if mul == np.inf:
            weight = infty
        else:
            weight =  np.minimum(mul*np.absolute(score),mul*infty)
        if np.isnan(weight):
            weight = 0
        weight = int(weight)
        #np.seterr(divide='warn')
        #np.seterr(divide='warn')
        asp_expr = strs_to_asp_expr(df['X'].iloc[k],df['Y'].iloc[k],df['Z'].iloc[k],df['do_targets'].iloc[k],d=d,w=weight,indep=indep)
        if len(str_to_list(df['do_targets'].iloc[k])) <= max_do:
            with open(filepath+asp_filename, 'a') as ff:
                ff.write(asp_expr)
    print('Saved asp file to:', filepath+asp_filename)
    return asp_filename


def transform_all_tests_files_to_asp_files(filepath, al=0.01, mul=1000, infty=1000):
    files = [file for file in os.listdir(filepath) if re.match('^tests.*\.csv$', file)]
    for file in files:
        transform_tests_file_to_asp_file(filepath, file, al, mul, infty)
    return None


############# END tranform text table of CIT to ASP file ##########################


############# run Clingo and save things into files ##########################



def get_penalty_value_from_file(filename,filepath=''):
    go = True
    value = 0
    with open(filepath+filename) as fl:
        for line in fl:
            if 'UNSATISFIABLE' in line:
                value = np.inf
                go = False
            elif go and 'SATISFIABLE' in line:
                value = 0
                go = False
            elif go and 'Optimization : ' in line:
                value = [int(s) for s in line.split() if s.isdigit()][0]
    return value

def get_edge_from_files(filename_vs, filename_pro, filepath=''):
    w_vs = get_penalty_value_from_file(filename_vs,filepath) #penalty against edge
    w_pro = get_penalty_value_from_file(filename_pro,filepath) #penalty pro edge
    if w_vs == w_pro:
        confidence_pro_edge = 0
        z = 1/2
    else:
        confidence_pro_edge = w_vs - w_pro
        z = (1+np.sign(confidence_pro_edge))/2
    return (z,confidence_pro_edge) #absolute(confidence_pro_edge))

def create_tmp_edge_files(node0,node1,filepath='',edge_type='edge',create_tmp=True):
    tmp_file_vs = edge_type+'_'+str(node0)+'_'+str(node1)+'_vs_.txt'
    tmp_file_pro = edge_type+'_'+str(node0)+'_'+str(node1)+'_pro_.txt'
    if create_tmp:
        with open(filepath+tmp_file_vs, 'w') as ff_vs:
            ff_vs.write(':-'+edge_type+'('+str(node0)+','+str(node1)+').')
        with open(filepath+tmp_file_pro, 'w') as ff_pro:
            ff_pro.write(edge_type+'('+str(node0)+','+str(node1)+').')
    return(tmp_file_vs,tmp_file_pro)


def get_edge_from_asp(node0, node1, asp_input_file, filepath='', clingodir='', aspdir='',
                      edge_type='edge', sep='s', create_tmp=True, remove_tmp=False):
    # tmp_file_vs = edge_type+'_'+str(node0)+'_'+str(node1)+'_vs_.txt'
    # tmp_file_pro = edge_type+'_'+str(node0)+'_'+str(node1)+'_pro_.txt'
    # if create_tmp:
    #    print(filepath)
    #    print(tmp_file_vs)
    #    print(filepath+tmp_file_vs)
    #    with open((filepath+tmp_file_vs), 'w') as ff_vs:
    #        ff_vs.write(':-'+edge_type+'('+str(node0)+','+str(node1)+').')
    #    with open((filepath+tmp_file_pro), 'w') as ff_pro:
    #        ff_pro.write(edge_type+'('+str(node0)+','+str(node1)+').')
    tmp_file_vs, tmp_file_pro = create_tmp_edge_files(node0, node1, filepath, edge_type, create_tmp)
    result_file_vs = splitext(basename(asp_input_file))[0] + '_sep_' + sep + '_' + tmp_file_vs
    result_file_pro = splitext(basename(asp_input_file))[0] + '_sep_' + sep + '_' + tmp_file_pro
    command = clingodir + "clingo --quiet=2,1 -W no-atom-undefined "
    # command = clingodir + "clingo "
    if sep == 's':
        command += aspdir + "sigma_hej_cyclic.pl "
    if sep == 'd':
        command += aspdir + "hej_cyclic.pl "
    if sep == 'a':
        command += aspdir + "hej_acyclic.pl "
    command += aspdir + "partial_comp_tree.pl "
    command += filepath + asp_input_file + ' '
    command_vs = command + filepath + tmp_file_vs
    command_vs += ' > ' + filepath + result_file_vs
    command_pro = command + filepath + tmp_file_pro
    command_pro += ' > ' + filepath + result_file_pro
    c_vs = subprocess.call(command_vs, shell=True)
    c_pro = subprocess.call(command_pro, shell=True)
    # c_vs.communicate()
    # c_pro.communicate()
    z, w = get_edge_from_files(result_file_vs, result_file_pro, filepath)
    if remove_tmp:
        os.remove(filepath + tmp_file_vs)
        os.remove(filepath + tmp_file_pro)
    os.remove(filepath + result_file_vs)
    os.remove(filepath + result_file_pro)
    return (z, w)


def get_graph_from_asp(asp_input_file, filepath='', clingodir='', aspdir='', sep='s', remove_tmp=False):
    with open(filepath + asp_input_file) as f:
        d = int(f.readline().split()[-1].split('.')[0])
    edges = np.zeros(shape=(d, d))
    confs = np.zeros(shape=(d, d))
    edges_score = np.zeros(shape=(d, d))
    confs_score = np.zeros(shape=(d, d))
    for row in range(d):
        for col in range(d):  # edge, conf row <- col
            if col != row:
                (z_edge, w_edge) = get_edge_from_asp(col, row,
                                                     asp_input_file, filepath, clingodir, aspdir,
                                                     edge_type='edge', sep=sep, create_tmp=True, remove_tmp=False)
                edges[row, col] = z_edge
                edges_score[row, col] = w_edge
    edges_file_name = 'edges_pred_' + sep + '_sep_' + splitext(basename(asp_input_file)[6:])[0] + '.csv'
    np.savetxt(filepath + edges_file_name, edges, fmt='%1.1f', delimiter=",")
    edges_score_file_name = 'edges_score_' + sep + '_sep_' + splitext(basename(asp_input_file)[6:])[0] + '.csv'
    np.savetxt(filepath + edges_score_file_name, edges_score, fmt='%i', delimiter=",")
    for row in range(d):
        for col in range(d):
            if col > row:
                (z_conf, w_conf) = get_edge_from_asp(row, col,
                                                     asp_input_file, filepath, clingodir, aspdir,
                                                     edge_type='conf', sep=sep, create_tmp=True, remove_tmp=False)
                confs[row, col] = z_conf
                confs[col, row] = z_conf
                confs_score[row, col] = w_conf
                confs_score[col, row] = w_conf
    confs_file_name = 'confs_pred_' + sep + '_sep_' + splitext(basename(asp_input_file)[6:])[0] + '.csv'
    np.savetxt(filepath + confs_file_name, confs, fmt='%1.1f', delimiter=",")
    confs_score_file_name = 'confs_score_' + sep + '_sep_' + splitext(basename(asp_input_file)[6:])[0] + '.csv'
    np.savetxt(filepath + confs_score_file_name, confs_score, fmt='%i', delimiter=",")
    return (edges, confs, edges_score, confs_score)


def get_graphs_from_asp(asp_input_file, filepath='', clingodir='', aspdir='', remove_tmp=True):
    for sep in ['s', 'd', 'a']:
        get_graph_from_asp(asp_input_file, filepath, clingodir, aspdir, sep, remove_tmp=False)
    if remove_tmp == True:
        files = [file for file in os.listdir(filepath) if re.match('^edge.*\.txt$', file)]
        for file in files:
            os.remove(filepath + file)
        files = [file for file in os.listdir(filepath) if re.match('^conf.*\.txt$', file)]
        for file in files:
            os.remove(filepath + file)
    return None

############# END run Clingo and save things into files ##########################

############# run experiments  ##########################

def run_all_CIT_and_save(filepath,n=10000,actfct=np.tanh,sc=1,noise='normal',sd=1,nbr_do=0,max_do=0,do_strategy=2):
    W,b = load_weights_and_bias(filepath)
    A = np.loadtxt(filepath+"graph_true.csv",delimiter=',')
    d = A.shape[0]
    tests_filename = 'tests'+name_suffix(actfct.__name__,sc,noise,sd,nbr_do,max_do)  #do_targets)
    header = not os.path.isfile(filepath+tests_filename)
    partitions = all_xyZ_partitions_list(range(d))
    if do_strategy == 0: # all interventions with target size <= max_do
        do_list = [t for t in powerlist(range(d)) if len(t) <= max_do]
    elif do_strategy == 1: # random max_do-node interventions, drawn with replacement
	do_list =[[]]
	for ll in range(nbr_do):
	    do_list +=[random.sample(range(d), max_do)]
    elif do_strategy == 2: # random single-node interventions, drawn without replacement
        do_list = [[]] + [[t] for t in random.sample(range(d), nbr_do)]
    for do_targets in do_list:
        do_t = np.array(do_targets)
        S = sample_from_mSCM(W,b,do_t,n,d,actfct,sc,sd,noise)
        S = normalranktransform(S)
        for id_x, id_y, id_z in partitions:
            one_row = pd.DataFrame()
            #one_row.loc[0,'n'] = int(n)
            one_row.loc[0,'X'] = '['+';'.join([str(s) for s in id_x])+']'
            one_row.loc[0,'Y'] = '['+';'.join([str(s) for s in id_y])+']'
            one_row.loc[0,'do_targets'] = '['+';'.join([str(s) for s in do_targets])+']'
            one_row.loc[0,'Z'] = '['+';'.join([str(s) for s in id_z])+']'
            one_row.loc[0,'p-val'] = pcor_p(S, id_x, id_y, id_z)
            with open((filepath+tests_filename), 'a') as f:
                one_row.to_csv(f, header=header,index=False)
            header = False
    print('Saved test results to : ',filepath+tests_filename)
    return tests_filename



def sample_mSCM_run_all_and_save(d=4,k=2,p=0.3,m=0,nbr=-1,add_ind_noise_to_A=True,add_ind_noise_to_W=True,
                                 include_latent=True,folderpath="/zfs/ivi/causality/pforre1/sigmasep/mSCM_data/",
                                 AF=[np.tanh],SC=[1],NOI=['uniform'],SD=[3],n=10000,
                                 AL =[0.01],MUL=[1000],infty=1000,nbr_do=0,max_do=0,do_strategy=2,
                                 clingodir="/zfs/ivi/causality/opt/clingo-4.5.4-linux-x86_64/",
                                 aspdir="/zfs/ivi/causality/pforre1/sigmasep/ASP/"):
    #np.seterr(divide='ignore', invalid='ignore')
    start_time = time.time()
    AF = set(AF)
    SC = set(SC)
    NOI = set(NOI)
    SD = set(SD)
    AL = set(AL)
    MUL = set(MUL)
    if nbr < 0:
        nbr = len(os.listdir(folderpath))+1
    ainA = int(add_ind_noise_to_A)
    ainW = int(add_ind_noise_to_W)
    il = int(include_latent)
    foldername = 'dataset_%06d'%(nbr)
    foldername += '_d'+str(d)+'k'+str(k)+'m'+str(m)+'p'+str(p)+'ain'
    foldername += str(ainA)+str(ainW) +'il'+str(il)+'/'
    filepath = folderpath+foldername
    os.makedirs(filepath)
    A = sample_adjacency_matrix(d,k,p,add_ind_noise_to_A)
    V = extract_edges(A)
    H = extract_confs(A)
    np.savetxt(filepath+"graph_true.csv",A,fmt='%i',delimiter=",")
    np.savetxt(filepath+"edges_true.csv",V,fmt='%i',delimiter=",")
    np.savetxt(filepath+"confs_true.csv",H,fmt='%i',delimiter=",")
    c = num_hidden_units_to_add(A,m)
    W, b = sample_weights_and_bias(A,c,add_ind_noise_to_W,include_latent)
    np.savetxt(filepath+"weights_bias.csv",np.concatenate((W,b),axis=1),delimiter=",")
    for actfct,sc,noise,sd in product(AF,SC,NOI,SD):
        tests_filename = run_all_CIT_and_save(filepath,n=n,actfct=actfct,sc=sc,noise=noise,sd=sd,nbr_do=nbr_do,max_do=max_do,do_strategy=do_strategy)
        for al, mul in product(AL,MUL):
            asp_input_file = transform_tests_file_to_asp_file(filepath,tests_filename,al=al,mul=mul,infty=infty)
            get_graphs_from_asp(asp_input_file,filepath,clingodir,aspdir,remove_tmp=False)
        #print('Saved test results to : ',filepath+tests_filename)
    files = [file for file in os.listdir(filepath) if re.match('^edge.*\.txt$',file)]
    for file in files:
        os.remove(filepath+file)
    files = [file for file in os.listdir(filepath) if re.match('^conf.*\.txt$',file)]
    for file in files:
        os.remove(filepath+file)
    run_time = time.time() - start_time
    print('--- Doing all CIT together took: ',run_time,' seconds. ---')
    return (filepath, run_time)





def run_all_for_mSCM_A_and_save(A,nbr=-1,  m=0, add_ind_noise=True, include_latent=True,
                       folderpath="/zfs/ivi/causality/pforre1/sigmasep/mSCM_data/",
                       AF=[np.tanh], SC=[1], NOI=['normal'], SD=[1], n=10000,
                       AL=[0.01], MUL=[1000], infty=1000, nbr_do=0,max_do=0,do_strategy=2,
                       clingodir="/zfs/ivi/causality/opt/clingo-4.5.4-linux-x86_64/",
                       aspdir="/zfs/ivi/causality/pforre1/sigmasep/ASP/"):
    start_time = time.time()
    d = A.shape[0]
    AF = set(AF)
    SC = set(SC)
    NOI = set(NOI)
    SD = set(SD)
    AL = set(AL)
    MUL = set(MUL)
    if nbr < 0:
        nbr = len(os.listdir(folderpath)) + 1
    foldername = 'dataset_%06d' % (nbr)
    foldername += '_d' + str(d) + 'k1'  + 'm' + str(m) +'do'+str(nbr_do)+ '/'
    filepath = folderpath + foldername
    os.makedirs(filepath)
    V = extract_edges(A)
    H = extract_confs(A)
    np.savetxt(filepath + "graph_true.csv", A, fmt='%i', delimiter=",")
    np.savetxt(filepath + "edges_true.csv", V, fmt='%i', delimiter=",")
    np.savetxt(filepath + "confs_true.csv", H, fmt='%i', delimiter=",")
    c = num_hidden_units_to_add(A, m)
    W, b = sample_weights_and_bias(A, c, add_ind_noise, include_latent)
    np.savetxt(filepath + "weights_bias.csv", np.concatenate((W, b), axis=1), delimiter=",")
    for actfct, sc, noise, sd in product(AF, SC, NOI, SD):
        tests_filename = run_all_CIT_and_save(filepath, n=n, actfct=actfct, sc=sc, noise=noise, sd=sd, nbr_do=nbr_do,max_do=max_do,do_strategy=do_strategy)
        for al, mul in product(AL, MUL):
            asp_input_file = transform_tests_file_to_asp_file(filepath, tests_filename, al=al, mul=mul, infty=infty)
            get_graphs_from_asp(asp_input_file, filepath, clingodir, aspdir, remove_tmp=False)
            # print('Saved test results to : ',filepath+tests_filename)
    files = [file for file in os.listdir(filepath) if re.match('^edge.*\.txt$', file)]
    for file in files:
        os.remove(filepath + file)
    files = [file for file in os.listdir(filepath) if re.match('^conf.*\.txt$', file)]
    for file in files:
        os.remove(filepath + file)
    run_time = time.time() - start_time
    print('--- Doing all tests together took: ', run_time, ' seconds. ---')
    return (filepath, run_time)



