import os

import numpy as np
import torch
import timeit

from problem import get_problem
from model import ParetoSetModel

from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)
    
# -----------------------------------------------------------------------------
# list of 11 test problems, which are defined in problem.py
ins_list = ['f1','f2','f3','f4','f5','f6',
            're21', 're24', 're33','re36','re37']


# number of independent runs
n_run = 3

# scalarization method: ['ls', 'tch', 'stch']
method = 'stch'
# number of learning steps
n_steps = 2000 
# number of sampled preferences per step
n_pref_update = 10 


# device
device = 'cpu'

# -----------------------------------------------------------------------------
for test_ins in ins_list:
    print(test_ins)
    
    
    if test_ins in ['re21', 're24', 're32', 're33','re36','re37']:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{test_ins}.dat')
        ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat'))
        nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat'))
        
    if test_ins in ['f1','f2','f3','f4','f5','f6']:
        ideal_point = np.zeros(2)
        nadir_point = np.ones(2)
    
    # get problem info
    problem = get_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj

    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        
        start = timeit.default_timer()
        store_hv_step = 0
        
        z = torch.zeros(n_obj).to(device)
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)
            
        # optimizer
        optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
    
        # t_step Pareto Set Learning with gradient descent
        for t_step in range(n_steps):
            psmodel.train()
        
            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_pref_update)
            pref_vec  = torch.tensor(pref).to(device).float() 
            
            # get the current coressponding solutions
            x = psmodel(pref_vec)
            value = problem.evaluate(x)  
            value = (value - torch.tensor(ideal_point).to(device)) / torch.tensor(nadir_point - ideal_point).to(device) 
            
           
            if method == 'ls':
                ls_value =  torch.sum(pref_vec * (value - z), axis = 1)
                loss =  torch.sum(ls_value)
                
            if method == 'tch':
                tch_value =  torch.max(pref_vec * (value - z), axis = 1)[0] 
                loss =  torch.sum(tch_value)
                
            if method == 'stch':
                mu =  0.01 
                stch_value = mu* torch.logsumexp(pref_vec * (value - z) / mu, axis = 1)   
                loss =  torch.sum(stch_value)

            # gradient-based pareto set model update 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
        stop = timeit.default_timer()
        
        
        # calculate and report the hypervolume value
        with torch.no_grad():
            
            psmodel.eval()
            
            generated_pf = []
            generated_ps = []
            
            if n_obj == 2:
                pref = np.stack([np.linspace(0,1,100), 1 - np.linspace(0,1,100)]).T
                pref = torch.tensor(pref).to(device).float()
            
            if n_obj == 3:
                pref_size = 1035
                pref = torch.tensor(das_dennis(44,3)).to(device).float()   
               
        
            sol = psmodel(pref)
            obj = problem.evaluate(sol)
            generated_ps = sol.cpu().numpy()
            generated_pf = obj.cpu().numpy()
            
            results_F_norm = (generated_pf - ideal_point) / (nadir_point - ideal_point) 
            
            hv = HV(ref_point=np.array([1.1] * n_obj))
            hv_value = hv(results_F_norm)
           
            print('Time: ', stop - start)  
            print("hv_gap", "{:.4e}".format(np.mean(hv_value)))
        
        
        # plot the learned Pareto front
        SMALL_SIZE = 8
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 15
        MAX_SIZE=18
        plt.rc('font', family='Times New Roman', size=BIGGER_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MAX_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
       
        if n_obj == 2:
        
            fig = plt.figure()
          
            plt.scatter(generated_pf[:,0],generated_pf[:,1], c = 'tomato',  alpha = 1, lw = 1, label='STCH', zorder = 2)
            
            plt.xlabel(r'$f_1(x)$',size = 16)
            plt.ylabel(r'$f_2(x)$',size = 16)
        
          
            handles = []
            pareto_front_label = plt.Line2D((0,1),(0,0), color='k', marker='o', linestyle='', label = 'Pareto Front')
            label = plt.Line2D((0,1),(0,0), color='tomato', marker='o', lw = 2, label = 'STCH')
            
            handles.extend([pareto_front_label])
            handles.extend([label])
            
            
            order = [0,1]
            plt.legend(fontsize = 14)
            plt.legend(handles=handles,fontsize = 14, scatterpoints=3,
                    bbox_to_anchor = (1, 1))
            
            plt.grid()
           
            
        if n_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
        
            ax.scatter(generated_pf[:,0],generated_pf[:,1],generated_pf[:,2], c = 'tomato', s =50, label = 'STCH')
            max_lim = np.max(generated_pf, axis = 0)
            min_lim = np.min(generated_pf, axis = 0)
            
            ax.set_xlim(min_lim[0], max_lim[0])
            ax.set_ylim(max_lim[1],min_lim[1])
            ax.set_zlim(min_lim[2], max_lim[2])
            
            if test_ins == 're37':
                ax.set_xlim(0, 1)
                ax.set_ylim(1,0)
                ax.set_zlim(-0.5, 0.5)
            
            ax.set_xlabel(r'$f_1(x)$',size = 12)
            ax.set_ylabel(r'$f_2(x)$',size = 12)
            ax.set_zlabel(r'$f_3(x)$',size = 12)
            
            plt.legend(loc=1, bbox_to_anchor=(1,1))
            
    
    print("************************************************************")


