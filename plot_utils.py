import numpy as np

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

col_lif='C0'
col_poisson='C1'

def cm_to_in(val):
    return val/2.54

full_width=cm_to_in(15.5)
half_width=full_width/2

dpi_out=300
scale_size=100/dpi_out
large = 12; med = 11; small = 10; xsmall=9
plot_params = {'axes.titlesize': med,
          'legend.fontsize': small,
          'axes.labelsize': small,
          'xtick.labelsize': small,
          'ytick.labelsize': small,
          'figure.titlesize': large,
          'lines.linewidth': 2   *scale_size,
          'lines.markersize': 3  *scale_size,
         'xtick.major.width' : 1 *scale_size,
         'xtick.minor.size' : 5  *scale_size,
         'xtick.minor.width' : 1 *scale_size,
         'ytick.major.size': 8   *scale_size,
         'ytick.major.width' : 1 *scale_size,
         'ytick.minor.size' : 5  *scale_size,
         'ytick.minor.width' : 1 *scale_size,
         'figure.constrained_layout.use': False,
          'font.size': 10
}
    
plt.rcParams.update(plot_params)

plt.rcParams['figure.figsize'] = [full_width,2/3* full_width]
plt.rcParams['figure.dpi'] = 100

def show_labels(ax,show_label_ax,ylabel,xlabel='Time (ms)',**kwargs):
    """Expected input: 'all' |'x'|'y'|None"""
    if show_label_ax:
        if show_label_ax==True or show_label_ax.lower()=='all':
            ax.set_xlabel(xlabel,**kwargs)
            ax.set_ylabel(ylabel,**kwargs)
        elif show_label_ax.lower()=='x':
            ax.set_xlabel(xlabel,**kwargs)
        elif show_label_ax.lower()=='y':
            ax.set_ylabel(ylabel,**kwargs)

def default_col_sim(sim):
    if type(sim)==str: 
        sim_type=sim
    else: sim_type=sim.type
    if 'lif' in sim_type: col_sim= col_lif
    elif 'pois' in sim_type: col_sim=col_poisson
    else: col_sim=None
    return col_sim
default_col_x='darkorchid'

def plot_input(sim, name="",ax=None):
    if ax is None:
        ax=plt.gca()
    ax.set_xlim([0, sim.T])
    ax.plot(sim.t_grid,sim.input_s[0],'C1')
    ax.set_ylabel('Input s')

    
    
def plotFirTimes(sim, name="",ax=None,show_label_ax=True, show_legend=True):
    if ax is None:
        ax=plt.gca()
    ax.set_title('Spike raster '+name)
    show_labels(ax,show_label_ax,xlabel='Time (ms)',ylabel='Neuron index')
    ax.set_xlim(0, sim.T)
    ax.set_ylim(0, sim.nNeur+1)
    timesPosSpikes=np.where(sim.weights[0,sim.spike_neur]>0)[0]
    timesNegSpikes=np.where(sim.weights[0,sim.spike_neur]<0)[0]
    ax.plot(sim.spike_times[timesPosSpikes]*sim.t_step,sim.spike_neur[timesPosSpikes]+1,'.r',  # markersize=0.5
             label='w+')
    ax.plot(sim.spike_times[timesNegSpikes]*sim.t_step,sim.spike_neur[timesNegSpikes]+1,'.b',
              label='w-') # markersize=0.5, xxx
    if show_legend:
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        
        
        
def plotNetPerf(sim, ax=None, sim2=None, name1="Estimate", name2="",cutoff=0,show_label_ax=True,show_legend=True,show_title=True,
                col='default',col2='default',col_x=default_col_x):
    if ax is None:
        ax=plt.gca()
    if show_title: ax.set_title('Network performance')
    show_labels(ax,show_label_ax,xlabel='Time (ms)',ylabel=r'$x, \hat x$')
    cutoff1=int(cutoff/sim.t_step)
    ax.set_xlim([cutoff, sim.T])
    ax.plot(sim.t_grid[cutoff1:], sim.signal[0,cutoff1:],c=col_x,label='True Signal')
    if col=='default': col_sim=default_col_sim(sim)
    else: col_sim=col #xxx
    ax.plot(sim.t_grid[cutoff1:],sim.predSignal[0,cutoff1:],label=name1,c=col_sim) # label='Estimate '+name1)
    if sim2 is not None:
        cutoff2=int(cutoff/sim2.t_step)
        if col2=='default': col_sim2=default_col_sim(sim2)
        else: col_sim2=col2 #xxx
        ax.plot(sim2.t_grid[cutoff2:],sim2.predSignal[0,cutoff2:], c=col_sim2, label=name2) # label='Estimate '+name2)
    if show_legend:
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left') #xxx
    #if np.max(sim.predSignal[0,:])>20: ax.ylim([-2,8])

def get_ax_size(ax): #TODO adjust figures with that?
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height



def plotSampleVm(sim, ax=None,name="",pos_only=False, show_label_ax=True,
                 show_legend=True, msize=5,moffset=0.06):
    if ax is None:
        ax=plt.gca()
    ax.set_xlim([0, sim.T])
    ax.set_title('Two sample membrane potentials '+name)
    show_labels(ax,show_label_ax,xlabel='Time (ms)',ylabel=r'$V$ (a.u.)')
    indPosSpikes=[pos_weight for pos_weight in np.where(sim.weights>0)[1]
                        if pos_weight in sim.spike_neur]
    indNegSpikes=[neg_weight for neg_weight in np.where(sim.weights<0)[1]
                        if neg_weight in sim.spike_neur]
    if len(indPosSpikes): sample_posWeight=indPosSpikes[0]
    else: # if there are no spikes
        sample_posWeight=np.where(sim.weights[0,:]>0)[0][0]
    ax.plot(sim.t_grid, sim.Vmembr[sample_posWeight,:],'r', \
             label='w = {:+}'.format(sim.weights[0,sample_posWeight]))
    col_threshold='black'
    if 'lif' in sim.type: 
        ax.axhline(sim.Vthr,color=col_threshold,linestyle='--')
        marker_voffset=moffset*sim.Vthr
        ax.text(sim.T*1.01,sim.Vthr,r'$\vartheta$',verticalalignment='center',color=col_threshold)
    else: marker_voffset=0
    ax.plot(sim.spike_times[sim.spike_neur==sample_posWeight]*sim.t_step, sim.Vmembr[sample_posWeight, 
                                        sim.spike_times[sim.spike_neur==sample_posWeight]]+marker_voffset,
            'k^', markersize=msize)
    if not pos_only:
        if len(indNegSpikes)>0: sample_negWeight=indNegSpikes[0]
        else: sample_negWeight=np.where(sim.weights[0,:]<0)[0][0]

        ax.plot(sim.t_grid, sim.Vmembr[sample_negWeight,:],'b', \
                 label='w = {:+}'.format(sim.weights[0,sample_negWeight]))
        ax.plot(sim.spike_times[sim.spike_neur==sample_negWeight]*sim.t_step, sim.Vmembr[sample_negWeight, 
                                        sim.spike_times[sim.spike_neur==sample_negWeight]]+marker_voffset,'k^',      \
                 label='Spike', markersize=msize)
        if show_legend:
            ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
            
            
            
def pltSmoothedRates(sim, ax=None, sig=50, name1="LIF", sim2=None,
                     sig2=None, name2="Poisson", show_label_ax='all', 
                     show_title=True, show_legend=True):
    if ax is None:
        ax=plt.gca()
    ind_pos_weight=np.where(sim.weights>0)[1]
    ind_neg_weight=np.where(sim.weights<0)[1]
    
    sim_rate_pos=1/len(ind_pos_weight)*np.sum(sim.spikes[ind_pos_weight,:]/sim.t_step,0)
    sim_rate_neg=1/len(ind_neg_weight)*np.sum(sim.spikes[ind_neg_weight,:]/sim.t_step,0)
    
    
    plt.margins(0)
    if show_title: plt.title('Firing rates - population means')
    if name1=='': name1='spikeM'
    if name2=='': name2='rateM'
    if sig2==None: sig2=sig
    ax.plot(sim.t_grid,1000*gaussian_filter1d(sim_rate_pos,sigma=sig),"r",markersize=1,
             label=r'$w+$ '+name1)
    ax.plot(sim.t_grid,1000*gaussian_filter1d(sim_rate_neg,sigma=sig),'b',markersize=1,
             label=r'$w-$ '+name1) #use long minus xxx
    if sim2!=None:
        ind_pos_weight2=np.where(sim2.weights>0)[1]
        ind_neg_weight2=np.where(sim2.weights<0)[1]   
        sim2_rate_pos=1/len(ind_pos_weight2)*np.sum(sim2.spikes[ind_pos_weight2,:]/sim2.t_step,0) 
        sim2_rate_neg=1/len(ind_neg_weight2)*np.sum(sim2.spikes[ind_neg_weight2,:]/sim2.t_step,0)
        
        ax.plot(sim.t_grid,1000*gaussian_filter1d(sim2_rate_pos,sigma=sig2),'-',c='violet',label=r'$w+$ '+name2)
        ax.plot(sim.t_grid,1000*gaussian_filter1d(sim2_rate_neg,sigma=sig2),'-',c='skyblue',label=r'$w-$ '+name2)
    if show_legend: plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    show_labels(ax,show_label_ax,xlabel='Time (ms)',ylabel='Firing rate (Hz)')
    
    
def plt_spikeHists(sim, name='', nBins=50,stat=False):
    if stat: 
        start_ind=int(5/(sim.lambd*sim.t_step))
    else: start_ind = 0
    posNeur=np.where(sim.Jconns[0,:]>0)[0]
    negNeur=np.where(sim.Jconns[0,:]<0)[0]
    pos_spikes=np.where(sim.spikes[posNeur,start_ind:]==1)[1]*sim.t_step+start_ind*sim.t_step
    neg_spikes=np.where(sim.spikes[negNeur,start_ind:]==1)[1]*sim.t_step+start_ind*sim.t_step
    if name: plt.title(name+' - Spikes over time')
    _=plt.hist([pos_spikes,neg_spikes],stacked=True,color=['b','r'],bins=nBins,label=[r'$w+$','$w-$'])
    ax= plt.gca()
    plt.margins(y=0.2)
    ax.text(0.85,0.9,f'Total spikes: {int(np.sum(sim.spikes))}',transform=ax.transAxes)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel('Time in ms')
    plt.ylabel('Number of spikes')
    