#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# switch to Physical Review compatible font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb,txfontsb}']
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'roman'
matplotlib.rcParams['font.size'] = 8

# DOS of NRL tight-binding model
nrg_nrl , dos_nrl = np.loadtxt("nrl.dos",usecols=(0,1),unpack=True)
ef_nrl = 2.974605

# DOS of DFT data from VASP
nrg_dft , dos_dft = np.loadtxt("dft.dos",usecols=(0,1),unpack=True)
ef_dft = 8.4055

# temperature dependence of Fermi energy & internal energy
t_hot, ef_hot , tb_hot , dft_hot = np.loadtxt("energy.out",usecols=(0,1,2,3),unpack=True)

# tight-binding parameters
r_tb, sss, sps, pps, ppp, sds, pds, pdp, dds, ddp, ddd = np.loadtxt("model.nrl",usecols=(0,1,2,3,4,5,6,7,8,9,10),unpack=True)

# decay of the density matrix
r_dm3, dm3 = np.loadtxt("decay.T3",usecols=(0,1),unpack=True)
r_dm1, dm1 = np.loadtxt("decay.T1",usecols=(0,1),unpack=True)
r_dm03, dm03 = np.loadtxt("decay.T03",usecols=(0,1),unpack=True)
r_dm01, dm01 = np.loadtxt("decay.T01",usecols=(0,1),unpack=True)
r_dm003, dm003 = np.loadtxt("decay.T003",usecols=(0,1),unpack=True)
rgrid = np.linspace(0.0001,50,1000)

fig = plt.figure(figsize=(3.37,5.06))

ax1 = plt.subplot2grid((7, 2), (0, 0), colspan=2, rowspan=2)

plt.ylim(0.0,5.5)
plt.xlim(-10.5,27.5)

ax1.plot(nrg_dft-ef_dft,dos_dft*(2.5+1.5*np.sign(nrg_dft-ef_dft)),color='red',lw=0.75)
ax1.plot(nrg_nrl-ef_nrl,dos_nrl*(2.5+1.5*np.sign(nrg_nrl-ef_nrl)),color='black',lw=0.75)
ax1.plot((0,0),(0,5.5),color='white',lw=1.0)
ax1.plot((0,0),(0,5.5),color='black',lw=0.5)

ax1.arrow(0.5,4.1,1.5,0.0,width=0.07,lw=0.0,color='black',head_width=0.4,head_length=0.6)
ax1.text(0.5,4.5,r'4$ \times$ amplification')
ax1.text(18.5,4.5,r'$T = 0.03$ eV')
ax1.text(13.9,3.5,r'NRL')
ax1.text(21.4,2.6,r'DFT')

ax1.set_yticks([0,1,2,3,4,5])

plt.xlabel(r'Energy relative to chemical potential (eV)', labelpad=3)
plt.ylabel(r'Density of states (eV$^{-1}$)', labelpad=4)

ax2 = plt.subplot2grid((7, 2), (2, 0), rowspan=3)

plt.xlim(0.0,3.0)
plt.ylim(-0.125,3.325)

ax2.plot((0,3.0),(0,0),color='black',lw=0.5)
ax2.plot(t_hot,ef_hot-ef_hot[0],color='black',lw=0.75)
ax2.plot(t_hot,tb_hot-dft_hot,color='red',lw=0.75)

ax2.text(0.18,1.55,r'$\mu(T) - \mu(0)$')
ax2.text(1.53,0.04,r'$E_{\mathrm{NRL}}-E_{\mathrm{DFT}}$')

plt.xlabel(r'Temperature (eV)', labelpad=3)
plt.ylabel(r'Energy (eV)', labelpad=4)

ax3 = plt.subplot2grid((7, 2), (2, 1), rowspan=3)

plt.xlim(1.6,6.6)
plt.ylim(-1.625,1.875)

ax3.plot((1.5,6.5),(0,0),color='black',lw=0.5)
ax3.plot(r_tb,pps,lw=0.75,color='black')
ax3.plot(r_tb,sps,lw=0.75,color='darkred')
ax3.plot(r_tb,pdp,lw=0.75,color='red')
ax3.plot(r_tb,ddp,lw=0.75,color='orange')
ax3.plot(r_tb,ddd,lw=0.75,color='lime')
ax3.plot(r_tb,ppp,lw=0.75,color='green')
ax3.plot(r_tb,dds,lw=0.75,color='deepskyblue')
ax3.plot(r_tb,sds,lw=0.75,color='blue')
ax3.plot(r_tb,pds,lw=0.75,color='purple')
ax3.plot(r_tb,sss,lw=0.75,color='black')

ax3.text(1.8,pps[0]-0.03,r'pp$\sigma$',fontsize=6)
ax3.text(1.83,sps[0]-0.03,r'sp$\sigma$',fontsize=6)
ax3.text(1.8,pdp[0]-0.00,r'pd$\pi$',fontsize=6)
ax3.text(1.8,ddp[0]-0.09,r'dd$\pi$',fontsize=6)
ax3.text(1.8,ddd[0]-0.09,r'dd$\delta$',fontsize=6)
ax3.text(1.8,ppp[0]-0.12,r'pp$\pi$',fontsize=6)
ax3.text(1.8,dds[0]-0.05,r'dd$\sigma$',fontsize=6)
ax3.text(1.83,sds[0]-0.20,r'sd$\sigma$',fontsize=6)
ax3.text(1.8,pds[0]-0.15,r'pd$\sigma$',fontsize=6)
ax3.text(1.86,sss[0]-0.09,r'ss$\sigma$',fontsize=6)

ax3.set_xticks([2,3,4,5,6])

plt.xlabel(r'Cu-Cu separation (\AA)', labelpad=1)

ax4 = plt.subplot2grid((7, 2), (5, 0), colspan=2, rowspan=2)

plt.xlim(-1,50)
plt.ylim(1e-5,10)

ax4.semilogy(rgrid,4.0*pow(rgrid,-2),linestyle='--',dashes=(2.5,2.5),lw=0.75,color='black')
ax4.semilogy(rgrid,4.0*pow(rgrid,-2)*np.exp(-(0.00403/0.529)*rgrid),lw=0.75,color='black')
ax4.semilogy(r_dm003, dm003,lw=0,mew=0,ms=1.8,marker='.',color='black')
ax4.semilogy(rgrid,4.0*pow(rgrid,-2)*np.exp(-(0.0403/0.529)*rgrid),lw=0.75,color='red')
ax4.semilogy(r_dm03, dm03,lw=0,mew=0,ms=1.8,marker='.',color='red')
ax4.semilogy(rgrid,4.0*pow(rgrid,-2)*np.exp(-(0.133/0.529)*rgrid),lw=0.75,color='green')
ax4.semilogy(r_dm1, dm1,lw=0,mew=0,ms=1.8,marker='.',color='green')
ax4.semilogy(rgrid,4.0*pow(rgrid,-2)*np.exp(-(0.370/0.529)*rgrid),lw=0.75,color='blue')
ax4.semilogy(r_dm3, dm3,lw=0,mew=0,ms=1.8,marker='.',color='blue')

ax4.semilogy((50.7,36.8),(3e-4,1.7),lw=0,mew=0,ms=3.5,marker='o',color='black',clip_on=False)
ax4.semilogy((45.0,36.8),(6.1e-6,1.7*pow(5,-1)),lw=0,mew=0.75,ms=3.5,marker='+',color='red',clip_on=False)
ax4.semilogy((23.0,36.8),(6.1e-6,1.7*pow(5,-2)),lw=0,mew=0,ms=3.5,marker='^',color='green',clip_on=False)
ax4.semilogy((16.0,36.8),(6.1e-6,1.7*pow(5,-3)),lw=0,mew=0,ms=3.5,marker='s',color='blue',clip_on=False)
ax4.text(38.5,1.1,r'$T = 0.03$ eV')
ax4.text(38.5,1.1*pow(5,-1),r'$T = 0.3$ eV')
ax4.text(38.5,1.1*pow(5,-2),r'$T = 1$ eV')
ax4.text(38.5,1.1*pow(5,-3),r'$T = 3$ eV')
ax4.tick_params(axis='x',length=3)

ax4.set_yticks([10,1,0.1,0.01,0.001,0.0001,0.00001])

plt.xlabel(r'Distance from central atom (\AA)', labelpad=1)
plt.ylabel(r'Density matrix magnitude', labelpad=2.5)

ax1.text(-8.9,4.5,r'(a)')
ax2.text(0.31,2.95,r'(b)')
ax3.text(5.6,1.5,r'(c)')
ax4.text(1.3,7e-5,r'(d)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig1.pdf",bbox_inches='tight',pad_inches=0.01)
