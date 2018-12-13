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

# function/residual data
func_npole, func_error , func_fnorm, func_energy, func_num, func_force = np.loadtxt("function.txt",unpack=True,skiprows=1)
res_npole, res_error , res_fnorm, res_energy, res_num, res_force = np.loadtxt("residual.txt",unpack=True,skiprows=1)

# localization error data
low_rad, low_fnorm, low_energy, low_num, low_stress = np.loadtxt("local_low.txt",unpack=True,skiprows=1)
mid_rad, mid_fnorm, mid_energy, mid_num, mid_stress = np.loadtxt("local_mid.txt",unpack=True,skiprows=1)
high_rad, high_fnorm, high_energy, high_num, high_stress = np.loadtxt("local_high.txt",unpack=True,skiprows=1)

# local/random radius data
LR_radius, L_fnorm, R_fnorm, L_energy, R_energy, L_num, R_num, L_force, R_force = np.loadtxt("radius.txt",unpack=True,skiprows=1)

fig = plt.figure(figsize=(3.37,5.7))

ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=1)

plt.ylim(1e-8,10)
plt.xlim(1e-8,1)

ax1.loglog(func_error,func_fnorm,'ks',ms=3)
ax1.loglog(func_error,func_force,'ro',ms=3)
ax1.loglog(func_error,func_energy,'gx',ms=4)
ax1.loglog(func_error,func_num,'b+',ms=4)

plt.xlabel(r'function error', labelpad=2)
plt.ylabel(r'observable error (multiple units)')

ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=1)

plt.ylim(1.0e-8,10.0)
plt.xlim(1.0e-8,1.0)

ax2.loglog(res_error,res_fnorm,'ks',ms=3)
ax2.loglog(res_error,res_force,'ro',ms=3)
ax2.loglog(res_error,res_energy,'gx',ms=4)
ax2.loglog(res_error,res_num,'b+',ms=4)

plt.xlabel(r'residual error', labelpad=2)
#plt.ylabel(r'observable error (multiple units)')

x0 = 4e-8
x1 = 1.6e-7
y0 = 1.6
y1 = 1.0
dy = 0.2

plt.plot(x0,1.35*y0,'ks',ms=4)
plt.plot(x0,y0*dy,'ro',ms=4)
plt.plot(x0,y0*dy**2,'o',ms=4,color='white',mec='red',mew=1)
plt.plot(x0,y0*dy**3,'gx',ms=5)
plt.plot(x0,y0*dy**4,'b+',ms=5)

plt.text(x1,y1,r'$\mathbf{M}\odot\mathbf{P}$ ($\| \cdot \|_F$/atom)')
plt.text(x1,y1*dy,r'force (eV/\AA)')
plt.text(x1,y1*dy**2,r'stress (GPa)')
plt.text(x1,y1*dy**3,r'$E$ (eV/atom)')
plt.text(x1,y1*dy**4,r'$N$ (atom$^{-1}$)')

ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1, rowspan=1)

plt.ylim(1.0e-6,1.0)
plt.xlim(6.0,14.0)

ax3.semilogy(LR_radius,L_fnorm,'ks',ms=3)
ax3.semilogy(LR_radius,L_force,'ro',ms=3)
ax3.semilogy(LR_radius,L_energy,'gx',ms=4)
ax3.semilogy(LR_radius,L_num,'b+',ms=4)

ax3.set_xticks([6,8,10,12,14])
plt.xlabel(r'localization radius (\AA)', labelpad=1)
plt.ylabel(r'observable error (multiple units)')

ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1, rowspan=1)

plt.ylim(1.0e-6,1.0)
plt.xlim(6.0,14.0)

ax4.semilogy(LR_radius,R_fnorm,'ks',ms=3)
ax4.semilogy(LR_radius,R_force,'ro',ms=3)
ax4.semilogy(LR_radius,R_energy,'gx',ms=4)
ax4.semilogy(LR_radius,R_num,'b+',ms=4)

ax4.set_xticks([6,8,10,12,14])
plt.xlabel(r'coloring radius (\AA)', labelpad=1)
#plt.ylabel(r'observable error (multiple units)')

ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2, rowspan=1)

plt.ylim(1.0e-5,1.0)
plt.xlim(3.0,60.0)

ax5.semilogy(high_rad,high_fnorm,'ks',ms=3)
ax5.semilogy(high_rad,high_stress,'o',ms=3,color='white',mec='red',mew=0.75)
ax5.semilogy(high_rad,high_energy,'gx',ms=4)
ax5.semilogy(high_rad,high_num,'b+',ms=4)

ax5.semilogy(mid_rad[6:],mid_fnorm[6:],'ks',ms=3)
ax5.semilogy(mid_rad[6:],mid_stress[6:],'o',ms=3,color='white',mec='red',mew=0.75)
ax5.semilogy(mid_rad[6:],mid_energy[6:],'gx',ms=4)
ax5.semilogy(mid_rad[6:],mid_num[6:],'b+',ms=4)

ax5.semilogy(low_rad[18:],low_fnorm[18:],'ks',ms=3)
ax5.semilogy(low_rad[18:],low_stress[18:],'o',ms=3,color='white',mec='red',mew=0.75)
ax5.semilogy(low_rad[18:],low_energy[18:],'gx',ms=4)
ax5.semilogy(low_rad[18:],low_num[18:],'b+',ms=4)

ax5.semilogy([13,13],[1e-5,1],'k--',lw=1)
ax5.semilogy([25,25],[1e-5,1],'k--',lw=1)

ax5.text(4,2.5e-5,r'$T = 1$ eV')
ax5.text(14,2.5e-5,r'$T = 0.3$ eV')
ax5.text(26,2.5e-5,r'$T = 0.03$ eV')

plt.xlabel(r'localization radius (\AA)', labelpad=1)
plt.ylabel(r'observable error (multiple units)')

ax1.text(3.5e-2,0.5e-7,r'(a)')
ax2.text(3.5e-2,0.5e-7,r'(b)')
ax3.text(12.5,0.18,r'(c)')
ax4.text(12.5,0.18,r'(d)')
ax5.text(55.5,2.5e-5,r'(e)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig5.pdf",bbox_inches='tight',pad_inches=0.01)
