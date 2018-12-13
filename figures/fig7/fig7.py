import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator

# switch to Physical Review compatible font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb,txfontsb}']
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'roman'
matplotlib.rcParams['font.size'] = 8

# preconditioning data
nshell, linear, squares, random, perturb, nonlinear = np.loadtxt("hi_local.txt",unpack=True)
nshell2, linear2, squares2, random2, perturb2, nonlinear2 = np.loadtxt("hi21.txt",unpack=True)

fig = plt.figure(figsize=(3.37,4.0))

ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.ylim(1e-4,1e0)
plt.xlim(0.0,20.0)

ax1.semilogy(nshell+1,squares,'ws',ms=5,mec='gray',clip_on=False,zorder=10)
ax1.semilogy(nshell+1,linear,'ks',ms=3,clip_on=False,zorder=10)
ax1.semilogy(nshell+1,random,'ro',ms=3,clip_on=False,zorder=10)
ax1.semilogy(nshell+1,perturb,'gx',ms=4,clip_on=False,zorder=10)
ax1.semilogy(nshell+1,nonlinear,'b+',ms=5,clip_on=False,zorder=10)

dy = 0.5
plt.plot(1.5,5e-3*dy**0,'ks',ms=4)
plt.plot(1.5,5e-3*dy**1,'ws',ms=5,mec='gray')
plt.plot(1.5,5e-3*dy**2,'gx',ms=4)
plt.plot(1.5,5e-3*dy**3,'b+',ms=5)
plt.plot(1.5,5e-3*dy**4,'ro',ms=3)

plt.text(2.1,4.2e-3*dy**0,r'no self-energy')
plt.text(2.1,4.2e-3*dy**1,r'least-squares self-energy')
plt.text(2.1,4.2e-3*dy**2,r'perturbative self-energy')
plt.text(2.1,4.2e-3*dy**3,r'optimized self-energy')
plt.text(2.1,4.2e-3*dy**4,r'randomized coarse-graining')

plt.xlabel(r'\# of shells')
plt.ylabel(r'local error')

ax2 = plt.subplot2grid((2, 1), (1, 0), colspan=1, rowspan=1)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.ylim(1e-7,1e-2)
plt.xlim(0,20)

ax2.semilogy(nshell2,squares2,'ws',ms=5,mec='gray',clip_on=False,zorder=10)
ax2.semilogy(nshell2,linear2,'ks',ms=3,clip_on=False,zorder=10)
ax2.semilogy(nshell2,random2,'ro',ms=3,clip_on=False,zorder=10)
ax2.semilogy(nshell2,perturb2,'gx',ms=4,clip_on=False,zorder=10)
ax2.semilogy(nshell2,nonlinear2,'b+',ms=5,clip_on=False,zorder=10)

plt.xlabel(r'shell \#')
plt.ylabel(r'shell error')

ax1.text(18.0,2.6e-1,r'(a)')
ax2.text(18.0,4e-7,r'(b)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig7.pdf",bbox_inches='tight',pad_inches=0.01)
