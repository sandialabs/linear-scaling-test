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

# benchmark data
index1, natom1, time1, mem1 = np.loadtxt("solver1.txt",unpack=True)
index2, natom2, time_hi2, time_mid2, time_lo2, mem2 = np.loadtxt("solver2.txt",unpack=True)
index6, natom6, time_hi6, time_mid6, time_lo6, mem6 = np.loadtxt("solver6.txt",unpack=True)
index8, natom8, time_hi8, time_mid8, time_lo8, mem8 = np.loadtxt("solver8.txt",unpack=True)

#fig = plt.figure(figsize=(6.3,8))
fig = plt.figure(figsize=(6.85,8.65))

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)

plt.ylim(1e-2,1e6)
plt.xlim(1,1e6)

ax1.loglog(natom1,time1,'ks',ms=3)
ax1.loglog(natom2,time_hi2,'ro',ms=3)
ax1.loglog(natom6,time_hi6,'g+',ms=5,mew=0.75)
ax1.loglog(natom8,time_hi8,'bx',ms=4,mew=0.75)

plt.xlabel(r'\# of atoms', labelpad=2)
plt.ylabel(r'wall time (s)', labelpad=1)

ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)

plt.ylim(1e-2,1e6)
plt.xlim(1,1e6)

ax2.loglog(natom1,time1,'ks',ms=3)
ax2.loglog(natom2,time_mid2,'ro',ms=3)
ax2.loglog(natom6,time_mid6,'g+',ms=5,mew=0.75)
ax2.loglog(natom8,time_mid8,'bx',ms=4,mew=0.75)

plt.xlabel(r'\# of atoms', labelpad=2)
plt.ylabel(r'wall time (s)', labelpad=1)

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)

plt.ylim(1e-2,1e6)
plt.xlim(1,1e6)

ax3.loglog(natom1,time1,'ks',ms=3)
ax3.loglog(natom2,time_lo2,'ro',ms=3)
ax3.loglog(natom6,time_lo6,'g+',ms=5,mew=0.75)
ax3.loglog(natom8,time_lo8,'bx',ms=4,mew=0.75)

plt.xlabel(r'\# of atoms', labelpad=2)
plt.ylabel(r'wall time (s)', labelpad=1)

ax4 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)

plt.ylim(1e4,1e8)
plt.xlim(1,1e6)

ax4.loglog(natom1,mem1,'ks',ms=3)
ax4.loglog(natom2,mem2,'ro',ms=3)
ax4.loglog(natom6,mem6,'g+',ms=5,mew=0.75)
ax4.loglog(natom8,mem8,'bx',ms=4,mew=0.75)

plt.xlabel(r'\# of atoms', labelpad=1)
plt.ylabel(r'memory (kB)', labelpad=3)

ax1.loglog([1,1e6],[48*60*60,48*60*60],'k--',lw=1)
ax2.loglog([1,1e6],[48*60*60,48*60*60],'k--',lw=1)
ax3.loglog([1,1e6],[48*60*60,48*60*60],'k--',lw=1)
ax4.loglog([1,1e6],[64e6,64e6],'k--',lw=1)

x = np.empty(101)
for i in range(101):
    x[i] = 10.0**(0.06*i)

ax1.loglog(x,1.5e-7*x**3,'k-',lw=0.75,zorder=0)
ax2.loglog(x,1.5e-7*x**3,'k-',lw=0.75,zorder=0)
ax3.loglog(x,1.5e-7*x**3,'k-',lw=0.75,zorder=0)
ax4.loglog(x,1.5*x**2,'k-',lw=0.75,zorder=0)

ax1.loglog(x,0.3e-4*x**2,'r-',lw=0.75,zorder=0)
ax2.loglog(x,0.4e-4*x**2,'r-',lw=0.75,zorder=0)
ax3.loglog(x,0.6e-4*x**2,'r-',lw=0.75,zorder=0)
ax4.loglog(x,3e2*x**(4.0/3.0),'r-',lw=0.75,zorder=0)

ax1.loglog(x,0.17*x,'g-',lw=0.75,zorder=0)
ax2.loglog(x,3.2*x,'g-',lw=0.75,zorder=0)
ax3.loglog(x,536.0*x,'g-',lw=0.75,zorder=0)
ax4.loglog(x,3.3e2*x,'g-',lw=0.75,zorder=0)

ax1.text(1e4,4,r'$T = 1$ eV')
ax2.text(1e4,4,r'$T = 0.3$ eV')
ax3.text(1e4,4,r'$T = 0.03$ eV')

ax1.text(1e3,0.5,r'\# of pole pairs = 3')
ax2.text(1e3,0.5,r'\# of pole pairs = 4')
ax3.text(1e3,0.5,r'\# of pole pairs = 6')

ax1.text(1e3,0.25,r'localization radius = 8 $\AA$')
ax2.text(1e3,0.25,r'localization radius = 14 $\AA$')
ax3.text(1e3,0.25,r'localization radius = 38 $\AA$')

ax1.text(1e3,0.125,r'coloring radius = 12 $\AA$')
ax2.text(1e3,0.125,r'coloring radius = 22 $\AA$')
ax3.text(1e3,0.125,r'coloring radius = 96 $\AA$')

ax1.text(4.0,0.03e6,r'(a)')
ax2.text(4.0,0.03e6,r'(b)')
ax3.text(4.0,0.03e6,r'(c)')
ax4.text(4.0,np.sqrt(0.03)*1e8,r'(d)')

x0 = 0.3e1
x1 = 0.5e1
y0 = 4.2e3
y1 = 3.57e3
dy = 0.5
ax1.plot(x0,y0,'ks',ms=4)
ax1.plot(x0,y0*dy,'ro',ms=4)
ax1.plot(x0,y0*dy**2,'g+',ms=5)
ax1.plot(x0,y0*dy**3,'bx',ms=5)

ax1.text(x1,y1,r'full diagonalization')
ax1.text(x1,y1*dy,r'selected inversion')
ax1.text(x1,y1*dy**2,r'localized trace')
ax1.text(x1,y1*dy**3,r'randomized trace')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig6.pdf",bbox_inches='tight',pad_inches=0.01)
