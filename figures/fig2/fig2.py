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

# chemical potential data
cnum, natom , mu_low, mu_mid, mu_high = np.loadtxt("mu.out",skiprows=1,unpack=True)

fig = plt.figure(figsize=(3.37,2.3))

plt.ylim(1.5,4.0)
plt.xlim(0.0,0.6)

def mu_model(T,N):
    return (2.975 - 1.4*(N**(-1/3))) + np.exp(-1.6/T) * (4.34 - 1.5*(N**(-1/3)))

n1 = 1.0e100
n2 = 1000.0
n3 = 1.0

plt.plot([n1**(-1/3),n2**(-1/3)],[mu_model(1.0,n1),mu_model(1.0,n2)],'-',color='orange',lw=0.75)
plt.plot([n1**(-1/3),n2**(-1/3)],[mu_model(0.3,n1),mu_model(0.3,n2)],'-',color='red',lw=0.75)
plt.plot([n1**(-1/3),n2**(-1/3)],[mu_model(0.03,n1),mu_model(0.03,n2)],'-',color='black',lw=0.75)

plt.plot([n2**(-1/3),n3**(-1/3)],[mu_model(1.0,n2),mu_model(1.0,n3)],':',color='orange',lw=0.75)
plt.plot([n2**(-1/3),n3**(-1/3)],[mu_model(0.3,n2),mu_model(0.3,n3)],':',color='red',lw=0.75)
plt.plot([n2**(-1/3),n3**(-1/3)],[mu_model(0.03,n2),mu_model(0.03,n3)],':',color='black',lw=0.75)

m1 = 's'
m2 = '^'
m3 = 'o'
plt.plot(natom**(-1/3),mu_high,m1,color='orange',ms=2,clip_on=False,zorder=10)
plt.plot(natom**(-1/3),mu_mid,m2,color='red',ms=2,clip_on=False,zorder=10)
plt.plot(natom**(-1/3),mu_low,m3,color='black',ms=2,clip_on=False,zorder=10)

plt.plot(0.05,2.25+0.05,marker=m1,color='orange',ms=3)
plt.plot(0.05,2.0+0.05,marker=m2,color='red',ms=3)
plt.plot(0.05,1.75+0.05,marker=m3,color='black',ms=3)

plt.text(0.075,2.25,r'$T = 1$ eV')
plt.text(0.075,2.0,r'$T = 0.3$ eV')
plt.text(0.075,1.75,r'$T = 0.03$ eV')

plt.xlabel(r'$N_{\mathrm{Cu}}^{-1/3}$')
plt.ylabel(r'$\mu$ (eV)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig2.pdf",bbox_inches='tight',pad_inches=0.01)
