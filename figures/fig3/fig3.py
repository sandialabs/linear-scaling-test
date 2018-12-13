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
ncore, tprepost , tblas3, tlapack, tpexsi, tsparse = np.loadtxt("strong-scaling.txt",skiprows=1,unpack=True)

fig = plt.figure(figsize=(2.2,3.0))

plt.ylim(10.0,3.0e5)
plt.xlim(0.1,100.0)

plt.loglog(ncore,tlapack[0]/ncore,'-r',lw=1)
plt.loglog(ncore,tsparse[0]/ncore,'-g',lw=1)
plt.loglog(ncore,tpexsi[0]/ncore,'-b',lw=1)
plt.loglog(ncore,tblas3[0]/ncore,'-c',lw=1)
plt.loglog(ncore,tprepost[0]/ncore,'-k',lw=1)

plt.loglog(ncore,tlapack,'^r',ms=3)
plt.loglog(ncore,tsparse,'og',ms=3)
plt.loglog(ncore,tpexsi,'sb',ms=3)
plt.loglog(ncore,tblas3,'+c',ms=4)
plt.loglog(ncore,tprepost,'vk',ms=3)

plt.plot(0.4,35*1.9**4,'^r',ms=4)
plt.plot(0.4,35*1.9**3,'og',ms=4)
plt.plot(0.4,35*1.9**2,'sb',ms=4)
plt.plot(0.4,35*1.9,'+c',ms=5)
plt.plot(0.4,35,'vk',ms=4)

plt.text(0.8,30*1.9**4,r'LAPACK')
plt.text(0.8,30*1.9**3,r'block-sparse')
plt.text(0.8,30*1.9**2,r'PEXSI')
plt.text(0.8,30*1.9,r'level-3 BLAS')
plt.text(0.8,30,r'pre/post processing')

plt.xlabel(r'\# of cores')
plt.ylabel(r'wall time (s)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig3.pdf",bbox_inches='tight',pad_inches=0.01)
