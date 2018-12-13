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

# preconditioning data
pre_radius, pre_time1 , pre_error1 = np.loadtxt("preconditioner.txt",unpack=True)
pre_niter, pre_time2 , pre_error2 = np.loadtxt("preconditioner2.txt",unpack=True)
pre_niter, pre_time3 , pre_error3 = np.loadtxt("preconditioner3.txt",unpack=True)

# polynomial/rational efficiency data
poly_T, poly_order, poly_error, poly_time = np.loadtxt("polynomial.txt",unpack=True)
rat_T, rat_order, rat_error, rat_time = np.loadtxt("rational.txt",unpack=True)

# random ensemble data
random_num1, random_err1 = np.loadtxt("random1.txt",unpack=True)
random_num2, random_err2 = np.loadtxt("random2.txt",unpack=True)
random_num3, random_err3 = np.loadtxt("random3.txt",unpack=True)
random_num4, random_err4 = np.loadtxt("random4.txt",unpack=True)
random_num5, random_err5 = np.loadtxt("random5.txt",unpack=True)
random_num6, random_err6 = np.loadtxt("random6.txt",unpack=True)
random_num7, random_err7 = np.loadtxt("random7.txt",unpack=True)
random_num8, random_err8 = np.loadtxt("random8.txt",unpack=True)

fig = plt.figure(figsize=(3.37,5.5))

ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=1)

plt.ylim(1e-11,1e-1)
plt.xlim(0.0,1.5)

ax1.semilogy(pre_time2,pre_error2,'ks',ms=3)
ax1.semilogy(pre_time3,pre_error3,'g^',ms=3)
ax1.semilogy(pre_time1,pre_error1,'ro',ms=3)

plt.plot(0.7,0.8*1e-2,'ks',ms=4)
plt.plot(0.7,0.64*1e-3,'g^',ms=4)
plt.plot(0.7,0.512*1e-4,'ro',ms=4)

plt.text(0.77,0.8*0.6e-2,r'unpreconditioned CG solve')
plt.text(0.77,0.64*0.6e-3,r'preconditioned CG solve')
plt.text(0.77,0.512*0.6e-4,r'sparse approximate inverse')

plt.xlabel(r'solver time (s)', labelpad=2)
plt.ylabel(r'residual error',labelpad=1)

ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1, rowspan=1)

plt.ylim(1.0e-9,0.1)
plt.xlim(1.0e1,1.0e4)

ax2.loglog(rat_time[0:5],rat_error[0:5],'s',ms=3.5,mec='black',color='white',mew=0.75)
ax2.loglog(rat_time[5:11],rat_error[5:11],'s',ms=3.5,mec='black',color='white',mew=0.75)
ax2.loglog(rat_time[5:11],rat_error[5:11],'k.',ms=0.75)
ax2.loglog(rat_time[11:],rat_error[11:],'ks',ms=3)
ax2.loglog(poly_time[0:7],poly_error[0:7],'o',ms=3.5,mec='red',color='white',mew=0.75)
ax2.loglog(poly_time[7:14],poly_error[7:14],'o',ms=3.5,mec='red',color='white',mew=0.75)
ax2.loglog(poly_time[7:14],poly_error[7:14],'r.',ms=0.75)
ax2.loglog(poly_time[14:],poly_error[14:],'ro',ms=3)

plt.plot(0.95e3,2.0*1e-2,'s',ms=4,mec='black',color='white',mew=1.0)
plt.plot(0.95e3,3.0*1e-3,'o',ms=4,mec='red',color='white',mew=1.0)

plt.plot(1.25e3,2.0*1e-2,'s',ms=4,mec='black',color='white',mew=1.0)
plt.plot(1.25e3,3.0*1e-3,'o',ms=4,mec='red',color='white',mew=1.0)
plt.plot(1.25e3,2.0*1e-2,'k.',ms=1)
plt.plot(1.25e3,3.0*1e-3,'r.',ms=1)

plt.plot(1.65e3,2.0*1e-2,'ks',ms=4)
plt.plot(1.65e3,3.0*1e-3,'ro',ms=4)

plt.plot(1.65e3,2.0*1e-2,'ks',ms=4)
plt.plot(1.65e3,3.0*1e-3,'ro',ms=4)

plt.text(2.2e3,2.0*0.7e-2,r'rational')
plt.text(2.2e3,3.0*0.7e-3,r'polynomial')

plt.xlabel(r'solver time (s)', labelpad=1)
plt.ylabel(r'error tolerance')

plt.text(3.0e1,1e-5,r'$T = 1$ eV',rotation=-60)
plt.text(2.2e2,1e-5,r'$T = 0.3$ eV',rotation=-60)
plt.text(2.0e3,1e-5,r'$T = 0.03$ eV',rotation=-60)

ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1)

plt.ylim(1.0e-3,3.0e1)
plt.xlim(1.0,1.0e6)

ax3.loglog(9*random_num8,random_err8,'ks',ms=3)
ax3.loglog(9*random_num1,random_err1,'ro',ms=3)
ax3.loglog(9*random_num2,random_err2,'x',color='orange',ms=4)
ax3.loglog(9*random_num3,random_err3,'^',color='yellow',ms=3,mec='black',mew=0.75)
ax3.loglog(9*random_num4,random_err4,'<',color='green',ms=3)
ax3.loglog(9*random_num5,random_err5,'>',color='blue',ms=3)
ax3.loglog(9*random_num6,random_err6,'v',color='indigo',ms=3)
ax3.loglog(9*random_num7,random_err7,'+',color='violet',ms=4)

xshift = 0.25*9
xshift2 = 0.2*9
yshift = 0.75
plt.text(random_num1[0]*xshift,random_err1[0]*yshift,r'0 \AA')
plt.text(random_num2[0]*xshift,random_err2[0]*yshift,r'3 \AA')
plt.text(random_num3[0]*xshift,random_err3[0]*yshift,r'4 \AA')
plt.text(random_num4[0]*xshift,random_err4[0]*yshift,r'5 \AA')
plt.text(random_num5[0]*xshift,random_err5[0]*yshift,r'7 \AA')
plt.text(random_num6[0]*xshift,random_err6[0]*yshift,r'9 \AA')
plt.text(random_num7[0]*xshift2,random_err7[0]*yshift,r'12 \AA')
plt.text(random_num8[0]*xshift2,random_err8[0]*yshift,r'18 \AA')
plt.text(6.3,0.3,r'coloring radius',rotation=-47)

plt.xlabel(r'\# of random vectors', labelpad=1)
plt.ylabel(r'RMS force errors (eV/\AA)')

ax1.text(0.08,1e-10,r'(a)')
ax2.text(14.5,0.85e-8,r'(b)')
ax3.text(2.0,0.3e-2,r'(c)')

plt.tight_layout(pad=0.0, w_pad=0.2, h_pad=0.0)
plt.savefig("../fig4.pdf",bbox_inches='tight',pad_inches=0.01)
