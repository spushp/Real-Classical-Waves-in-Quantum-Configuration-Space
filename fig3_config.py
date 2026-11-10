# Fig. 3: two-quantum configuration-space wave: UQM pair in each time, cross terms, sum-frequency structure
import numpy as np
from scipy.signal import hilbert
from common import plt, panel
c=1.0; w0=2.0; hbar=1.0
Nx=128; Lp=40.0
x=np.arange(Nx)*Lp/Nx-Lp/2; k=2*np.pi*np.fft.fftfreq(Nx,Lp/Nx)
Om=np.sqrt(w0**2+c**2*k**2)
phi0=np.exp(-x**2/(2*2.0**2))*np.exp(1j*1.0*x); phi0/=np.sqrt(np.sum(abs(phi0)**2)*Lp/Nx)
ph0=np.fft.fft(phi0)
def phi(t):  # one-quantum wave function phi(x,t) (normalized), shape (len(t),Nx)
    t=np.atleast_1d(t)[:,None]; return np.fft.ifft(ph0*np.exp(-1j*Om*t),axis=1)
def uplus(t):  # u^(+) = (hbar/2Omega)^{1/2} phi
    t=np.atleast_1d(t)[:,None]; return np.fft.ifft(ph0*np.sqrt(hbar/(2*Om))*np.exp(-1j*Om*t),axis=1)
# (a) equal-time configuration-space wave U2 = Re Phi2
ta=3.0; p=phi(ta)[0]; Phi2=np.outer(p,p); U2=Phi2.real
# (b) UQM pair in t1 at fixed (x1,x2,t2), and in equal-time t
i1=np.argmin(abs(x-1.0)); i2=np.argmin(abs(x+2.0)); t2=0.7
dt=0.02; T=600; t=np.arange(-T/2,T/2,dt)
P=phi(t); s1=P[:,i1]*phi(t2)[0,i2]        # multi-time, varying t1
s_eq=P[:,i1]*P[:,i2]                         # equal time
H1=np.imag(hilbert(s1.real)); Heq=np.imag(hilbert(s_eq.real))
mid=(abs(t)<T/4)
r1=np.linalg.norm(s1.imag[mid]+H1[mid])/np.linalg.norm(s1.imag[mid])
req=np.linalg.norm(s_eq.imag[mid]+Heq[mid])/np.linalg.norm(s_eq.imag[mid])
print("UQM residual multi-time t1:",r1,"equal-time:",req)
# second-order equations: exact generator (Om1+Om2)^2 vs naive Om1^2+Om2^2 (in Fourier space)
Ph2k=np.outer(ph0,ph0); K1,K2=np.meshgrid(Om,Om,indexing="ij")
naive=np.linalg.norm(((K1+K2)**2-(K1**2+K2**2))*Ph2k)/np.linalg.norm((K1+K2)**2*Ph2k)
print("relative residual of naive 3n-dim wave eq:",naive)
# (c) spectra at a point: ordinary product u(x1)u(x2) vs UQM wave U2=Re Phi2 (scaled to joint amplitude 2Re A2)
Up=uplus(t); u=2*Up.real
prod=u[:,i1]*u[:,i2]; A2=Up[:,i1]*Up[:,i2]
res28=np.max(abs(prod-(2*(A2).real+2*(Up[:,i1]*np.conj(Up[:,i2])).real)))
print("sum/difference decomposition, max abs residual:",res28)
win=np.hanning(len(t))
f=2*np.pi*np.fft.rfftfreq(len(t),dt)
S=lambda s: abs(np.fft.rfft(s*win))**2
fig,ax=plt.subplots(1,3,figsize=(7.0,2.2))
ext=[x[0],x[-1],x[0],x[-1]]
im=ax[0].imshow(U2.T,origin="lower",extent=ext,cmap="RdBu_r",vmin=-abs(U2).max(),vmax=abs(U2).max())
ax[0].set_xlabel(r"$x_1$"); ax[0].set_ylabel(r"$x_2$"); ax[0].set_xlim(-10,14); ax[0].set_ylim(-10,14)
ax[0].plot([-10,14],[-10,14],":",color="k",lw=0.6)
panel(ax[0],"(a)"); ax[0].text(0.04,0.93,r"$U_2=\mathrm{Re}\,\Phi_2$, $t=3$",transform=ax[0].transAxes,fontsize=6.5,va="top")
m=(t>0)&(t<8)
ax[1].plot(t[m],s1.real[m],color="0.6",lw=0.8,label=r"$\mathrm{Re}\,\Phi_2$")
ax[1].plot(t[m],s1.imag[m],color="C0",label=r"$\mathrm{Im}\,\Phi_2$")
ax[1].plot(t[m],-H1[m],"--",color="C3",label=r"$-\mathcal{H}_{t_1}\mathrm{Re}\,\Phi_2$")
ax[1].set_xlabel(r"$t_1$  ($t_2$ fixed)"); ax[1].set_ylabel(r"$\Phi_2(x_1,t_1;x_2,t_2)$")
ax[1].legend(frameon=False,loc="lower center",ncol=3,fontsize=5.6,columnspacing=0.6,handlelength=1.4,handletextpad=0.4); panel(ax[1],"(b)")
yl=abs(s1.real[m]).max()*1.25; ax[1].set_ylim(-1.35*yl,yl)
mm=f<6
nrm=S(prod)[mm].max()
ax[2].semilogy(f[mm],S(prod)[mm]/nrm+1e-12,color="0.5",label=r"$u(x_1,t)\,u(x_2,t)$")
ax[2].semilogy(f[mm],S(2*A2.real)[mm]/nrm+1e-12,"--",color="C3",label=r"$2\,\mathrm{Re}\,A_2$ (UQM)")
ax[2].axvline(2*w0,color="k",lw=0.5,ls=":")
ax[2].set_xlabel(r"temporal frequency $\omega$"); ax[2].set_ylabel("power (norm.)"); ax[2].set_ylim(1e-8,1e5)
ax[2].legend(frameon=False,loc="upper left",fontsize=6); panel(ax[2],"(c)")
ax[2].text(1.5,1e-4,"difference\nfrequencies",fontsize=6,ha="left"); ax[2].text(4.15,3e0,"sum\nfrequencies",fontsize=6,va="bottom")
plt.tight_layout(pad=0.3); plt.savefig("fig3_config.pdf"); plt.savefig("fig3_config.png",dpi=160)
