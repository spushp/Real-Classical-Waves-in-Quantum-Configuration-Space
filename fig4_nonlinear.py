# Fig. 4: cubic Klein-Gordon (real classical wave) vs nonlinear Schrodinger envelope
import numpy as np
from common import plt, panel
c=1.0; w0=1.0; hbar=1.0; eta=1.0   # Kerr coefficient eta
Nx=1024; Lp=400.0; dx=Lp/Nx
x=np.arange(Nx)*dx-Lp/2; k=2*np.pi*np.fft.fftfreq(Nx,dx)
Om=np.sqrt(w0**2+c**2*k**2)
F=np.fft.fft; IF=np.fft.ifft
wdt=20.0
def run(a,T=300.0,dt=0.05,rec_center=False):
    chi0=a*np.exp(-x**2/(2*wdt**2)).astype(complex)
    Psi0=IF(np.sqrt(2*hbar/Om)*F(chi0))
    u=Psi0.real.copy(); v=IF(Om*F(Psi0.imag)).real
    def acc(u): return -IF(Om**2*F(u)).real-eta*u**3
    n=int(round(T/dt)); rec=[]
    for i in range(n):
        if rec_center: rec.append(u[Nx//2]+1j*IF(F(v)/Om).real[Nx//2])
        k1u=v; k1v=acc(u)
        k2u=v+dt/2*k1v; k2v=acc(u+dt/2*k1u)
        k3u=v+dt/2*k2v; k3v=acc(u+dt/2*k2u)
        k4u=v+dt*k3v; k4v=acc(u+dt*k3u)
        u=u+dt/6*(k1u+2*k2u+2*k3u+k4u); v=v+dt/6*(k1v+2*k2v+2*k3v+k4v)
    Psi=u+1j*IF(F(v)/Om).real
    phi=IF(np.sqrt(Om/(2*hbar))*F(Psi)); chi_full=phi*np.exp(1j*w0*T)
    # NLS by split-step (Strang)
    g=3*eta*hbar/(4*w0**2)   # g/hbar
    chi=chi0.copy(); h=0.05; m=int(round(T/h))
    lin=np.exp(-1j*(c**2*k**2/(2*w0))*h/2)
    for i in range(m):
        chi=IF(lin*F(chi)); chi=chi*np.exp(-1j*g*abs(chi)**2*h); chi=IF(lin*F(chi))
    return chi0,chi_full,chi,np.array(rec)
amps=np.array([0.05,0.1,0.15,0.2,0.3,0.4])
errs=[];sav=None
for a in amps:
    chi0,cf,cn,_=run(a)
    e=np.linalg.norm(cf-cn)/np.linalg.norm(cn); errs.append(e); print(a,e)
    if abs(a-0.2)<1e-9: sav=(chi0,cf,cn)
errs=np.array(errs)
# wrong-orientation spectral weight at the packet centre (short run with record)
wrong=[]
amps2=np.array([0.05,0.1,0.2,0.3,0.4])
for a in amps2:
    *_,rec=run(a,T=200.0,rec_center=True)
    Sp=abs(np.fft.fft(rec*np.hanning(len(rec))))**2
    fr=np.fft.fftfreq(len(rec),0.05)
    wrong.append(Sp[fr>0].sum()/Sp.sum())   # numpy kernel e^{-i w t}: e^{-i w0 t} sits at fr<0
wrong=np.array(wrong); print("wrong",wrong)
# Duffing single oscillator frequency shift
A=np.linspace(0.05,0.6,10); wnum=[]
for Aa in A:
    q,p=Aa,0.0; h=0.002; ts=[];tprev=0; 
    def f(q,p): return p,-w0**2*q-eta*q**3
    t=0; cross=[]
    for i in range(int(200/h)):
        k1=f(q,p);k2=f(q+h/2*k1[0],p+h/2*k1[1]);k3=f(q+h/2*k2[0],p+h/2*k2[1]);k4=f(q+h*k3[0],p+h*k3[1])
        qn=q+h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); p=p+h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        if q>0 and qn<=0: cross.append(t+h*q/(q-qn))
        q=qn; t+=h
    per=np.mean(np.diff(cross)); wnum.append(2*np.pi/per)
wnum=np.array(wnum)
fig,ax=plt.subplots(2,2,figsize=(7.0,4.0))
ax=ax.ravel()
chi0,cf,cn=sav
ax[0].plot(x,abs(chi0),color="0.6",label=r"$|\chi(x,0)|$")
ax[0].plot(x,abs(cf),color="C0",label=r"cubic Klein--Gordon (real $u$), $t=300$")
ax[0].plot(x,abs(cn),"--",color="C3",label=r"nonlinear Schr\"odinger, $t=300$".replace('\\"o','ö'))
ax[0].set_xlim(-160,160); ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"envelope $|\chi|$"); ax[0].legend(frameon=False,fontsize=6,loc="upper right"); panel(ax[0],"(a)")
ax[0].set_ylim(0,0.31)
ax[1].loglog(amps,errs,"o-",ms=3,color="C0",label=r"$\|\chi_{\rm KG}-\chi_{\rm NLS}\|/\|\chi_{\rm NLS}\|$ at $t=300$")
ax[1].loglog(amps,errs[1]*(amps/amps[1])**2,":",color="k",label=r"$\propto a^2$")
ax[1].loglog(amps,errs[-2]*(amps/amps[-2])**4,"--",color="0.5",label=r"$\propto a^4$")
ax[1].set_xlabel(r"envelope amplitude $a$"); ax[1].set_ylabel("relative error"); ax[1].set_ylim(1e-4,8); ax[1].legend(frameon=False,fontsize=6); panel(ax[1],"(b)")
ax[2].plot(A,(wnum-w0)/w0,"o",ms=3,color="C0",label="numerical (Duffing)")
ax[2].plot(A,3*eta*A**2/(8*w0**2),"-",color="C3",label=r"$3\eta|a|^2/8\omega^2$")
ax[2].set_xlabel(r"oscillation amplitude $|a|$"); ax[2].set_ylabel(r"$(\omega_{\rm NL}-\omega)/\omega$"); ax[2].legend(frameon=False,fontsize=6); panel(ax[2],"(c)")
ax[3].loglog(amps2,wrong,"o-",ms=3,color="C0",label=r"wrong-orientation weight of $\Psi$")
ax[3].loglog(amps2,wrong[1]*(amps2/amps2[1])**4,":",color="k",label=r"$\propto a^4$")
ax[3].set_xlabel(r"envelope amplitude $a$"); ax[3].set_ylabel(r"fraction of power at $e^{+i\omega t}$"); ax[3].legend(frameon=False,fontsize=6); panel(ax[3],"(d)")
from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
for a_ in (ax[1],ax[3]):
    a_.xaxis.set_major_locator(FixedLocator([0.05,0.1,0.2,0.4])); a_.xaxis.set_major_formatter(FixedFormatter(["0.05","0.1","0.2","0.4"]))
    a_.xaxis.set_minor_formatter(NullFormatter())
plt.tight_layout(pad=0.3); plt.savefig("fig4_nonlinear.pdf"); plt.savefig("fig4_nonlinear.png",dpi=160)
np.savez("fig4.npz",amps=amps,errs=errs,amps2=amps2,wrong=wrong,A=A,wnum=wnum)
