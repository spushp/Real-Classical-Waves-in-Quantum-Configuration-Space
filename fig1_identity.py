# Fig. 1: key identity H_t u = -Omega^{-1} d_t u, energy isometry, and action/energy conservation
import numpy as np
from scipy.signal import hilbert
from common import plt, panel
hbar=1.0; c=1.0
Nx=400; Lx=10.0
x=np.linspace(-Lx,Lx,Nx+2)[1:-1]; dx=x[1]-x[0]
w0sq=1.0+1.5/np.cosh(x)**2+0.8*np.exp(-(x-4)**2)          # inhomogeneous cutoff, gap gamma=1
D2=(np.diag(-2*np.ones(Nx))+np.diag(np.ones(Nx-1),1)+np.diag(np.ones(Nx-1),-1))/dx**2
Om2=-c**2*D2+np.diag(w0sq)
lam,V=np.linalg.eigh(Om2); om=np.sqrt(lam)
fOm=lambda f,vec: V@(f(om)*(V.T@vec))
u0=np.exp(-(x+2)**2)*1.0+0.6*np.exp(-(x-1.5)**2/0.5)*np.cos(3*x)
u1=0.8*np.exp(-(x+0.5)**2/2)*np.sin(2*x)
# keep only modes below omega=6 (smooth data, resolved by time sampling)
P=lambda v: V@((om<6)*(V.T@v))
u0=P(u0); u1=P(u1)
a0=V.T@u0; a1=V.T@u1
def field(t):  # u(t), u_t(t) for array t  (modal exact evolution)
    t=np.atleast_1d(t)[:,None]
    um=np.cos(om*t)*a0+np.sin(om*t)/om*a1
    vm=-om*np.sin(om*t)*a0+np.cos(om*t)*a1
    return um@V.T, vm@V.T
# exact identity at t=0
Hex=-fOm(lambda w:1/w,u1)       # -Omega^{-1} u_t(0)
idx=np.arange(Nx)   # relative L2 discrepancy over the whole field
dt=0.05
Ts=np.array([25,50,100,200,400,800,1600])
errs=[]
for T in Ts:
    t=np.arange(-T/2,T/2+dt/2,dt)
    U,_=field(t)
    H=np.imag(hilbert(U[:,idx],axis=0))     # H[cos]=sin convention
    k0=np.argmin(abs(t))
    errs.append(np.linalg.norm(H[k0]-Hex[idx])/np.linalg.norm(Hex[idx]))
errs=np.array(errs); print("errs",errs)
# trace for panel (a)
T=400; t=np.arange(-T/2,T/2+dt/2,dt); j=np.argmin(abs(x-(-1.0)))
U,Ut=field(t)
Hnum=np.imag(hilbert(U[:,j]))
Hloc=np.array([-(V@((V.T@Ut[k])/om))[j] for k in range(len(t))])
# energy isometry: E[H u] = E[u]
E=lambda uu,vv:0.5*(np.sum(vv**2)+np.sum(fOm(lambda w:w,uu)**2))*dx
Hu0=-fOm(lambda w:1/w,u1); Hu1=fOm(lambda w:w,u0)   # Cauchy data of H_t u
print("energy isometry rel diff", abs(E(Hu0,Hu1)-E(u0,u1))/E(u0,u1))
print("H^2 = -1 rel diff", np.linalg.norm(-fOm(lambda w:1/w,Hu1)+u0)/np.linalg.norm(u0))
# RK4 integration of the real second-order equation; conservation of J and E, and ||phi||^2 = J/hbar
def rhs(y):
    uu,vv=y; return np.array([vv,-Om2@uu])
y=np.array([u0,u1]); h=0.01; nst=20000; rec=[]
def diag(uu,vv):
    Ev=E(uu,vv)
    Jv=0.5*(uu@fOm(lambda w:w,uu)+vv@fOm(lambda w:1/w,vv))*dx
    phi=(fOm(lambda w:w,uu)+1j*vv); phi=fOm(lambda w:1/np.sqrt(2*hbar*w),phi.real)+1j*fOm(lambda w:1/np.sqrt(2*hbar*w),phi.imag)
    nphi=np.sum(abs(phi)**2)*dx
    Hphi=hbar*(fOm(lambda w:w,phi.real)+1j*fOm(lambda w:w,phi.imag))
    EH=np.real(np.sum(np.conj(phi)*Hphi)*dx)
    return Ev,Jv,nphi,EH
E0,J0,_,_=diag(*y)
ts=[];dE=[];dJ=[];dN=[];dEH=[]
for n in range(nst+1):
    if n%200==0:
        Ev,Jv,nphi,EH=diag(*y); ts.append(n*h)
        dE.append(abs(Ev-E0)/E0); dJ.append(abs(Jv-J0)/J0); dN.append(abs(hbar*nphi-Jv)/Jv); dEH.append(abs(EH-Ev)/Ev)
    if n<nst:
        k1=rhs(y);k2=rhs(y+h/2*k1);k3=rhs(y+h/2*k2);k4=rhs(y+h*k3); y=y+h/6*(k1+2*k2+2*k3+k4)
print("max drift E,J",max(dE),max(dJ),"identities",max(dN),max(dEH))
fig,ax=plt.subplots(1,3,figsize=(7.0,2.1))
m=(t>-15)&(t<15)
ax[0].plot(t[m],U[m,j],color="0.6",lw=0.8,label=r"$u$")
ax[0].plot(t[m],Hnum[m],color="C0",label=r"$\mathcal{H}_t u$ (FFT, time record)")
ax[0].plot(t[m],Hloc[m],"--",color="C3",label=r"$-\Omega^{-1}\partial_t u$ (one instant)")
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"field at $x=-1$"); ax[0].legend(loc="lower left",frameon=False); panel(ax[0],"(a)")
ax[0].set_ylim(-1.6,1.4)
ax[1].loglog(Ts,errs,"o-",color="C0",ms=3,label="FFT record vs instant")
ax[1].loglog(Ts,errs[0]*Ts[0]/Ts,":",color="k",label=r"$\propto T_{\rm rec}^{-1}$")
ax[1].set_xlabel(r"record length $T_{\rm rec}$"); ax[1].set_ylabel(r"relative error of $\mathcal{H}_t u$")
ax[1].legend(frameon=False,loc="upper right"); ax[1].set_ylim(1.2e-5,3e-2); panel(ax[1],"(b)")
ts=np.array(ts)
ax[2].semilogy(ts,np.array(dE)+1e-17,color="C0",label=r"$|E(t)-E(0)|/E$")
ax[2].semilogy(ts,np.array(dJ)+1e-17,color="C1",label=r"$|J(t)-J(0)|/J$")
ax[2].semilogy(ts,np.array(dN)+1e-17,color="C2",label=r"$|\hbar\|\phi\|^2-J|/J$")
ax[2].semilogy(ts,np.array(dEH)+1e-17,"--",color="C3",label=r"$|\langle\phi,\hat H\phi\rangle-E|/E$")
ax[2].set_xlabel(r"$t$"); ax[2].set_ylim(1e-17,1e-2); ax[2].legend(frameon=False,loc="center left",bbox_to_anchor=(0.06,0.43),ncol=2,fontsize=5.5,columnspacing=0.5,handlelength=1.3,handletextpad=0.4); panel(ax[2],"(c)")
ax[2].yaxis.set_label_position("right")
plt.tight_layout(pad=0.3); plt.savefig("fig1_identity.pdf"); plt.savefig("fig1_identity.png",dpi=160)
np.savez("fig1.npz",Ts=Ts,errs=errs)
