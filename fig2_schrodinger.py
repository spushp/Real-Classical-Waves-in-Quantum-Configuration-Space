# Fig. 2: narrowband Schrodinger limit in a parabolic graded-index guide
import numpy as np
from common import plt, panel
hbar=1.0; c=1.0; w0=400.0; nu=0.5
Nx=900; Lx=1.1
x=np.linspace(-Lx,Lx,Nx+2)[1:-1]; dx=x[1]-x[0]
D2=(np.diag(-2*np.ones(Nx))+np.diag(np.ones(Nx-1),1)+np.diag(np.ones(Nx-1),-1))/dx**2
L=-c**2*D2+np.diag(w0**2*nu**2*x**2/c**2)
ell,V=np.linalg.eigh(L)
print("ell_n/(w0 nu) first:",(ell[:6]/(w0*nu)).round(4))
om_ex=np.sqrt(w0**2+ell); om_S=w0+ell/(2*w0)
# displaced ground state (coherent-like), projected onto modes n<=nmax
xd=0.2; ellosc=np.sqrt(c**2/(w0*nu))
g=np.exp(-(x-xd)**2/(2*ellosc**2)).astype(complex)
cfull=V.T@g
ts=np.logspace(-2,3.5,200)
res={}
for nmax in [8,14,22]:
    rng=np.random.default_rng(nmax)
    cn=np.zeros(len(ell),complex); cn[:nmax+1]=np.exp(2j*np.pi*rng.random(nmax+1)); cn/=np.linalg.norm(cn)
    Lam=ell[nmax]
    err=np.array([np.linalg.norm(cn*(np.exp(-1j*om_ex*t)-np.exp(-1j*om_S*t))) for t in ts])
    res[nmax]=(Lam,err)
    print(nmax,Lam, "max ratio err/bound (t<=tsat):", np.max(err[ts<1]/(Lam**2*ts[ts<1]/(8*w0**3))))
fig,ax=plt.subplots(1,3,figsize=(7.0,2.15))
cols=["C0","C1","C3"]
for (nmax,(Lam,err)),col in zip(res.items(),cols):
    ax[0].loglog(ts,err,color=col,label=rf"$n_{{\max}}={nmax}$")
    ax[0].loglog(ts,np.minimum(Lam**2*ts/(8*w0**3),2),":",color=col)
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"$\|e^{-i\hat Ht}\phi_0-e^{-i\hat H_St}\phi_0\|$")
ax[0].plot([],[],":",color="0.3",label=r"bound $\Lambda^2t/8\bar\omega_0^3$")
ax[0].legend(frameon=False,loc="lower right",title="uniform weight up to "+r"$n_{\max}$",title_fontsize=6); ax[0].set_ylim(1e-9,8); panel(ax[0],"(a)")
n=np.arange(30)
ax[1].plot(n,om_ex[:30],"o",ms=3,color="C0",label=r"exact $\omega_n=(\bar\omega_0^2+\ell_n)^{1/2}$")
ax[1].plot(n,om_S[:30],"-",color="C3",label="Schrödinger "+r"$\bar\omega_0+E_n/\hbar$")
ax[1].set_xlabel(r"mode index $n$"); ax[1].set_ylabel(r"mode frequency"); panel(ax[1],"(b)")
ax[1].legend(frameon=False,loc="upper left"); ax[1].set_ylim(399.5,418.5)
ins=ax[1].inset_axes([0.6,0.1,0.37,0.3])
ins.semilogy(n[1:],abs(om_ex[1:30]-om_S[1:30]),"o",ms=2,color="C0")
ins.semilogy(n[1:],ell[1:30]**2/(8*w0**3),"-",color="k",lw=0.7)
ins.tick_params(labelsize=5); ins.set_xlabel(r"$n$",fontsize=5,labelpad=0)
# real-wave reconstruction
nmax=14; cn=cfull.copy(); cn[nmax+1:]=0; cn/=np.linalg.norm(cn)
t=np.pi/(2*nu)
phi_ex=V@(cn*np.exp(-1j*om_ex*t))
chiS=V@(cn*np.exp(-1j*om_S*t))*np.exp(1j*w0*t)
u_ex=np.real(V@(np.sqrt(2*hbar/om_ex)*(V.T@phi_ex)))   # u = Re[(2 hbar/Omega)^{1/2} phi]
u_S=np.sqrt(2*hbar/w0)*np.abs(chiS)*np.cos(w0*t-np.angle(chiS))
env=np.sqrt(2*hbar/w0)*np.abs(chiS)
print("reconstruction rel err",np.linalg.norm(u_ex-u_S)/np.linalg.norm(u_ex))
ax[2].fill_between(x,-env,env,color="0.88",lw=0)
ax[2].plot(x,u_ex,color="C0",label=r"exact real wave $u$")
ax[2].plot(x,u_S,"--",color="C3",label=r"$\sqrt{2\hbar/\bar\omega_0}\,|\chi|\cos(\bar\omega_0t-\arg\chi)$")
ax[2].set_xlim(-0.3,0.3); ax[2].set_xlabel(r"$x$"); ax[2].set_ylabel(r"$u(x,t=\pi/2\nu)$"); panel(ax[2],"(c)")
ax[2].legend(frameon=False,loc="lower left",fontsize=5.6); ax[2].set_ylim(-0.016,0.016)
plt.tight_layout(pad=0.3); plt.savefig("fig2_schrodinger.pdf"); plt.savefig("fig2_schrodinger.png",dpi=160)
chiE=phi_ex*np.exp(1j*w0*t)
print("chi rel err",np.linalg.norm(chiE-chiS)/np.linalg.norm(chiE))
