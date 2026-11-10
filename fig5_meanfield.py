# Fig. 5: real classical oscillators <-> DNLS mean field <-> N-quantum linear dynamics, N = J/hbar
import numpy as np
from scipy.linalg import sqrtm, eigh
from scipy.special import gammaln
from common import plt, panel
w0=10.0; eps=1.0; Jact=1.0; lam=0.15
eta=4*lam*w0**2/(3*Jact)   # Kerr coefficient from lambda = 3 eta J/(4 w0^2)
Om2=np.array([[w0**2,-eps],[-eps,w0**2]]); Om=np.real(sqrtm(Om2)); Omi=np.linalg.inv(Om)
Om12=np.real(sqrtm(Om)); Omm12=np.linalg.inv(Om12)
z0=0.6; th=0.5*np.arccos(z0); psi0=np.array([np.cos(th),np.sin(th)],complex)
T=200.0; tgrid=np.linspace(0,T,801)
# --- classical real oscillators: q'' + Om^2 q + eta q^3 = 0
Psi=np.sqrt(2*Jact)*Omm12@psi0
q=Psi.real.copy(); v=Om@Psi.imag
acc=lambda q: -Om2@q-eta*q**3
h=0.001; zc=[]; j=0; tt=0.0
nst=int(round(T/h))
def zof(q,v):
    ps=Om12@(q+1j*(Omi@v)); p=abs(ps)**2; return (p[0]-p[1])/p.sum(), ps/np.sqrt(p.sum())
zc=[zof(q,v)[0]]; ts_c=[0.0]
for i in range(1,nst+1):
    k1q=v;k1v=acc(q);k2q=v+h/2*k1v;k2v=acc(q+h/2*k1q);k3q=v+h/2*k2v;k3v=acc(q+h/2*k2q);k4q=v+h*k3v;k4v=acc(q+h*k3q)
    q=q+h/6*(k1q+2*k2q+2*k3q+k4q); v=v+h/6*(k1v+2*k2v+2*k3v+k4v)
    if i%int(round((T/800)/h))==0: zc.append(zof(q,v)[0]); ts_c.append(i*h)
zc=np.array(zc); ts_c=np.array(ts_c)
# --- DNLS mean field: i psi' = Om psi + lam |psi|^2 psi
def dnls_rhs(p): return -1j*(Om@p+lam*abs(p)**2*p)
p=psi0.copy(); hm=0.005; mf=[p.copy()]
for i in range(1,int(round(T/hm))+1):
    k1=dnls_rhs(p);k2=dnls_rhs(p+hm/2*k1);k3=dnls_rhs(p+hm/2*k2);k4=dnls_rhs(p+hm*k3); p=p+hm/6*(k1+2*k2+2*k3+k4)
    if i%int(round((T/800)/hm))==0: mf.append(p.copy())
mf=np.array(mf); zmf=abs(mf[:,0])**2-abs(mf[:,1])**2
# --- N-quantum dynamics
def coh(N,ps):
    n=np.arange(N+1)
    lc=0.5*(gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1))
    with np.errstate(divide="ignore"):
        lp=n*np.log(abs(ps[0])+1e-300)+(N-n)*np.log(abs(ps[1])+1e-300)
    ph=n*np.angle(ps[0])+(N-n)*np.angle(ps[1])
    return np.exp(lc+lp)*np.exp(1j*ph)
def quantum(N,times):
    n=np.arange(N+1)
    d=Om[0,0]*n+Om[1,1]*(N-n)+(lam/N)*(n*(n-1)/2+(N-n)*(N-n-1)/2)
    off=Om[0,1]*np.sqrt((n[:-1]+1)*(N-n[:-1]))
    H=np.diag(d)+np.diag(off,1)+np.diag(off,-1)
    E,W=eigh(H); c0=W.T@coh(N,psi0)
    out=[]
    for t in times:
        c=W@(np.exp(-1j*E*t)*c0)
        n1=np.sum(abs(c)**2*n); a12=np.sum(np.conj(c[1:])*c[:-1]*np.sqrt((n[:-1]+1)*(N-n[:-1])))  # <a1^dag a2>
        out.append((c,n1,a12))
    return out
Ns=[10,40,160,640]
zq={}
idx=np.arange(len(tgrid))
for N in Ns:
    res=quantum(N,tgrid); zq[N]=np.array([(2*r[1]-N)/N for r in res])
# trace-norm error and fidelity vs N at fixed times
Nlist=np.array([10,20,40,80,160,320,640,1280])
tchk=[25.0,50.0,100.0]
trn={t:[] for t in tchk}; fid={t:[] for t in tchk}
for N in Nlist:
    res=quantum(N,tchk)
    for t,(c,n1,a12) in zip(tchk,res):
        k=int(round(t/(T/800))); pm=mf[k]
        gam=np.array([[n1,np.conj(a12)],[a12,N-n1]])/N   # gamma_{ss'} = <a_s'^dag a_s>/N
        D=gam-np.outer(pm,np.conj(pm)); trn[t].append(np.sum(abs(np.linalg.eigvalsh(D))))
        fid[t].append(abs(np.vdot(coh(N,pm),c))**2)
for t in tchk: print(t,"trace",np.round(trn[t],5),"fid",np.round(fid[t],4))
print("max |z_cl - z_mf|:",np.max(abs(zc-zmf)))
fig,ax=plt.subplots(1,3,figsize=(7.0,2.2),gridspec_kw={"width_ratios":[1.5,1,1]})
ax[0].plot(ts_c,zc,color="k",lw=1.4,label="real oscillators")
ax[0].plot(tgrid,zmf,"--",color="C3",lw=1.0,label="mean field")
for N,col in zip(Ns,["C9","C2","C1","C0"]):
    ax[0].plot(tgrid,zq[N],color=col,lw=0.7,alpha=0.9,label=rf"$N={N}$")
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"imbalance $z(t)$"); ax[0].legend(frameon=False,fontsize=5.3,ncol=3,loc="lower left",columnspacing=0.7,handlelength=1.3,handletextpad=0.4); panel(ax[0],"(a)")
ax[0].set_ylim(-1.3,1.0)
for t,col in zip(tchk,["C0","C1","C3"]):
    ax[1].loglog(Nlist,trn[t],"o-",ms=2.5,color=col,label=rf"$t={t:.0f}$")
ax[1].loglog(Nlist,trn[50.0][2]*Nlist[2]/Nlist,":",color="k",label=r"$\propto N^{-1}$")
ax[1].set_xlabel(r"$N=J/\hbar$"); ax[1].set_ylabel(r"trace-norm error of $\gamma_N^{(1)}(t)$"); ax[1].legend(frameon=False,fontsize=5.8); panel(ax[1],"(b)")
for t,col in zip(tchk,["C0","C1","C3"]):
    ax[2].semilogx(Nlist,fid[t],"o-",ms=2.5,color=col,label=rf"$t={t:.0f}$")
ax[2].set_xlabel(r"$N=J/\hbar$"); ax[2].set_ylabel("overlap with product state"); ax[2].set_ylim(0,1.05); ax[2].legend(frameon=False,fontsize=5.8,loc="lower left"); panel(ax[2],"(c)")
plt.tight_layout(pad=0.3); plt.savefig("fig5_meanfield.pdf"); plt.savefig("fig5_meanfield.png",dpi=160)
