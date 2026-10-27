# SPDX-License-Identifier: MIT
"""Small, deterministic, single-threaded checks. They support, not replace, proofs."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
import sys
import json
import math
import itertools
import warnings
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linprog, minimize, OptimizeWarning
from scipy.linalg import null_space
import psutil
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "verify" / "logs"
LOG.mkdir(parents=True, exist_ok=True)
OUT = []
RNG = np.random.default_rng(20261027)
TOL = 3e-9
COUNT = 0
NEG = 0

def say(s):
    print(s, flush=True)
    OUT.append(str(s))

def check(ok, message):
    global COUNT
    COUNT += 1
    if not bool(ok):
        raise AssertionError(message)

def close(x, y, message, tol=TOL):
    check(np.allclose(x, y, atol=tol, rtol=tol), message)

def negative(ok, message):
    global NEG
    check(not ok, "Negative control escaped: " + message)
    NEG += 1

def h(p):
    return sum(-float(t)*math.log2(float(t)) for t in (p, 1-p) if t > 1e-14)

def ent(r):
    e = np.linalg.eigvalsh((r+r.conj().T)/2)
    check(e.min() > -2e-8, "Entropy input is not positive")
    return float(sum(-x*math.log2(x) for x in e if x > 1e-13))

def state(v):
    return np.outer(v, v.conj())

def ptr(r, dims, keep):
    keep = list(keep)
    gone = [i for i in range(len(dims)) if i not in keep]
    perm = keep + gone + [i+len(dims) for i in keep+gone]
    a = math.prod(dims[i] for i in keep)
    b = math.prod(dims[i] for i in gone)
    return np.trace(r.reshape(dims+dims).transpose(perm).reshape(a,b,a,b), axis1=1, axis2=3)

def mi(r, da, db):
    return ent(ptr(r,[da,db],[0]))+ent(ptr(r,[da,db],[1]))-ent(r)

def unitary(d):
    a = RNG.normal(size=(d,d))+1j*RNG.normal(size=(d,d))
    q,r = np.linalg.qr(a)
    return q@np.diag(np.diag(r)/np.abs(np.diag(r)))

def dephase(r, dims, sub):
    ids = np.indices(dims).reshape(len(dims),-1)[sub]
    return r*(ids[:,None]==ids[None,:])

def lp(c, ell, eps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        r=linprog(c, A_ub=[ell], b_ub=[eps], A_eq=[np.ones(len(c))],
                  b_eq=[1], bounds=(0,None), method="highs-ds", options={"threads":1})
    check(r.success, "LP failed")
    return r

def erasure_chsh():
    I=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); Y=np.array([[0,-1j],[1j,0]])
    cnot=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    for p in np.linspace(0,.5,101):
        v=np.array([np.sqrt(1-p),0,0,np.sqrt(p)])
        a=np.diag([1-p,p]); r=state(v); c=np.diag([1-p,0,0,p]); prod=np.kron(a,a)
        E=h(p)
        rotate=np.eye(4)
        rotate[0,0]=rotate[3,3]=np.sqrt(1-p)
        rotate[0,3]=np.sqrt(p); rotate[3,0]=-np.sqrt(p)
        close(rotate@v,np.array([1,0,0,0]),"quantum held-erasure unitary")
        for rho,pen in ((r,2*E),(c,E),(prod,0)):
            close(ptr(rho,[2,2],[0]),a,"marginal A")
            close(ptr(rho,[2,2],[1]),a,"marginal B")
            close(mi(rho,2,2),pen,"erasure penalty")
            held=ent(rho)-E
            close(E-held,pen,"held/released subtraction")
        out=cnot@c@cnot.T
        close(ptr(out,[2,2],[1]),np.diag([1,0]),"CNOT erased A with control first")
        s=2*np.sqrt(p*(1-p)); b0=(Z+s*X)/np.sqrt(1+s*s); b1=(Z-s*X)/np.sqrt(1+s*s)
        val=float(np.trace(r@(np.kron(Z,b0+b1)+np.kron(X,b0-b1))).real)
        close(val,2*np.sqrt(1+s*s),"CHSH construction")
        corr=np.array([[np.trace(r@np.kron(a1,b2)).real for b2 in [X,Y,Z]] for a1 in [X,Y,Z]])
        eig=np.linalg.eigvalsh(corr.T@corr)
        close(val,2*np.sqrt(eig[-1]+eig[-2]),"CHSH singular-value optimum")
        p2=(1-np.sqrt(max(0,2-val*val/4)))/2
        close(2*h(p2),2*E,"CHSH inverse",tol=2e-7)
    negative(abs(mi(np.eye(4)/4,2,2)-2)<TOL,"product source assigned quantum penalty")
    say("PASS erasure and CHSH: 101 probabilities, all marginals and endpoints")

def magic_square():
    I=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); Y=np.array([[0,-1j],[1j,0]])
    sq=[[np.kron(X,I),np.kron(I,X),np.kron(X,X)],
        [np.kron(I,Z),np.kron(Z,I),np.kron(Z,Z)],
        [np.kron(X,Z),np.kron(Z,X),np.kron(Y,Y)]]
    eye=np.eye(4); phi=np.eye(4).reshape(-1)/2; rho=state(phi)
    rows=[]; cols=[]
    for idx in range(3):
        row=sq[idx]; col=[sq[k][idx] for k in range(3)]
        close(row[0]@row[1]@row[2],eye,"row parity")
        close(col[0]@col[1]@col[2],eye*(1 if idx<2 else -1),"column parity")
        for a,b in itertools.combinations(row,2): close(a@b,b@a,"row commute")
        for a,b in itertools.combinations(col,2): close(a@b,b@a,"col commute")
        def projectors(ops, parity):
            ans=[]
            for a,b in itertools.product([-1,1],repeat=2):
                proj=(eye+a*ops[0])@(eye+b*ops[1])/4
                close(proj@proj,proj,"projector")
                close(np.trace(proj),1,"rank-one basis")
                ans.append(((a,b,parity*a*b),proj))
            close(sum(v for _,v in ans),eye,"complete basis")
            return ans
        rows.append(projectors(row,1)); cols.append(projectors(col,1 if idx<2 else -1))
    for x,y in itertools.product(range(3),repeat=2):
        win=0.; total=0.
        for (aa,P),(bb,Q) in itertools.product(rows[x],cols[y]):
            prob=float(np.trace(rho@np.kron(P,Q.T)).real)
            check(prob>=-TOL,"negative magic probability")
            total+=prob
            if aa[y]==bb[x]: win+=prob
        close(total,1,"magic normalization"); close(win,1,"magic wins")
    close(ent(ptr(rho,[4,4],[0]))*2,4,"magic lower certificate")
    # Entropy budget for the thermal working registers in the text's SWAP construction.
    # The SWAP channel itself is not simulated here.
    close(ent(np.eye(4)/4)*2,4,"magic working-register entropy budget")
    negative(np.allclose(sq[0][2]@sq[1][2]@sq[2][2],eye),"wrong last-column parity")
    say("PASS magic square: nine contexts, all raw outcomes, rank-one bases; working-register entropy budget 4")

def recovery():
    for n in range(7):
        for q in range(9):
            for c in range(9):
                costs=[2*(n-a-b)+b for a in range(min(q,n)+1) for b in range(min(c,n-a)+1)]
                best=min(costs)
                formula=max(n-q,0)+max(n-q-c,0)
                close(best,formula,"integer recovery allocation")
    # Independently check the entropy effect of actual transfer and dephasing on Phi_(2^n).
    for n in range(1,4):
        d=2**n; phi=np.eye(d).reshape(-1)/np.sqrt(d); source=state(phi)
        for a in range(n+1):
            for b in range(n-a+1):
                dims=[2]*(2*n); rho=source.copy()
                for j in range(a,a+b): rho=dephase(rho,dims,j)
                keep=list(range(a+b,n))
                other=list(range(n,2*n))+list(range(a+b))
                work=ent(ptr(rho,dims,keep))+ent(ptr(rho,dims,other))
                close(work,2*(n-a-b)+b,"explicit quantum/classical recovery entropy")
    for E,q,c in [(2.75,.5,.75),(1.125,.75,2),(3.5,2,1.25),(.3,.7,0)]:
        # Allocation of entanglement mass to local, dephased and quantum paths.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",OptimizeWarning)
            res=linprog([2,1,0],A_ub=[[0,0,1],[0,1,0]],b_ub=[q,c],
                        A_eq=[[1,1,1]],b_eq=[E],bounds=(0,None),method="highs-ds",options={"threads":1})
        check(res.success,"fractional return LP")
        close(res.fun,max(E-q,0)+max(E-q-c,0),"fractional return cost")
    # Fixed arithmetic sanity assertions; they do not simulate a faulty protocol.
    negative(2==max(2,0)+max(2-1,0),"one classical bit saves two units (arithmetic)")
    negative(0==max(2,0)+max(2,0),"unbudgeted quantum reunion (arithmetic)")
    say("PASS recovery: 567 integer budgets, explicit density matrices through three Bell pairs, fractional LPs")

def sectors():
    for trial in range(60):
        dims=RNG.integers(1,40,size=9); costs=np.log2(dims); ell=RNG.uniform(.0001,1,size=9)
        eps=RNG.uniform(ell.min(),ell.max())
        result=lp(costs,ell,eps)
        vertices=[]
        for i in range(9):
            if ell[i]<=eps+TOL:
                p=np.zeros(9); p[i]=1; vertices.append(p)
        for i,j in itertools.combinations(range(9),2):
            if abs(ell[i]-ell[j])>TOL:
                t=(eps-ell[j])/(ell[i]-ell[j])
                if -TOL<=t<=1+TOL:
                    p=np.zeros(9); p[i]=t; p[j]=1-t; vertices.append(p)
        best=min(np.dot(p,costs) for p in vertices)
        close(best,result.fun,"two-sector LP")
        dual_loss=-float(result.ineqlin.marginals[0])
        witness=min(costs+dual_loss*ell)-dual_loss*eps
        close(witness,best,"dual witness")
        minmi=min(sum(-x*math.log2(x) for x in p if x>1e-13)+2*np.dot(p,costs) for p in vertices)
        check(2*best-TOL<=minmi<=2*best+1+TOL,"sector MI sandwich")
        for p in vertices:
            mix=.3*p+.7*vertices[0]
            obj=lambda x:sum(-y*math.log2(y) for y in x if y>1e-13)+2*np.dot(x,costs)
            check(obj(mix)+TOL>=.3*obj(p)+.7*obj(vertices[0]),"MI concavity")
    # A flagged two-sector core and its coherent precursor.
    p=.37; da=5; v=np.zeros(da*da); v[0]=np.sqrt(p/2); v[6]=np.sqrt(p/2)
    for j in range(2,5): v[j*da+j]=np.sqrt((1-p)/3)
    pure=state(v); core=pure.copy(); labels=[0,0,1,1,1]
    mask=np.array([labels[i] for i in range(da) for _ in range(da)])
    core=core*(mask[:,None]==mask[None,:])
    cost=p*math.log2(2)+(1-p)*math.log2(3)
    close(mi(core,da,da),h(p)+2*cost,"flagged core mutual information")
    check(mi(core,da,da)<=mi(pure,da,da)+TOL,"local sector measurement data processing")
    say("PASS sectors: 60 LP menus, two-support optima, dual certificates, concavity and coherent core")

def access():
    w=np.zeros(8); w[[1,2,4]]=1/np.sqrt(3); rho=state(w)
    ab=ptr(rho,[2,2,2],[0,1]); bc=ptr(rho,[2,2,2],[1,2]); b=ptr(rho,[2,2,2],[1])
    close(ab,bc,"W marginals")
    e,vec=np.linalg.eigh(ab); kernel=vec[:,e<1e-10]
    obstruction=np.kron(kernel@kernel.T,np.eye(2))+np.kron(np.eye(2),kernel@kernel.T)
    inter=null_space(obstruction)
    check(inter.shape[1]==2,"W support intersection dimension")
    zero=np.zeros(8); zero[0]=1
    close(inter@inter.T,state(zero)+rho,"W support intersection span")
    close(ent(ab)+ent(bc)-ent(b)-ent(rho),h(1/3),"W conditional MI")
    close(np.trace(ab@state(np.array([0,1,1,0])/np.sqrt(2))),2/3,"W uniqueness weight")
    for d in [2,3,7,16,32]:
        t=.8; s=np.ones(d)/np.sqrt(d); V=np.eye(d)+(np.exp(1j*t)-1)*state(s)
        r=np.zeros((d,d)); r[0,0]=1
        close(np.linalg.norm(V@r-r@V)**2,4*(1-np.cos(t))*(d-1)/d**2,"fixed-control obstruction")
        U=np.diag(np.exp(2j*np.pi*np.arange(d)/d))
        # The diagonal commutant of U intersects that of V only in the scalars.
        mat=np.column_stack([(V@np.diag(np.eye(d)[j])-np.diag(np.eye(d)[j])@V).reshape(-1) for j in range(d)])
        check(np.linalg.matrix_rank(mat,tol=1e-9)==d-1,"scalar commutant")
    # Sanity assertion: the staged surcharge h2(1/3) is nonzero.
    negative(abs(h(1/3))<TOL,"zero marginal-completion cost equals staged cost")
    say("PASS access: W uniqueness support and positive surcharge; fixed controls through dimension 32")

def weighted_and_multiphase():
    # Convex hull of three full-rank, noncommuting observers.
    for d in [2,3,4,7]:
        probes=[]
        for j in range(3):
            Q=unitary(d); vals=RNG.uniform(.1,1,size=d); vals/=sum(vals)
            probes.append(Q@np.diag(vals)@Q.conj().T)
        def objective(p): return np.trace(np.linalg.inv(sum(x*r for x,r in zip(p,probes)))).real
        res=minimize(objective,np.ones(3)/3,method="SLSQP",bounds=[(0,1)]*3,
                     constraints={"type":"eq","fun":lambda p:sum(p)-1},options={"ftol":1e-12,"maxiter":300})
        check(res.success,"observer optimization")
        sigma=sum(x*r for x,r in zip(res.x,probes)); inv=np.linalg.inv(sigma); S=np.trace(inv).real
        alpha=np.sqrt(2); q=np.exp(2j*np.pi*alpha); delta=np.angle(q**d); H=delta*inv/S
        close(np.trace(H),delta,"weighted trace constraint")
        costs=[np.trace(r@H@H).real for r in probes]
        close(max(costs),delta**2/S,"observer KKT optimum",tol=1e-6)
        eig,vec=np.linalg.eigh(H); W=vec@np.diag(np.exp(1j*eig)/q)@vec.conj().T
        we,Sv=np.linalg.eig(W); phases=np.angle(we); phases[-1]-=sum(phases)
        a=np.ones(d,dtype=complex)
        for j in range(1,d): a[j]=a[j-1]*np.exp(1j*phases[j])
        T=np.roll(np.eye(d),1,axis=0); A=np.diag(a)
        U=Sv@T.conj().T@Sv.conj().T; V=Sv@A.conj().T@Sv.conj().T
        close(q*V.conj().T@U.conj().T@V@U,vec@np.diag(np.exp(1j*eig))@vec.conj().T,"commutator realization",tol=2e-6)
    for m in [1,2,3]:
        theta=2**(1/(m+1)); alphas=theta**np.arange(1,m+1)
        for d in range(1,41):
            p=np.rint(d*alphas).astype(int); ds=np.abs(d*alphas-p)
            ell=4*sum(np.sin(np.pi*ds/d)**2); T=np.roll(np.eye(d),1,axis=0)
            actual=0
            for aj,pj in zip(alphas,p):
                U=np.diag(np.exp(2j*np.pi*pj*np.arange(d)/d))
                actual+=np.linalg.norm(U@T-np.exp(2j*np.pi*aj)*T@U)**2/d
            close(actual,ell,"multiphase clock optimum")
            check(np.sqrt(ell)+TOL>=4*max(ds)/d,"multiphase sine lower bound")
            M=np.zeros((m+1,m+1),dtype=np.int64)
            for j in range(m): M[j,j+1]=1
            M[m,0]=2; v=np.r_[d,p]; cols=[v]
            for j in range(m): cols.append(M@cols[-1])
            det=np.linalg.det(np.column_stack(cols))
            check(abs(det-round(det))<1e-5 and abs(round(det))>=1,"nonzero integer determinant")
    say("PASS weighted observers and multiphase: noncommuting probes, explicit commutators, 120 clock dimensions")

def reservoir():
    # Identity is tested for independent non-Gibbs system states, Gibbs bath and arbitrary unitaries.
    for trial in range(30):
        sysvals=RNG.uniform(.1,1,size=2); sysvals/=sum(sysvals); r=np.diag(sysvals)
        energies=np.array([0.,.7]); g=np.exp(-energies); g/=sum(g); gamma=np.diag(g)
        u=unitary(4); eta=u@np.kron(r,gamma)@u.conj().T
        rf=ptr(eta,[2,2],[0]); ef=ptr(eta,[2,2],[1])
        qb=np.dot(energies,np.diag(ef).real-g)/math.log(2)
        A=-ent(eta)+ent(rf)-np.trace(ef@np.diag(np.log2(g))).real
        close(qb,ent(r)-ent(rf)+A,"reservoir work identity")
        der=-ent(ef)-np.trace(ef@np.diag(np.log2(g))).real
        close(A,mi(eta,2,2)+der,"reservoir decomposition")
        check(A>=-TOL,"negative entropy production")
    say("PASS finite-reservoir entropy-production identity: 30 arbitrary unitaries")

def potential_and_continuity():
    # Pure states only: for pure ACB states E_R is the marginal entropy on either side of a cut.
    # No relative entropy of entanglement of a mixed state is evaluated.
    for trial in range(50):
        vec=RNG.normal(size=8)+1j*RNG.normal(size=8); vec/=np.linalg.norm(vec); r=state(vec)
        before=ent(ptr(r,[2,2,2],[0,1]))
        after=ent(ptr(r,[2,2,2],[0]))
        check(before-after<=1+TOL,"teleportation cut decrease bound")
        check(after-before<=1+TOL,"reverse quantum cut bound")
    # Extraction-to-hashing continuity bound, independent of a score-to-extraction theorem.
    phi=np.eye(4).reshape(-1)/2; target=state(phi)
    for noise in np.linspace(0,.5,26):
        Q=unitary(16); weights=RNG.uniform(0,1,size=16); weights/=sum(weights)
        other=Q@np.diag(weights)@Q.conj().T; omega=(1-noise)*target+noise*other
        t=float(np.linalg.svd(omega-target,compute_uv=False).sum()/2)
        bob=ptr(omega,[4,4],[1]); lb=2-t*math.log2(3)-h(t); ub=t*math.log2(15)+h(t)
        check(ent(bob)+TOL>=lb,"robust Bob entropy continuity")
        check(ent(omega)<=ub+TOL,"robust joint entropy continuity")
        check(2*(ent(bob)-ent(omega))+TOL>=4-2*t*math.log2(45)-4*h(t),"hashing-work conversion")
    say("PASS pure-state cut entropies and finite-error conversion: 50 pure cut tests, 26 extracted-state tests")

def main():
    say("The heat price of precision -- finite verification")
    say("Python "+sys.version.split()[0]+"; NumPy "+np.__version__+"; SciPy "+scipy.__version__)
    say("Platform "+platform.platform()+"; seed 20261027; tolerance "+str(TOL))
    with threadpool_limits(limits=1):
        say("Numerical pools "+json.dumps(threadpool_info()))
        for fn in [erasure_chsh,magic_square,recovery,sectors,access,weighted_and_multiphase,reservoir,potential_and_continuity]:
            fn()
            mem=psutil.Process().memory_info()
            peak=getattr(mem,"peak_wset",mem.rss)
            check(peak<4*1024**3,"Memory ceiling exceeded")
        mem=psutil.Process().memory_info()
        say("Peak working set MiB: "+str(round(getattr(mem,"peak_wset",mem.rss)/2**20,2)))
        say("PASS assertions: "+str(COUNT)+"; negative controls caught: "+str(NEG))
        say("These finite checks do not replace the written proofs or verify imported asymptotic theorems.")
    (LOG/"checks.txt").write_text("\n".join(OUT)+"\n",encoding="utf-8")

if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT.append(traceback.format_exc())
        (LOG/"checks.txt").write_text("\n".join(OUT)+"\n",encoding="utf-8")
        raise
