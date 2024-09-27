import numpy as np
import scipy
import matplotlib.pyplot as plt

# Function to calculate the overlap of a walker with the trial wave function
def calculate_overlap( psi1, psi2 ):
    if ( len(psi1.shape) == len(psi2.shape)  ):
        return np.einsum("...MFAi,...MFAi->...", psi1.conj(), psi2 )
    else:
        return np.einsum("MFAi,...MFAi->...", psi1.conj(), psi2 )

def calculate_local_energy( walkers, E_MOL, MU_MOL, WC, A0 ):
    a   = np.diag( np.sqrt(np.arange(1,NFOCK)), k=1 )

    # El
    elocal  = np.einsum( "zMFAi,Ai,zMFAi->z", walkers.conj(), E_MOL, walkers, optimize=True ) # H_EL
    #print( "E_el", np.average(np.einsum( "zMFAi,Ai,zMFAi->z", walkers.conj(), E_MOL, walkers, optimize=True ),axis=0)  )
    # Ph
    elocal += np.einsum( "zMFAi,M,F,zMFAi->z", walkers.conj(), WC, np.arange(NFOCK) + 0.500, walkers, optimize=True ) # H_PH
    #print( "E_ph", np.average(np.einsum( "zMFAi,M,F,zMFAi->z", walkers.conj(), WC, np.arange(NFOCK) + 0.500, walkers, optimize=True ),axis=0)  )
    # El-Ph
    elocal += np.einsum( "M,zMFAi,Aij,FG,zMGAj->z", WC * A0, walkers.conj(), MU_MOL, a.T + a, walkers, optimize=True ) # H_EL-PH
    #print( "E_el-ph", np.average(np.einsum( "M,zMFAi,Aij,FG,zMGAj->z", WC * A0, walkers.conj(), MU_MOL, a.T + a, walkers, optimize=True ),axis=0)  )
    # 1e-DSE
    elocal  += np.einsum( "zMFAi,M,Aij,Ajk,zMFAk->z", walkers.conj(), WC * A0**2, MU_MOL, MU_MOL, walkers, optimize=True ) # H_DSE
    #print( "E_1e-DSE", np.average(np.einsum( "zMFAi,M,Aij,Ajk,zMFAk->z", walkers.conj(), WC * A0**2, MU_MOL, MU_MOL, walkers, optimize=True ),axis=0)  )
    
    """
    pQED in the many-body electronic state basis does not have a 4-index tensor for the 2e-DSE
    HOW DO WE CALCULATE THE 2e-DSE IN THE MANY-BODY ELECTRONIC STATE BASIS USING einsum ?
    THE BELOW CODE IS INCORRECT !!! IT DOES NOT MAKE SENSE TO WRITE FOUR WAVEFUNCTIONS IN THIS CONEXT, RIGHT ?
    """
    # 2e-DSE
    for MOL_A in range( NMOL ):
        for MOL_B in range( NMOL ):
            if ( MOL_A != MOL_B ):
                elocal += np.einsum( "zMFAi,zMFAj,M,Aij,Bkl,zMFBk,zMFBl->z", walkers.conj(), walkers.conj(), WC * A0**2, MU_MOL, MU_MOL, walkers, walkers, optimize=True )

    return np.real( elocal )

# # Hubbard-Stratonovich transformation
# def hubbard_stratonovich( MU_MOL, WC, A0, dt, nwalkers ):
#     NMOL, NEL, _ = MU_MOL.shape
#     if ( NMOL < 2 ):
#         return np.zeros( (nwalkers, NMOL, NEL, NEL) )

#     MU_FULL  = np.einsum( "Aij,Bkl->AikBjl", MU_MOL, MU_MOL)
#     for A in range( NMOL ):
#         MU_FULL[A,:,:,A,:,:] *= 0
#     MU_FULL  = MU_FULL.reshape( (NMOL*NEL**2,NMOL*NEL**2) )
    
#     H_MU     = np.einsum( "M,...->...", WC * A0**2, MU_FULL )
#     u,s,vt   = np.linalg.svd( H_MU )
#     Ltensor = np.einsum("i,ai->ai", np.sqrt(s), u ).T
#     Ltensor = Ltensor.reshape( (-1,NMOL,NEL,NEL) )

#     xi      = np.random.normal( 0.0, 1.0, size=(Ltensor.shape[0],nwalkers) )
#     B       = -1j * np.sqrt(dt) * np.einsum( "nz,nAij->zAij", xi, Ltensor )
#     return B

# Hubbard-Stratonovich transformation
def hubbard_stratonovich( MU_MOL, WC, A0, dt, nwalkers, MOL_A, MOL_B ):
    NMOL, NEL, _ = MU_MOL.shape

    MU_AB    = np.kron( MU_MOL[MOL_A,:,:], MU_MOL[MOL_B,:,:] )
    H_DSE    = np.einsum( "M,jk->jk", WC * A0**2, MU_AB )
    u,s,vt   = np.linalg.svd( H_DSE )
    Ltensor  = np.einsum("i,ai->ai", np.sqrt(s), u )#.T
    Ltensor  = Ltensor.reshape( (-1,NEL,NEL) )
    xi       = np.random.normal( 0.0, 1.0, size=(Ltensor.shape[0],nwalkers) )
    HS       = -1j * np.sqrt(dt) * np.einsum( "nz,nij->zij", xi, Ltensor )
    return HS


def propagate( walkers, E_MOL, MU_MOL, WC, A0, dt ):

    taylor_order = 6
    
    # Electronic Propagation
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("Ai,zMFAi->zMFAi", E_MOL, walkers) / (i + 1)
        walkers += temp

    # Photon Propagation
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("M,F,zMFAi->zMFAi", WC, np.arange(NFOCK), walkers) / (i + 1)
        walkers += temp

    # Bilinear Propagation
    a = np.diag( np.sqrt(np.arange(1,NFOCK)), k=1 )
    aT_p_a = a.T + a
    bilinear = np.einsum("M,FG,Aij->FGAij", WC * A0, aT_p_a, MU_MOL )
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("FGAij,zMGAj->zMFAi", bilinear, walkers) / (i + 1)
        walkers += temp

    # 1e-DSE Propagation
    DSE_1e  = np.einsum("M,Aij,Ajk->Aik", WC * A0**2, MU_MOL, MU_MOL )
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("Aij,zMFAj->zMFAi", DSE_1e, walkers) / (i + 1)
        walkers += temp

    # 2e-DSE Propagation
    HS      = hubbard_stratonovich( MU_MOL, WC, A0, dt, len(walkers), 0, 1 )
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = np.einsum("zij,zMFAj->zMFAi", HS, walkers) / (i + 1)
        walkers += temp

    # for MOL_A in range( NMOL ):
    #     for MOL_B in range( MOL_A+1, NMOL ):
    #         HS      = 2 * hubbard_stratonovich( MU_MOL, WC, A0, dt, len(walkers), MOL_A, MOL_B )
    #         temp    = walkers.copy()
    #         for i in range( taylor_order ):
    #             temp[:,:,:,MOL_A,:] = np.einsum("zij,zMFj->zMFi", HS, walkers[:,:,:,MOL_A,:]) / (i + 1)
    #             walkers += temp



    # 1e-DSE Propagation
    DSE_1e  = np.einsum("M,Aij,Ajk->Aik", WC * A0**2, MU_MOL, MU_MOL )
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("Aij,zMFAj->zMFAi", DSE_1e, walkers) / (i + 1)
        walkers += temp

    # Bilinear Propagation
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("FGAij,zMGAj->zMFAi", bilinear, walkers) / (i + 1)
        walkers += temp

    # Photon Propagation
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("M,F,zMFAi->zMFAi", WC, np.arange(NFOCK), walkers) / (i + 1)
        walkers += temp

    # Electronic Propagation
    temp    = walkers.copy()
    for i in range( taylor_order ):
        temp = -0.500 * dt * np.einsum("Ai,zMFAi->zMFAi", E_MOL, walkers) / (i + 1)
        walkers += temp

    return walkers

def update_weight( walkers, overlap_old, trial_wavefunc, weights, Eold, Enew, dt ):


    # Phaseless Approximation
    overlap_new  = calculate_overlap(walkers, walkers )
    ratios       = overlap_new / overlap_old
    angles       = np.angle(ratios)
    phases       = np.array([ np.max([ 0, np.cos(angle) ]) for angle in angles ])
    
    # Local Energy Formalism
    V_ave       = 0.5 * (Enew + Eold)
    PROB        = np.exp( -dt * (V_ave - np.average(V_ave)) )
    weights     = np.abs(weights) * PROB * np.abs(ratios) * phases

    return weights.real, overlap_new

# Function to perform AFQMC simulation
def afqmc_simulation(trial_wavefunc, E_MOL, MU_MOL, WC, A0, nsteps, dt, nwalkers):
    walkers       = np.array([trial_wavefunc] * nwalkers, dtype=np.complex128)
    weights       = np.ones( nwalkers )

    overlaps      = calculate_overlap(walkers, walkers)

    Eold          = calculate_local_energy( walkers, E_MOL, MU_MOL, WC, A0 )
    energy_trace = []
    for step in range(nsteps):
        energy_trace.append(Eold)
        #print( "Step %d of %d" % (step, nsteps), "<E> =", np.average(Eold) )
        walkers           = propagate(walkers, E_MOL, MU_MOL, WC, A0, dt)
        if ( step % 1 == 0 ):
            overlap = calculate_overlap( walkers, walkers )
            walkers = np.einsum("zMFAi,z->zMFAi", walkers, 1/np.sqrt(overlap))
        Enew              = calculate_local_energy( walkers, E_MOL, MU_MOL, WC, A0 )
        weights, overlaps = update_weight( walkers, overlaps, trial_wavefunc, weights, Eold, Enew, dt )
        energy            = np.sum(weights * Enew) / np.sum(weights)
        Eold              = Enew


    wfn = np.einsum( "z,z...->...", weights , walkers ) / np.sum(weights)
    return np.array( energy_trace ), wfn



def get_EXACT( E_MOL, MU_MOL, WC, A0, NMODE, NFOCK ):
    NMOL, NEL    = E_MOL.shape
    a = np.diag( np.sqrt(np.arange(1,NFOCK)), k=1 )

    H_exact = np.zeros( (NEL**NMOL * NFOCK**NMODE, NEL**NMOL * NFOCK**NMODE) )

    
    if ( NMOL == 1 ):
        H_exact += np.kron( np.diag(E_MOL[0,:]), np.eye(NFOCK) )
        H_exact += np.kron( np.eye(NEL), np.diag(WC[0]*(np.arange(NFOCK)+0.500)) )
        H_exact += WC[0] * A0[0]    * np.kron( MU_MOL[0,:,:] , a.T + a )
        H_exact += WC[0] * A0[0]**2 * np.kron( MU_MOL[0,:,:] @ MU_MOL[0,:,:] , np.eye(NFOCK) )

    elif ( NMOL == 2 ):
        H_EL = np.kron(np.diag(E_MOL[0,:]), np.eye(NEL) ) + np.kron(np.eye(NEL), np.diag(E_MOL[1,:]) )
        H_EL = np.kron( H_EL, np.eye(NFOCK) )
        H_PH = np.kron(np.eye(NEL), np.eye(NEL) )
        H_PH = np.kron( H_PH, np.diag(WC[0]*(np.arange(NFOCK)+0.500)) )
        H_EL_PH = np.kron( MU_MOL[0,:,:], np.eye(NEL) ) + np.kron( np.eye(NEL), MU_MOL[1,:,:] )
        H_EL_PH = WC[0] * A0[0] * np.kron( H_EL_PH, a.T + a )
        H_DSE   = np.kron( MU_MOL[0,:,:] @ MU_MOL[0,:,:], np.eye(NEL) ) + np.kron( np.eye(NEL), MU_MOL[1,:,:] @ MU_MOL[1,:,:] )
        #H_DSE  += 2 * np.kron( MU_MOL[0,:,:], MU_MOL[1,:,:] )
        H_DSE   = WC[0] * A0[0]**2 * np.kron( H_DSE, np.eye(NFOCK) )
        H_exact += H_EL + H_PH + H_EL_PH + H_DSE

        # H_exact  +=                        np.kron(np.diag(E_MOL[0,:]),            np.kron(np.eye(NEL),                    np.eye(NFOCK)))
        # H_exact  +=                        np.kron(np.eye(NEL),                    np.kron(np.diag(E_MOL[1,:]),            np.eye(NFOCK)))
        # H_exact  +=                        np.kron(np.eye(NEL),                    np.kron(np.eye(NEL),                    np.diag(WC[0]*(np.arange(NFOCK)+0.500))))
        # H_exact  +=     WC[0] * A0[0]    * np.kron(MU_MOL[0,:,:],                  np.kron(np.eye(NEL) ,                   a.T + a))
        # H_exact  +=     WC[0] * A0[0]    * np.kron(np.eye(NEL),                    np.kron(MU_MOL[1,:,:] ,                 a.T + a))
        # H_exact  +=     WC[0] * A0[0]**2 * np.kron(MU_MOL[0,:,:] @ MU_MOL[0,:,:],  np.kron(np.eye(NEL) ,                   np.eye(NFOCK)))
        # H_exact  +=     WC[0] * A0[0]**2 * np.kron(np.eye(NEL),                    np.kron(MU_MOL[1,:,:] @ MU_MOL[1,:,:] , np.eye(NFOCK)))
        # H_exact  += 2 * WC[0] * A0[0]**2 * np.kron(MU_MOL[0,:,:],                  np.kron(MU_MOL[1,:,:] ,                 np.eye(NFOCK)))

    E_EXACT, U_exact = np.linalg.eigh( H_exact )
    print( "E (Exact)", E_EXACT[:] )
    # U_GS = U_exact[:,0].reshape( (NMODE,NFOCK,NMOL,NEL) )
    # U_GS = np.einsum("AiMF->MFAi", U_GS)
    # U_GS = np.array([U_GS])
    # print( "E_L(U_exact)", calculate_local_energy( U_GS, E_MOL, MU_MOL, WC, A0 ) )
    print( "E_EL (exact):", U_exact[:,0] @ H_EL @ U_exact[:,0] )
    print( "E_PH (exact):", U_exact[:,0] @ H_PH @ U_exact[:,0] )
    print( "E_EL_PH (exact):", U_exact[:,0] @ H_EL_PH @ U_exact[:,0] )
    print( "E_DSE (exact):", U_exact[:,0] @ H_DSE @ U_exact[:,0] )
    return E_EXACT[0]


if ( __name__ == "__main__" ):

    A0_LIST = np.arange(0.0, 0.3, 0.1) # np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

    ENERGY_A0       = np.zeros( len(A0_LIST) )
    ENERGY_A0_EXACT = np.zeros( len(A0_LIST) )

    NMOL          = 2
    NEL           = 2
    NMODE         = 1
    NFOCK         = 2
    
    E_MOL         = np.zeros( (NMOL,NEL) )
    MU_MOL        = np.zeros( (NMOL,NEL,NEL) )
    
    for el in range(NEL):
        E_MOL[:,el] = el
    MU_MOL[:,:,:] = 1.0
    for MOL_A in range(NMOL):
        for el in range(NEL):
            MU_MOL[MOL_A,el,el] = 0.0

    for a0i,a0 in enumerate(A0_LIST):

        WC = np.array([1.0] * NMODE)
        A0 = np.array([a0] * NMODE)

        trial_wavefunc = np.zeros( (NMODE, NFOCK, NMOL, NEL), dtype=np.complex128 )
        trial_wavefunc[:,0,:,0] = 1.0 # Collective Ground State
        trial_wavefunc = trial_wavefunc / np.linalg.norm( trial_wavefunc )

        nwalkers = 500
        T        = 10.0
        dt       = 0.05
        nsteps   = int(T / dt) + 1
        time     = np.linspace(0, nsteps*dt, nsteps)
        EQ_TIME  = nsteps//4
        energy_trace, wfn = afqmc_simulation(trial_wavefunc, E_MOL, MU_MOL, WC, A0, nsteps, dt, nwalkers)
        PHOT   = np.real(np.einsum( "MFAi,F,MFAi->", wfn.conj(), np.arange(NFOCK), wfn ))
        print( "A0 = %1.3f\n\t<aTa> = %1.6f" % (A0[0], PHOT) )

        energy_trace = np.average( energy_trace, axis=1 )
        energy_ave   = np.average( energy_trace[EQ_TIME:] )
        ENERGY_A0[a0i] = energy_ave
        print( "\tAFQMC Energy: = %1.6f" % energy_ave )

        # Plotting energy convergence
        plt.plot( time, energy_trace, "-", c='black', lw=1, alpha=0.5)
        plt.plot( time, time*0 + energy_ave, "-", lw=4, label='$A_0$ = %1.2f a.u.' % a0 )
        if ( NMOL <= 2 and NMODE == 1 ):
            E_EXACT = get_EXACT( E_MOL, MU_MOL, WC, A0, NMODE, NFOCK )
            plt.plot( time, time*0 + E_EXACT, "--", lw=4, label='$A_0$ = %1.2f a.u. (Exact)' % a0 )
        ENERGY_A0_EXACT[a0i] = E_EXACT
        plt.legend()
        plt.xlabel('Projection Time (a.u.)', fontsize=15)
        plt.ylabel('Energy (a.u.)', fontsize=15)
        plt.tight_layout()
        plt.savefig("AFQMC.png", dpi=300)
    plt.clf()

    plt.plot( A0_LIST, ENERGY_A0, "-o", lw=2, label='AFQMC')
    plt.plot( A0_LIST, ENERGY_A0_EXACT, "-o", lw=2,  label='Exact')
    plt.xlabel('Coupling Strength, $A_0$ (a.u.)', fontsize=15)
    plt.ylabel('Energy (a.u.)', fontsize=15)
    plt.tight_layout()
    plt.savefig("AFQMC_A0_SCAN.png", dpi=300)
    plt.clf()