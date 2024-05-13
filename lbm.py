from time import time as ti
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import fromfunction
from numpy import pi
import numpy as np

from torch import zeros,tensor,roll,sin,sqrt,linspace
from torch import sum as tsum
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# lbm Functions
#---------------------------------------------------------------------------------------------------------------------------
def lbm_sim(bdry, obstacle, nx, ny, Re, uLB=0.04, C=62, img=False, t_img=500, map_color='jet', 
            coefs=True, tF_start=15000, tF_out=100, F_num=50):
    maxIter = tF_start + F_num*tF_out  # Number of iterations
    ly = ny-1                          # Height of the domain in lattice units
    a = ny/9                           # Geometric factor
    nulb = uLB*a/Re                    # Viscosity in lattice units
    omega = 1 / (3*nulb+0.5)           # Relaxation frequency = 1/tau

    # lattice velocities
    v = tensor([ 
                [1,1],
                [1,0],
                [1,-1],
                [0,1],
                [0,0],
                [0,-1],
                [-1,1],
                [-1,0],
                [-1,-1]
                ]).int().to(device)
    
    # weights
    t = tensor([ 
                1/36,
                1/9,
                1/36,
                1/9,
                4/9,
                1/9,
                1/36,
                1/9,
                1/36
                ]).float().to(device)
    
    #------------------------------------------------------------------------------
    # initial velocity profile
    vel = inivel(uLB, ly, 2, nx, ny)
    #--------------------------------------------------------------
    # initialize fin to equilibirum (rho = 1)
    fin = equilibrium(1,vel,v,t,nx,ny).to(device)
    #--------------------------------------------------------------
    # Run main loop
    Cl,Cd = lbm_loop(maxIter,bdry,obstacle,fin,vel,uLB,omega,C,nx,ny,v,t,img,t_img,map_color,
                     coefs,tF_start,tF_out,F_num)
    return Cl, Cd

#---------------------------------------------------------------------------------------------------------------------------
# Velocity inside obstacle is non 0
def lbm_loop(maxIter, bdry, obstacle, fin, vel, uLB, omega, C, nx, ny, v, t, img, t_img, map_color,
             coefs, tF_start, tF_out, F_num):
    #==============================================================================
    #   Time-Stepping
    #==============================================================================
    t0 = ti()
    
    # these variables used to group indices
    col_0 = tensor([0,1,2]).long().to(device)
    col_1 = tensor([3,4,5]).long().to(device)
    col_2 = tensor([6,7,8]).long().to(device)

    # Force and lift coefficients
    Cl = zeros(F_num).to(device)
    Cd = zeros(F_num).to(device)
    # Factor that divides hydrodynamic forces
    Cdiv = (C*uLB**2) / 2 # ρuC / 2 with C the chord length of the wing
    Cind = 0

    for time in tqdm(range(maxIter)):
        # outflow boundary condition (right side) NEUMANN BC, no gradient
        fin[col_2,-1,:] = fin[col_2,-2,:]
    
        # compute macroscopic variables
        rho,u = macroscopic(fin,nx,ny,v)
    
        # inlet boundary condition (left wall)
        u[:,0,:] = vel[:,0,:]
        rho[0,:] = 1/(1-u[0,0,:])*( tsum(fin[col_1,0,:], axis = 0)+
                                    2*tsum(fin[col_2,0,:], axis = 0))
    
        # Equilibrium
        feq = equilibrium(rho,u,v,t,nx,ny)
    
        fin[col_0,0,:] = feq[col_0,0,:] + fin[col_2,0,:]-feq[col_2,0,:]
    
        # Collide
        fout = fin - omega*(fin-feq)

        # Calculate force
        # Compute lift and drag coefficients
        if coefs & (time>=tF_start) & (time%tF_out == 0):
            F = force(fout,bdry,v)
            Cd[Cind] = F[0].item() / Cdiv
            Cl[Cind] = F[1].item() / Cdiv
            Cind += 1
    
        # stream
        for i in range(9):
            # This automatically implements a periodic
            # boundary unless overwritten
            fin[i,:,:] = roll(  
                              roll(
                                    fout[i,:,:], v[i,0].item(), dims = 0
                                   ),
                              v[i,1].item(), dims = 1 
                              )
            
        # Halfway bounceback rule at boundary nodes
        for n in bdry:
            # n[0] is x position, n[1] is y position, 
            # n[2] are links that cut into obstacle
            for i in n[2]:
                fin[8-i,n[0],n[1]] = fout[i,n[0],n[1]]

        # Make distribution equal 0 at solid nodes
        fin[:,obstacle] = 0
    
        # Output an image every t_img iterations
        if img & (time%t_img == 0):
            plt.clf()
            u_cpu = u.cpu()
            p_cpu = rho.cpu() / 3
            dfydx = u[0,1:-1,2:] - u[0,1:-1,0:-2] 
            dfxdy = u[1,2:,1:-1] - u[1,0:-2,1:-1]
            vort_cpu = (dfydx - dfxdy).cpu()
            # speed field
            plt.imshow(sqrt(u_cpu[0]**2+u_cpu[1]**2).T, cmap = map_color)
            plt.clim(0,0.16)
            plt.colorbar()
            plt.savefig("vel{0:03d}.png".format(time//t_img))
            # pressure field
            plt.clf()
            plt.imshow(p_cpu.T, cmap = map_color)
            plt.clim(0.30,0.34)
            plt.colorbar()
            plt.savefig("pr{0:03d}.png".format(time//t_img))
            # Vorticity field
            plt.clf()
            plt.imshow(vort_cpu.T, cmap = 'bwr')
            plt.clim(-0.16,0.16)
            plt.colorbar()
            plt.savefig("vort{0:03d}.png".format(time//t_img))
    
    tf = ti() - t0
    print("time to execute = ",tf)
    return Cl, Cd

#---------------------------------------------------------------------------------------------------------------------------
def obstacle_fun(cx, cy, r):
    def ret_fun(x, y):
        return (x-cx)**2+(y-cy)**2<r**2
    return ret_fun

def inivel( uLB, ly, d, nx, ny):
  vel = zeros((d,nx,ny)).to(device)
  for dir in range(d):
    vel[dir,:,:] = (1-dir) * uLB
  return vel

def macroscopic(fin, nx, ny, v):
    rho = tsum(fin,axis=0).to(device)
    u = zeros((2,nx,ny)).to(device)
    for i in range(9):
        u[0,:,:] += v[i,0]*fin[i,:,:]
        u[1,:,:] += v[i,1]*fin[i,:,:]
    u /= rho
    return rho, u

def force(fout, bdry, v):
    F = zeros(2).to(device)
    for n in bdry:
        # n[0] is x position, n[1] is y position, 
        # n[2] are links that cut into obstacle
        for i in n[2]:
            F[0] += v[i,0]*fout[i,n[0],n[1]]
            F[1] += -v[i,1]*fout[i,n[0],n[1]]
    return 2*F

def equilibrium(rho, u, v, t, nx, ny):
    usqr = (3/2)*(u[0]**2+u[1]**2)
    feq = zeros((9,nx,ny))
    for i in range(9):
        cu = 3*(v[i,0]*u[0,:,:] + v[i,1]*u[1,:,:])
        feq[i,:,:] = rho*t[i]*(1+cu+0.5*cu**2-usqr)
    return feq.to(device)