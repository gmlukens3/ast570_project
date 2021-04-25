from pygadgetreader import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import matplotlib.colors as mcolors
import csv
from statistics import median
from scipy import integrate 
from scipy import special as sci
from scipy import misc
import scipy.stats as st
#from galpy import potential
#from sympy import symbols, diff

G=4.30091*10**-6 #G in terms of km/s ^2 M_sun ^-1 kpc

Z_d = 0.15 #scale height in kpc
r_d = 2 #scale length in kpc
a_h = 5
a_b = 0.25
Sigma_0 = 1
m_star = 1 # one solar mass to get it into SI units

N_disk = 100000#use 10**8 when actually running
N_halo = 10000
N_bulge = 18790 #use 10**5 when actually running

#have to be careful with velocities, since positions are in kpc, so velocities should be kpc/s
m_to_kpc = 3.08567758128e19 #meters per kpc

M_d = m_star * N_disk 
M_b = m_star * N_bulge 
M_h = m_star * N_halo 

##index 7920 was the "header" of the lists read in so has velocity 0,1,2 and position 0,1,2 at snapshot 000. must be removed
halo_xyz_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'pos', 'dm')
disk_xyz_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'pos', 'disk')
bulge_xyz_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'pos', 'bulge')

halo_vel_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'vel', 'dm')
disk_vel_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'vel', 'disk')
bulge_vel_final = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_240', 'vel', 'bulge')

halo_x_f = []
halo_y_f = []
halo_z_f = []
halo_vx_f = []
halo_vy_f = []
halo_vz_f = []

disk_x_f = []
disk_y_f = []
disk_z_f = []
disk_vx_f = []
disk_vy_f = []
disk_vz_f = []

bulge_x_f = []
bulge_y_f = []
bulge_z_f = []
bulge_vx_f = []
bulge_vy_f = []
bulge_vz_f = []

for i in range(0,len(halo_xyz_final)):
    halo_x_f.append(halo_xyz_final[i][0])
    halo_y_f.append(halo_xyz_final[i][1])
    halo_z_f.append(halo_xyz_final[i][2])
    halo_vx_f.append(halo_vel_final[i][0])
    halo_vy_f.append(halo_vel_final[i][1])
    halo_vz_f.append(halo_vel_final[i][2])
        
for i in range(0,len(disk_xyz_final)):
    disk_x_f.append(disk_xyz_final[i][0])
    disk_y_f.append(disk_xyz_final[i][1])
    disk_z_f.append(disk_xyz_final[i][2])
    disk_vx_f.append(disk_vel_final[i][0])
    disk_vy_f.append(disk_vel_final[i][1])
    disk_vz_f.append(disk_vel_final[i][2])

for i in range(0,len(bulge_xyz_final)):
    bulge_x_f.append(bulge_xyz_final[i][0])
    bulge_y_f.append(bulge_xyz_final[i][1])
    bulge_z_f.append(bulge_xyz_final[i][2])
    bulge_vx_f.append(bulge_vel_final[i][0])
    bulge_vy_f.append(bulge_vel_final[i][1])
    bulge_vz_f.append(bulge_vel_final[i][2])


def modified_disk_mass(r):
    M = M_d *(1 - (1+r/r_d)*np.exp(-r/r_d)) #ignoring the tanh(z/z_d) term, which will be included later
    return M

def disk_mass(r,z):
    M = M_d *(1 - (1+r/r_d)*np.exp(-r/r_d)) * np.tanh(z/Z_d)
    return M

#took M_b equal to (0.5* 4pi*rho_0*a^3) since M_b is when r-> infinity for Hernquist
def bulge_mass(r):
    M = M_b * (r/a_b)**2 * (1/(1+r/a_b)**2)
    return M

#took M_total out to about 5*a_h where M(5a_h) is approximately equal to 4pi*rho_0*a^3  for the halo, so 4pi*rho_0*a^3 = M_b
def halo_mass(r):
    M = M_h * (np.log(1+r/a_h) - ((r/a_h)/(1+r/a_h)))
    return M

def hernquist_dispersion_equation_halo(r):
    h = halo_density(r) * G* (modified_disk_mass(r) + bulge_mass(r) + halo_mass(r)) / r**2
    return h

def hernquist_dispersion_equation_bulge(r):
    b = bulge_density(r) * G* (modified_disk_mass(r) + bulge_mass(r) + halo_mass(r)) / r**2
    return b

def disk_potential_approx(r):
    potential = -G*M_d / (r) * (1-np.exp(-r/r_d))
    return potential

#def disk_potential(R,z):
   # A = M_d  / (4*np.pi*r_d**2 * Z_d)
    #diskpotential = potential.MN3ExponentialDiskPotential(amp = A, hr = r_d, hz = Z_d, sech=True)
    #return potential.evaluate(R,z,diskpotential)

def bulge_potential(r): 
    potential = -G * M_b * (1/(r+a_b))
    return potential 

def halo_potential(r):
    potential = -G * M_h * np.log(1+r/a_h)/(r)
    return potential

def modbulge_potential(r): 
    a_bm = 0.25
    M_bm = 50000
    potential = -G * M_bm * (a_bm/(r+a_bm))
    return potential 
def modhalo_potential(r):
    a_hm = 5
    M_hm = 50000
    potential = -G * M_hm * np.log(1+r/a_hm)/(r/a_hm)
    return potential

def total_potential_approx(r):
    return disk_potential_approx(r) + bulge_potential(r) + halo_potential(r)

#def modtotal_pot(r):
    #return disk_potential_approx(r) + modhalo_potential(r) + modbulge_potential(r)
#def total_potential(R):
    #return disk_potential(R) + bulge_potential(R) + halo_potential(R)


#############################
#halo density plots

def halo_density(r):
    density = M_h  / (4*np.pi*a_h**3) / ((r/a_h) * (1+r/a_h)**2)
    return density


#def veldisp_radial_halo_model(r):
    #integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    #return integral / halo_density(r)
#rhalo = np.linspace(min(r_h), max(r_h), 25000).tolist()
#veldisp_r_h_model=[]
#for r in rhalo:
    #veldisp_r_h_model.append(veldisp_radial_halo_model(r))
#plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
#plt.legend(fontsize=20)


#density plots
r_h_f = []
density_halo = []
density_halo_model = []
radius_median_halo = []
vel_r_h_f = []
veldisp_r_h_f = []
vel_circ_h_f = []
R_h_f = []
index = 0
for i in range(0,len(halo_x_f)):
    r_h_f.append(np.sqrt(halo_x_f[i]**2 + halo_y_f[i]**2 + halo_z_f[i]**2))
    R_h_f.append(np.sqrt(halo_x_f[i]**2 + halo_y_f[i]**2))
    vel_r_h_f.append(((halo_vx_f[i] * halo_x_f[i]) + (halo_vy_f[i] * halo_y_f[i]) + (halo_vz_f[i] * halo_z_f[i]))/(r_h_f[i]))
    vel_circ_h_f.append(((halo_x_f[i]*halo_vy_f[i]) - (halo_y_f[i]*halo_vx_f[i]))/r_h_f[i])
    
r_h_f,vel_r_h_f,vel_circ_h_f,R_h_f = zip(*sorted(zip(r_h_f,vel_r_h_f,vel_circ_h_f,R_h_f)))
r_h_f = list(r_h_f)
vel_r_h_f = list(vel_r_h_f)
vel_circ_h_f = list(vel_circ_h_f)
R_h_f = list(R_h_f)

for r in r_h_f:
    r_1 = r
    r_2 = r_h_f[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h_f[index:index+1000])
    veldisp_r_h_f.append((np.std(vel_r_h_f[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_halo.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h_f)-1000:
        break


##############################
# bulge density plots
def bulge_density(r):
    density = M_b  / (4*np.pi*a_b**3) / ((r/a_b)**1 * (1+r/a_b)**3)
    return density


density_bulge = []
#density_bulge_model = []
radius_median_bulge = []

vel_r_b_f = []
veldisp_r_b_f = []
vel_circ_b_f = []
r_b_f = []
R_b_f = []
for i in range(0,len(bulge_x_f)):
    r_b_f.append(np.sqrt(bulge_x_f[i]**2 + bulge_y_f[i]**2 + bulge_z_f[i]**2))
    R_b_f.append(np.sqrt(bulge_x_f[i]**2 + bulge_y_f[i]**2))
    vel_r_b_f.append(((bulge_vx_f[i] * bulge_x_f[i]) + (bulge_vy_f[i] * bulge_y_f[i])+(bulge_vz_f[i] * bulge_z_f[i]))/(r_b_f[i]))
    vel_circ_b_f.append(((bulge_x_f[i]*bulge_vy_f[i]) - (bulge_y_f[i]*bulge_vx_f[i]))/r_b_f[i])
r_b_f,vel_r_b_f,vel_circ_b_f,R_b_f = zip(*sorted(zip(r_b_f,vel_r_b_f,vel_circ_b_f,R_b_f)))
r_b_f = list(r_b_f)
vel_r_b_f = list(vel_r_b_f)
vel_circ_b_f = list(vel_circ_b_f)
R_b_f = list(R_b_f)


index = 0
for r in r_b_f:
    r_1 = r
    r_2 = r_b_f[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b_f[index:index+1000])
    veldisp_r_b_f.append((np.std(vel_r_b_f[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_bulge.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_bulge_model.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b_f)-1000:
        break



############## disk density plots
def disk_density(r,z):
    #r2 = r**2 + z**2
    density = M_d  / (4*np.pi*r_d**2 * Z_d) * np.exp(-r/r_d) / (np.cosh(z/Z_d))**2
    return density

def disk_density_approx(r,z):
    #r2 = r**2 + z**2
    density = M_d  / (4*np.pi*r_d**2 * Z_d * r) * np.exp(-r/r_d) / (np.cosh(z/Z_d))**2
    return density

vel_r_d_f = []
vel_circ_d_f = []
vel_z_d_f = disk_vz_f
R_f = []
for i in range(0,len(disk_x_f)):
    R_f.append(np.sqrt(disk_x_f[i]**2 + disk_y_f[i]**2))
    vel_r_d_f.append(((disk_vx_f[i] * disk_x_f[i]) + (disk_vy_f[i] * disk_y_f[i]))/(R_f[i]))
    vel_circ_d_f.append(((disk_x_f[i]*disk_vy_f[i]) - (disk_y_f[i]*disk_vx_f[i]))/R_f[i])

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_f = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list_f = []
density_list_f = []
#density_model_list_f = []
zdisk_list_f = []
Rdisk_list_f = []
indices_list_f = []
#density_model_approx_list_f = []
veldisp_r_d_f = []
veldisp_azi_d_f = []
veldisp_z_d_f = []
#R_f_nobulge = []
# = []

#for i in range(0,len(R_f)):
    #if disk_z_f[i] >= -1.5 and disk_z_f[i]<= 1.5:
        #R_f_nobulge.append(R_f[i])
        #vel_r_f_nobulge.append(vel_r_f[i])
 

R_f_sorted,vel_r_d_f,vel_circ_d_f, vel_z_d_f = zip(*sorted(zip(R_f,vel_r_d_f,vel_circ_d_f,vel_z_d_f)))
R_f_sorted = list(R_f_sorted)
vel_r_d_f = list(vel_r_d_f)
vel_circ_d_f = list(vel_circ_d_f)
vel_z_d_f = list(vel_z_d_f)
#R_f_sorted.sort()
index = 0
rad_median_f = []

for r in vel_r_d_f:
    rad_median_f.append(median(R_f_sorted[index:index+100]))
    veldisp_r_d_f.append((np.std(vel_r_d_f[index:index+100]))**2)
    veldisp_azi_d_f.append((np.std(vel_circ_d_f[index:index+100]))**2)
    veldisp_z_d_f.append((np.std(vel_z_d_f[index:index+100]))**2)

    index += 1
    
    if index == len(vel_r_d_f)-100:
        break
        

z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]
for i in range(0,len(z_cuts)-1):
    zdisk_list_f.append([])
    indices_list_f.append([])
    Rdisk_list_f.append([])
    density_list_f.append([])
   # density_model_list_f.append([])
    #density_model_approx_list_f.append([])
    medianradius_list_f.append([])
    

for j in range(0,len(disk_z_f)):
    for i in range(0,len(z_cuts)-1):
        if disk_z_f[j] >= z_cuts[i] and disk_z_f[j] <= z_cuts[i+1]:
            indices_list_f[i].append(j)
            zdisk_list_f[i].append(disk_z_f[j])
            Rdisk_list_f[i].append(R_f[j])

            
for i in range(0,3):
    Rdisk_list_f[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist() 
    #for r in rmod:
       # density_model_approx_list_f[i].append(disk_density_approx(r,z_m))
        #density_model_list_f[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_f[i]:
        
        r_1 = Rdisk_list_f[i][index]
        r_2 = Rdisk_list_f[i][index + 2]
        r_m = median(Rdisk_list_f[i][index:index+2])
        
        
    
        density = 2 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1f0
        density_list_f[i].append( density )
        medianradius_list_f[i].append(r_m)
    
        index+=1
    
        if index == len(Rdisk_list_f[i])-2:
            index = 0
            break
        
for i in range(3,15):
    Rdisk_list_f[i].sort()
    index = 0  
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_f[i].append(disk_density_approx(r,z_m))
        #density_model_list_f[i].append(disk_density(r, z_m))
    
    for r in Rdisk_list_f[i]:
        
        r_1 = Rdisk_list_f[i][index]
        r_2 = Rdisk_list_f[i][index + 100]
        r_m = median(Rdisk_list_f[i][index:index+100])
        
        density = 100 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1f0:
        density_list_f[i].append( density )
        medianradius_list_f[i].append(r_m)
        
        
        index+=1
    
        if index == len(Rdisk_list_f[i])-100:
            index = 0
            break

for i in range(15,18):
    Rdisk_list_f[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_f[i].append(disk_density_approx(r,z_m))
       # density_model_list_f[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_f[i]:
        
        r_1 = Rdisk_list_f[i][index]
        r_2 = Rdisk_list_f[i][index + 2]
        r_m = median(Rdisk_list_f[i][index:index+2])
        
        density = 2/ (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_f[i].append( density )
        medianradius_list_f[i].append(r_m)
        
               
        index+=1

        if index == len(Rdisk_list_f[i])-2:
            index = 0
            break

z_density_disk_f = []
z_median_f = []
zdisk_f = disk_z_f
Rsort_f = R_f
zdisk_f,Rsort_f = zip(*sorted(zip(zdisk_f,Rsort_f)))

zdisk_f = list(zdisk_f)
Rsort_f = list(Rsort_f)

r_mean_d_f = np.mean(Rsort_f)
index = 0
for z in zdisk_f:
    z_1 = zdisk_f[index]
    z_2 = zdisk_f[index+100]
    z_m = median(zdisk_f[index:index+100])
    z_median_f.append(z_m)
    r_m = median(Rsort_f[index:index+100])
    density = 100 / (np.pi*r_mean_d_f**2)/(z_2 - z_1)
    
    z_density_disk_f.append(density)
    index += 1
    if index == (len(zdisk_f)-100):
        break

    
R_density_disk_f = []
R_density_disk_model_f = []
R_density_disk_model_approx_f = []
R_median_f = []
Rsort_f = R_f
z_f = disk_z_f
Rsort_f,z_f= zip(*sorted(zip(Rsort_f, z_f)))
z_f = list(z_f)
Rsort_f = list(Rsort_f)

index = 0
for r in Rsort_f:
    r_1 = Rsort_f[index]
    r_2 = Rsort_f[index+1000]
    r_m = np.median(Rsort_f[index:index+1000])
    R_median_f.append(r_m)
    density = 1000 / (np.pi*(r_2**2 - r_1**2))
    R_density_disk_f.append(density)
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    R_density_disk_model_f.append(c*np.exp(-r_m/r_d))
    R_density_disk_model_approx_f.append(c*np.exp(-r_m/r_d)/r)
    index += 1
    if index == (len(Rsort_f)-1000):
        break

anisotropy_d_f = []
for i in range(0,len(veldisp_r_d_f)):
    anisotropy_d_f.append(1 - (veldisp_azi_d_f[i] + veldisp_z_d_f[i])/ veldisp_r_d_f[i])
    
####################################
#initial conditions#

halo_xyz_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'pos', 'dm')
disk_xyz_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'pos', 'disk')
bulge_xyz_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'pos', 'bulge')

halo_vel_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'vel', 'dm')
disk_vel_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'vel', 'disk')
bulge_vel_00 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_000', 'vel', 'bulge')

halo_x_00 = []
halo_y_00 = []
halo_z_00 = []
halo_vx_00 = []
halo_vy_00 = []
halo_vz_00 = []

disk_x_00 = []
disk_y_00 = []
disk_z_00 = []
disk_vx_00 = []
disk_vy_00 = []
disk_vz_00 = []

bulge_x_00 = []
bulge_y_00 = []
bulge_z_00 = []
bulge_vx_00 = []
bulge_vy_00 = []
bulge_vz_00 = []

for i in range(0,len(halo_xyz_00)):
    halo_x_00.append(halo_xyz_00[i][0])
    halo_y_00.append(halo_xyz_00[i][1])
    halo_z_00.append(halo_xyz_00[i][2])
    halo_vx_00.append(halo_vel_00[i][0])
    halo_vy_00.append(halo_vel_00[i][1])
    halo_vz_00.append(halo_vel_00[i][2])
        
for i in range(0,len(disk_xyz_00)):
    disk_x_00.append(disk_xyz_00[i][0])
    disk_y_00.append(disk_xyz_00[i][1])
    disk_z_00.append(disk_xyz_00[i][2])
    disk_vx_00.append(disk_vel_00[i][0])
    disk_vy_00.append(disk_vel_00[i][1])
    disk_vz_00.append(disk_vel_00[i][2])

for i in range(0,len(bulge_xyz_00)):
    bulge_x_00.append(bulge_xyz_00[i][0])
    bulge_y_00.append(bulge_xyz_00[i][1])
    bulge_z_00.append(bulge_xyz_00[i][2])
    bulge_vx_00.append(bulge_vel_00[i][0])
    bulge_vy_00.append(bulge_vel_00[i][1])
    bulge_vz_00.append(bulge_vel_00[i][2])


#############################
#halo density plots

#def veldisp_radial_halo_model(r):
    #integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    #return integral / halo_density(r)
#rhalo = np.linspace(min(r_h), max(r_h), 25000).tolist()
#veldisp_r_h_model=[]
#for r in rhalo:
    #veldisp_r_h_model.append(veldisp_radial_halo_model(r))
#plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
#plt.legend(fontsize=20)


#density plots
r_h_00 = []
density_halo_00 = []
density_halo_model_00 = []
radius_median_halo_00 = []
vel_r_h_00 = []
veldisp_r_h_00 = []
vel_circ_h_00 = []
R_h_00 = []

index = 0
for i in range(0,len(halo_x_00)):
    r_h_00.append(np.sqrt(halo_x_00[i]**2 + halo_y_00[i]**2 + halo_z_00[i]**2))
    R_h_00.append(np.sqrt(halo_x_00[i]**2 + halo_y_00[i]**2))
    vel_r_h_00.append(((halo_vx_00[i] * halo_x_00[i]) + (halo_vy_00[i] * halo_y_00[i]) + (halo_vz_00[i] * halo_z_00[i]))/(r_h_00[i]))
    vel_circ_h_00.append(((halo_x_00[i]*halo_vy_00[i]) - (halo_y_00[i]*halo_vx_00[i]))/r_h_00[i])
    
r_h_00,vel_r_h_00,vel_circ_h_00,R_h_00 = zip(*sorted(zip(r_h_00,vel_r_h_00,vel_circ_h_00,R_h_00)))
r_h_00 = list(r_h_00)
vel_r_h_00 = list(vel_r_h_00)
vel_circ_h_00 = list(vel_circ_h_00)
R_h_00 = list(R_h_00)

for r in r_h_00:
    r_1 = r
    r_2 = r_h_00[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h_00[index:index+1000])
    veldisp_r_h_00.append((np.std(vel_r_h_00[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_halo_00.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo_00.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h_00)-1000:
        break


##############################
# bulge density plots


density_bulge_00 = []
density_bulge_model_00 = []
radius_median_bulge_00 = []

vel_r_b_00 = []
veldisp_r_b_00 = []
vel_circ_b_00 = []
r_b_00 = []
R_b_00 = []
for i in range(0,len(bulge_x_00)):
    r_b_00.append(np.sqrt(bulge_x_00[i]**2 + bulge_y_00[i]**2 + bulge_z_00[i]**2))
    R_b_00.append(np.sqrt(bulge_x_00[i]**2 + bulge_y_00[i]**2))
    vel_r_b_00.append(((bulge_vx_00[i] * bulge_x_00[i]) + (bulge_vy_00[i] * bulge_y_00[i])+(bulge_vz_00[i] * bulge_z_00[i]))/(r_b_00[i]))
    vel_circ_b_00.append(((bulge_x_00[i]*bulge_vy_00[i]) - (bulge_y_00[i]*bulge_vx_00[i]))/r_b_00[i])

r_b_00,vel_r_b_00,vel_circ_b_00,R_b_00 = zip(*sorted(zip(r_b_00,vel_r_b_00,vel_circ_b_00,R_b_00)))
r_b_00 = list(r_b_00)
vel_r_b_00 = list(vel_r_b_00)
vel_circ_b_00 = list(vel_circ_b_00)
R_b_00 = list(R_b_00)

index = 0
for r in r_b_00:
    r_1 = r
    r_2 = r_b_00[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b_00[index:index+1000])
    veldisp_r_b_00.append((np.std(vel_r_b_00[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_bulge_00.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge_00.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_bulge_model_00.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b_00)-1000:
        break



############## disk density plots

vel_r_d_00 = []
vel_circ_d_00 = []
vel_z_d_00 = disk_vz_00
R_00 = []
for i in range(0,len(disk_x_00)):
    R_00.append(np.sqrt(disk_x_00[i]**2 + disk_y_00[i]**2))
    vel_r_d_00.append(((disk_vx_00[i] * disk_x_00[i]) + (disk_vy_00[i] * disk_y_00[i]))/(R_00[i]))
    vel_circ_d_00.append(((disk_x_00[i]*disk_vy_00[i]) - (disk_y_00[i]*disk_vx_00[i]))/R_00[i])

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_00 = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list_00 = []
density_list_00 = []
density_model_list_00 = []
zdisk_list_00 = []
Rdisk_list_00 = []
indices_list_00 = []
density_model_approx_list_00 = []
veldisp_r_d_00 = []
veldisp_azi_d_00 = []
veldisp_z_d_00 = []
rmod = []
#R_00_nobulge = []
# = []

#for i in range(0,len(R_00)):
    #if disk_z_00[i] >= -1.5 and disk_z_00[i]<= 1.5:
        #R_00_nobulge.append(R_00[i])
        #vel_r_00_nobulge.append(vel_r_00[i])
 

R_f_sorted_00,vel_r_d_00,vel_circ_d_00, vel_z_d_00 = zip(*sorted(zip(R_00,vel_r_d_00,vel_circ_d_00,vel_z_d_00)))
R_f_sorted_00 = list(R_f_sorted_00)
vel_r_d_00 = list(vel_r_d_00)
vel_circ_d_00 = list(vel_circ_d_00)
vel_z_d_00 = list(vel_z_d_00)
rad_median_00 = []
#R_00_sorted.sort()
index = 0

for r in vel_r_d_00:
    rad_median_00.append(median(R_f_sorted_00[index:index+100]))
    veldisp_r_d_00.append((np.std(vel_r_d_00[index:index+100]))**2)
    veldisp_azi_d_00.append((np.std(vel_circ_d_00[index:index+100]))**2)
    veldisp_z_d_00.append((np.std(vel_z_d_00[index:index+100]))**2)
    index += 1
    
    if index == len(vel_r_d_00)-100:
        break
        

z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]

for i in range(0,len(z_cuts)-1):
    rmod.append([])
    zdisk_list_00.append([])
    indices_list_00.append([])
    Rdisk_list_00.append([])
    density_list_00.append([])
    density_model_list_00.append([])
    density_model_approx_list_00.append([])
    medianradius_list_00.append([])
    

for j in range(0,len(disk_z_00)):
    for i in range(0,len(z_cuts)-1):
        if disk_z_00[j] >= z_cuts[i] and disk_z_00[j] <= z_cuts[i+1]:
            indices_list_00[i].append(j)
            zdisk_list_00[i].append(disk_z_00[j])
            Rdisk_list_00[i].append(R_00[j])

            
for i in range(0,3):
    Rdisk_list_00[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    
    if i == 0:
        rmod[i] = np.linspace(0.1,13,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 1:
        rmod[i] = np.linspace(0.1,18,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 2:
        rmod[i] = np.linspace(0.1,20,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))

    for r in Rdisk_list_00[i]:
        
        r_1 = Rdisk_list_00[i][index]
        r_2 = Rdisk_list_00[i][index + 10]
        r_m = median(Rdisk_list_00[i][index:index+10])
        
        
    
        density = 10 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000
        density_list_00[i].append( density )
        medianradius_list_00[i].append(r_m)
    
        index+=1
    
        if index == len(Rdisk_list_00[i])-10:
            index = 0
            break
        
for i in range(3,15):
    Rdisk_list_00[i].sort()
    index = 0  
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
   
    if i == 3 or i==14:
        rmod[i] = np.linspace(2.5,12,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i ==4 or i== 5 or i==12 or i ==13:
        rmod[i] = np.linspace(0.1,16.5,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 6 or i==11:
        rmod[i] = np.linspace(0.1,17.5,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 7 or i==8 or i==9 or i==10:
        rmod[i] = np.linspace(0.1,19,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))        
         
    for r in Rdisk_list_00[i]:
        
        r_1 = Rdisk_list_00[i][index]
        r_2 = Rdisk_list_00[i][index + 1000]
        r_m = median(Rdisk_list_00[i][index:index+1000])
        
        density = 1000 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_00[i].append( density )
        medianradius_list_00[i].append(r_m)
        
        
        index+=1
    
        if index == len(Rdisk_list_00[i])-1000:
            index = 0
            break

for i in range(15,18):
    Rdisk_list_00[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    if i == 15:
        rmod[i] = np.linspace(0.1,20,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 16:
        rmod[i] = np.linspace(0.1,18,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))
    if i == 17:
        rmod[i] = np.linspace(0.1,13,100).tolist()
        for r in rmod[i]:
            density_model_approx_list_00[i].append(disk_density_approx(r,z_m))
            density_model_list_00[i].append(disk_density(r, z_m))

    for r in Rdisk_list_00[i]:
        
        r_1 = Rdisk_list_00[i][index]
        r_2 = Rdisk_list_00[i][index + 10]
        r_m = median(Rdisk_list_00[i][index:index+10])
        
        density = 10 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_00[i].append( density )
        medianradius_list_00[i].append(r_m)
        
               
        index+=1

        if index == len(Rdisk_list_00[i])-10:
            index = 0
            break

z_density_disk_00 = []
z_density_disk_model_00 = []
z_median_00 = []
zdisk_00 = disk_z_00
Rsort_00 = R_00
zdisk_00,Rsort_00 = zip(*sorted(zip(zdisk_00,Rsort_00)))

zdisk_00 = list(zdisk_00)
Rsort_00 = list(Rsort_00)

r_mean_d_00 = np.mean(Rsort_00)
index = 0
for z in zdisk_00:
    z_1 = zdisk_00[index]
    z_2 = zdisk_00[index+100]
    z_m = median(zdisk_00[index:index+100])
    z_median_00.append(z_m)
    r_m = median(Rsort_00[index:index+100])
    density = 100 / (np.pi*r_mean_d_00**2)/(z_2 - z_1)
    
    z_density_disk_00.append(density)
    c = 3.85 ##normalization
    z_density_disk_model_00.append(c * disk_density(r_mean_d_00, z_m))
    index += 1
    if index == (len(zdisk_00)-100):
        break

R_density_disk_00 = []
R_density_disk_model_00 = []
R_density_disk_model_approx_00 = []
R_median_00 = []
Rsort_00 = R_00
z_00 = disk_z_00
Rsort_00,z_00= zip(*sorted(zip(Rsort_00, z_00)))
z_00 = list(z_00)
Rsort_00 = list(Rsort_00)

index = 0
for r in Rsort_00:
    r_1 = Rsort_00[index]
    r_2 = Rsort_00[index+1000]
    r_m = np.median(Rsort_00[index:index+1000])
    R_median_00.append(r_m)
    density = 1000 / (np.pi*(r_2**2 - r_1**2))
    R_density_disk_00.append(density)
    index += 1
    if index == (len(Rsort_00)-1000):
        break
    
Rmod = np.linspace(0.01,25,1000)
for R in Rmod:
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    R_density_disk_model_00.append(c*np.exp(-R/r_d))
    R_density_disk_model_approx_00.append(c*np.exp(-R/r_d)/R)
    
anisotropy_d_00 = []
for i in range(0,len(veldisp_r_d_00)):
    anisotropy_d_00.append(1 - (veldisp_azi_d_00[i] + veldisp_z_d_00[i])/ veldisp_r_d_00[i])
########################################1st snapshot



halo_xyz_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'pos', 'dm')
disk_xyz_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'pos', 'disk')
bulge_xyz_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'pos', 'bulge')

halo_vel_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'vel', 'dm')
disk_vel_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'vel', 'disk')
bulge_vel_01 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_060', 'vel', 'bulge')

halo_x_01 = []
halo_y_01 = []
halo_z_01 = []
halo_vx_01 = []
halo_vy_01 = []
halo_vz_01 = []

disk_x_01 = []
disk_y_01 = []
disk_z_01 = []
disk_vx_01 = []
disk_vy_01 = []
disk_vz_01 = []

bulge_x_01 = []
bulge_y_01 = []
bulge_z_01 = []
bulge_vx_01 = []
bulge_vy_01 = []
bulge_vz_01 = []

for i in range(0,len(halo_xyz_01)):
    halo_x_01.append(halo_xyz_01[i][0])
    halo_y_01.append(halo_xyz_01[i][1])
    halo_z_01.append(halo_xyz_01[i][2])
    halo_vx_01.append(halo_vel_01[i][0])
    halo_vy_01.append(halo_vel_01[i][1])
    halo_vz_01.append(halo_vel_01[i][2])
        
for i in range(0,len(disk_xyz_01)):
    disk_x_01.append(disk_xyz_01[i][0])
    disk_y_01.append(disk_xyz_01[i][1])
    disk_z_01.append(disk_xyz_01[i][2])
    disk_vx_01.append(disk_vel_01[i][0])
    disk_vy_01.append(disk_vel_01[i][1])
    disk_vz_01.append(disk_vel_01[i][2])

for i in range(0,len(bulge_xyz_01)):
    bulge_x_01.append(bulge_xyz_01[i][0])
    bulge_y_01.append(bulge_xyz_01[i][1])
    bulge_z_01.append(bulge_xyz_01[i][2])
    bulge_vx_01.append(bulge_vel_01[i][0])
    bulge_vy_01.append(bulge_vel_01[i][1])
    bulge_vz_01.append(bulge_vel_01[i][2])


#############################
#halo density plots

#def veldisp_radial_halo_model(r):
    #integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    #return integral / halo_density(r)
#rhalo = np.linspace(min(r_h), max(r_h), 25010).tolist()
#veldisp_r_h_model=[]
#for r in rhalo:
    #veldisp_r_h_model.append(veldisp_radial_halo_model(r))
#plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
#plt.legend(fontsize=20)


#density plots
r_h_01 = []
density_halo_01 = []
density_halo_model_01 = []
radius_median_halo_01 = []
vel_r_h_01 = []
veldisp_r_h_01 = []
vel_circ_h_01 = []
R_h_01 = []

index = 0
for i in range(0,len(halo_x_01)):
    r_h_01.append(np.sqrt(halo_x_01[i]**2 + halo_y_01[i]**2 + halo_z_01[i]**2))
    R_h_01.append(np.sqrt(halo_x_01[i]**2 + halo_y_01[i]**2))
    vel_r_h_01.append(((halo_vx_01[i] * halo_x_01[i]) + (halo_vy_01[i] * halo_y_01[i]) + (halo_vz_01[i] * halo_z_01[i]))/(r_h_01[i]))
    vel_circ_h_01.append(((halo_x_01[i]*halo_vy_01[i]) - (halo_y_01[i]*halo_vx_01[i]))/r_h_01[i])
    
r_h_01,vel_r_h_01,vel_circ_h_01,R_h_01 = zip(*sorted(zip(r_h_01,vel_r_h_01,vel_circ_h_01,R_h_01)))
r_h_01 = list(r_h_01)
vel_r_h_01 = list(vel_r_h_01)
vel_circ_h_01 = list(vel_circ_h_01)
R_h_01 = list(R_h_01)

for r in r_h_01:
    r_1 = r
    r_2 = r_h_01[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h_01[index:index+1000])
    veldisp_r_h_01.append((np.std(vel_r_h_01[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_halo_01.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo_01.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h_01)-1000:
        break


##############################
# bulge density plots


density_bulge_01 = []
density_bulge_model_01 = []
radius_median_bulge_01 = []

vel_r_b_01 = []
veldisp_r_b_01 = []
vel_circ_b_01 = []
r_b_01 = []
R_b_01 = []
for i in range(0,len(bulge_x_01)):
    r_b_01.append(np.sqrt(bulge_x_01[i]**2 + bulge_y_01[i]**2 + bulge_z_01[i]**2))
    R_b_01.append(np.sqrt(bulge_x_01[i]**2 + bulge_y_01[i]**2))
    vel_r_b_01.append(((bulge_vx_01[i] * bulge_x_01[i]) + (bulge_vy_01[i] * bulge_y_01[i])+(bulge_vz_01[i] * bulge_z_01[i]))/(r_b_01[i]))
    vel_circ_b_01.append(((bulge_x_01[i]*bulge_vy_01[i]) - (bulge_y_01[i]*bulge_vx_01[i]))/r_b_01[i])

r_b_01,vel_r_b_01,vel_circ_b_01,R_b_01 = zip(*sorted(zip(r_b_01,vel_r_b_01,vel_circ_b_01,R_b_01)))
r_b_01 = list(r_b_01)
vel_r_b_01 = list(vel_r_b_01)
vel_circ_b_01 = list(vel_circ_b_01)
R_b_01 = list(R_b_01)

index = 0
for r in r_b_01:
    r_1 = r
    r_2 = r_b_01[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b_01[index:index+1000])
    veldisp_r_b_01.append((np.std(vel_r_b_01[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_bulge_01.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge_01.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_bulge_model_01.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b_01)-1000:
        break




############## disk density plots

vel_r_d_01 = []
vel_circ_d_01 = []
vel_z_d_01 = disk_vz_01
R_01 = []
for i in range(0,len(disk_x_01)):
    R_01.append(np.sqrt(disk_x_01[i]**2 + disk_y_01[i]**2))
    vel_r_d_01.append(((disk_vx_01[i] * disk_x_01[i]) + (disk_vy_01[i] * disk_y_01[i]))/(R_01[i]))
    vel_circ_d_01.append(((disk_x_01[i]*disk_vy_01[i]) - (disk_y_01[i]*disk_vx_01[i]))/R_01[i])

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_01 = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list_01 = []
density_list_01 = []
density_model_list_01 = []
zdisk_list_01 = []
Rdisk_list_01 = []
indices_list_01 = []
density_model_approx_list_01 = []
veldisp_r_d_01 = []
veldisp_azi_d_01 = []
veldisp_z_d_01 = []

#R_01_nobulge = []
# = []

#for i in range(0,len(R_01)):
    #if disk_z_01[i] >= -1.5 and disk_z_01[i]<= 1.5:
        #R_01_nobulge.append(R_01[i])
        #vel_r_01_nobulge.append(vel_r_01[i])
 

R_f_sorted_01,vel_r_d_01,vel_circ_d_01, vel_z_d_01 = zip(*sorted(zip(R_01,vel_r_d_01,vel_circ_d_01,vel_z_d_01)))
R_f_sorted_01 = list(R_f_sorted_01)
vel_r_d_01 = list(vel_r_d_01)
vel_circ_d_01 = list(vel_circ_d_01)
vel_z_d_01 = list(vel_z_d_01)
#R_01_sorted.sort()
index = 0
rad_median_01 = []

for r in vel_r_d_01:
    rad_median_01.append(median(R_f_sorted_01[index:index+100]))
    veldisp_r_d_01.append((np.std(vel_r_d_01[index:index+100]))**2)
    veldisp_azi_d_01.append((np.std(vel_circ_d_01[index:index+100]))**2)
    veldisp_z_d_01.append((np.std(vel_z_d_01[index:index+100]))**2)
    index += 1
    
    if index == len(vel_r_d_01)-100:
        break
        
        

z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]
for i in range(0,len(z_cuts)-1):
    zdisk_list_01.append([])
    indices_list_01.append([])
    Rdisk_list_01.append([])
    density_list_01.append([])
    #density_model_list_01.append([])
    #density_model_approx_list_01.append([])
    medianradius_list_01.append([])
    

for j in range(0,len(disk_z_01)):
    for i in range(0,len(z_cuts)-1):
        if disk_z_01[j] >= z_cuts[i] and disk_z_01[j] <= z_cuts[i+1]:
            indices_list_01[i].append(j)
            zdisk_list_01[i].append(disk_z_01[j])
            Rdisk_list_01[i].append(R_01[j])

            
for i in range(0,3):
    Rdisk_list_01[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist() 
    #for r in rmod:
        #density_model_approx_list_01[i].append(disk_density_approx(r,z_m))
        #density_model_list_01[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_01[i]:
        
        r_1 = Rdisk_list_01[i][index]
        r_2 = Rdisk_list_01[i][index + 10]
        r_m = median(Rdisk_list_01[i][index:index+10])
        
        
    
        density = 10 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1010
        density_list_01[i].append( density )
        medianradius_list_01[i].append(r_m)
    
        index+=1
    
        if index == len(Rdisk_list_01[i])-10:
            index = 0
            break
        
for i in range(3,15):
    Rdisk_list_01[i].sort()
    index = 0  
    #z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
       # density_model_approx_list_01[i].append(disk_density_approx(r,z_m))
        #density_model_list_01[i].append(disk_density(r, z_m))
    
    for r in Rdisk_list_01[i]:
        
        r_1 = Rdisk_list_01[i][index]
        r_2 = Rdisk_list_01[i][index + 500]
        r_m = median(Rdisk_list_01[i][index:index+500])
        
        density = 500 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1010:
        density_list_01[i].append( density )
        medianradius_list_01[i].append(r_m)
        
        
        index+=1
    
        if index == len(Rdisk_list_01[i])-500:
            index = 0
            break

for i in range(15,18):
    Rdisk_list_01[i].sort()
    index = 0
    #z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
   # rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_01[i].append(disk_density_approx(r,z_m))
        #density_model_list_01[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_01[i]:
        
        r_1 = Rdisk_list_01[i][index]
        r_2 = Rdisk_list_01[i][index + 10]
        r_m = median(Rdisk_list_01[i][index:index+10])
        
        density = 10 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_01[i].append( density )
        medianradius_list_01[i].append(r_m)
        
               
        index+=1

        if index == len(Rdisk_list_01[i])-10:
            index = 0
            break
        
z_density_disk_01 = []
z_median_01 = []
zdisk_01 = disk_z_01
Rsort_01 = R_01
zdisk_01,Rsort_01 = zip(*sorted(zip(zdisk_01,Rsort_01)))

zdisk_01 = list(zdisk_01)
Rsort_01 = list(Rsort_01)

r_mean_d_01 = np.mean(Rsort_01)
index = 0
for z in zdisk_01:
    z_1 = zdisk_01[index]
    z_2 = zdisk_01[index+100]
    z_m = median(zdisk_01[index:index+100])
    z_median_01.append(z_m)
    r_m = median(Rsort_01[index:index+100])
    density = 100 / (np.pi*r_mean_d_01**2)/(z_2 - z_1)
    
    z_density_disk_01.append(density)
    index += 1
    if index == (len(zdisk_01)-100):
        break

R_density_disk_01 = []
R_density_disk_model_01 = []
R_density_disk_model_approx_01 = []
R_median_01 = []
Rsort_01 = R_01
z_01 = disk_z_01
Rsort_01,z_01= zip(*sorted(zip(Rsort_01, z_01)))
z_01 = list(z_01)
Rsort_01 = list(Rsort_01)

index = 0
for r in Rsort_01:
    r_1 = Rsort_01[index]
    r_2 = Rsort_01[index+1000]
    r_m = np.median(Rsort_01[index:index+1000])
    R_median_01.append(r_m)
    density = 1000 / (np.pi*(r_2**2 - r_1**2))
    R_density_disk_01.append(density)
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    R_density_disk_model_01.append(c*np.exp(-r_m/r_d))
    R_density_disk_model_approx_01.append(c*np.exp(-r_m/r_d)/r)
    index += 1
    if index == (len(Rsort_01)-1000):
        break
    
anisotropy_d_01 = []
for i in range(0,len(veldisp_r_d_01)):
    anisotropy_d_01.append(1 - (veldisp_azi_d_01[i] + veldisp_z_d_01[i])/ veldisp_r_d_01[i])
#################################

halo_xyz_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'pos', 'dm')
disk_xyz_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'pos', 'disk')
bulge_xyz_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'pos', 'bulge')

halo_vel_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'vel', 'dm')
disk_vel_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'vel', 'disk')
bulge_vel_02 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_120', 'vel', 'bulge')

halo_x_02 = []
halo_y_02 = []
halo_z_02 = []
halo_vx_02 = []
halo_vy_02 = []
halo_vz_02 = []

disk_x_02 = []
disk_y_02 = []
disk_z_02 = []
disk_vx_02 = []
disk_vy_02 = []
disk_vz_02 = []

bulge_x_02 = []
bulge_y_02 = []
bulge_z_02 = []
bulge_vx_02 = []
bulge_vy_02 = []
bulge_vz_02 = []

for i in range(0,len(halo_xyz_02)):
  
    halo_x_02.append(halo_xyz_02[i][0])
    halo_y_02.append(halo_xyz_02[i][1])
    halo_z_02.append(halo_xyz_02[i][2])
    halo_vx_02.append(halo_vel_02[i][0])
    halo_vy_02.append(halo_vel_02[i][1])
    halo_vz_02.append(halo_vel_02[i][2])
    
for i in range(0,len(disk_xyz_02)):
    disk_x_02.append(disk_xyz_02[i][0])
    disk_y_02.append(disk_xyz_02[i][1])
    disk_z_02.append(disk_xyz_02[i][2])
    disk_vx_02.append(disk_vel_02[i][0])
    disk_vy_02.append(disk_vel_02[i][1])
    disk_vz_02.append(disk_vel_02[i][2])

for i in range(0,len(bulge_xyz_02)):
    bulge_x_02.append(bulge_xyz_02[i][0])
    bulge_y_02.append(bulge_xyz_02[i][1])
    bulge_z_02.append(bulge_xyz_02[i][2])
    bulge_vx_02.append(bulge_vel_02[i][0])
    bulge_vy_02.append(bulge_vel_02[i][1])
    bulge_vz_02.append(bulge_vel_02[i][2])


#############################
#halo density plots
#def veldisp_radial_halo_model(r):
    #integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    #return integral / halo_density(r)
#rhalo = np.linspace(min(r_h), max(r_h), 25020).tolist()
#veldisp_r_h_model=[]
#for r in rhalo:
    #veldisp_r_h_model.append(veldisp_radial_halo_model(r))
#plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
#plt.legend(fontsize=20)


#density plots
r_h_02 = []
density_halo_02 = []
density_halo_model_02 = []
radius_median_halo_02 = []
vel_r_h_02 = []
veldisp_r_h_02 = []
vel_circ_h_02 = []
R_h_02 = []

index = 0
for i in range(0,len(halo_x_02)):
    r_h_02.append(np.sqrt(halo_x_02[i]**2 + halo_y_02[i]**2 + halo_z_02[i]**2))
    R_h_02.append(np.sqrt(halo_x_02[i]**2 + halo_y_02[i]**2))
    vel_r_h_02.append(((halo_vx_02[i] * halo_x_02[i]) + (halo_vy_02[i] * halo_y_02[i]) + (halo_vz_02[i] * halo_z_02[i]))/(r_h_02[i]))
    vel_circ_h_02.append(((halo_x_02[i]*halo_vy_02[i]) - (halo_y_02[i]*halo_vx_02[i]))/r_h_02[i])
    
r_h_02,vel_r_h_02,vel_circ_h_02,R_h_02 = zip(*sorted(zip(r_h_02,vel_r_h_02,vel_circ_h_02,R_h_02)))
r_h_02 = list(r_h_02)
vel_r_h_02 = list(vel_r_h_02)
vel_circ_h_02 = list(vel_circ_h_02)
R_h_02 = list(R_h_02)

for r in r_h_02:
    r_1 = r
    r_2 = r_h_02[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h_02[index:index+1000])
    veldisp_r_h_02.append((np.std(vel_r_h_02[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_halo_02.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo_02.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h_02)-1000:
        break


##############################
# bulge density plots


density_bulge_02 = []
density_bulge_model_02 = []
radius_median_bulge_02 = []

vel_r_b_02 = []
veldisp_r_b_02 = []
vel_circ_b_02 = []
r_b_02 = []
R_b_02 = []
for i in range(0,len(bulge_x_02)):
    r_b_02.append(np.sqrt(bulge_x_02[i]**2 + bulge_y_02[i]**2 + bulge_z_02[i]**2))
    R_b_02.append(np.sqrt(bulge_x_02[i]**2 + bulge_y_02[i]**2))
    vel_r_b_02.append(((bulge_vx_02[i] * bulge_x_02[i]) + (bulge_vy_02[i] * bulge_y_02[i])+(bulge_vz_02[i] * bulge_z_02[i]))/(r_b_02[i]))
    vel_circ_b_02.append(((bulge_x_02[i]*bulge_vy_02[i]) - (bulge_y_02[i]*bulge_vx_02[i]))/r_b_02[i])

r_b_02,vel_r_b_02,vel_circ_b_02,R_b_02 = zip(*sorted(zip(r_b_02,vel_r_b_02,vel_circ_b_02,R_b_02)))
r_b_02 = list(r_b_02)
vel_r_b_02 = list(vel_r_b_02)
vel_circ_b_02 = list(vel_circ_b_02)
R_b_02 = list(R_b_02)

index = 0
for r in r_b_02:
    r_1 = r
    r_2 = r_b_02[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b_02[index:index+1000])
    veldisp_r_b_02.append((np.std(vel_r_b_02[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_bulge_02.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge_02.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_bulge_model_02.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b_02)-1000:
        break

############## disk density plots

vel_r_d_02 = []
vel_circ_d_02 = []
vel_z_d_02 = disk_vz_02
R_02 = []
for i in range(0,len(disk_x_02)):
    R_02.append(np.sqrt(disk_x_02[i]**2 + disk_y_02[i]**2))
    vel_r_d_02.append(((disk_vx_02[i] * disk_x_02[i]) + (disk_vy_02[i] * disk_y_02[i]))/(R_02[i]))
    vel_circ_d_02.append(((disk_x_02[i]*disk_vy_02[i]) - (disk_y_02[i]*disk_vx_02[i]))/R_02[i])

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_02 = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list_02 = []
density_list_02 = []
density_model_list_02 = []
zdisk_list_02 = []
Rdisk_list_02 = []
indices_list_02 = []
density_model_approx_list_02 = []
veldisp_r_d_02 = []
veldisp_azi_d_02 = []
veldisp_z_d_02 = []

#R_02_nobulge = []
# = []

#for i in range(0,len(R_02)):
    #if disk_z_02[i] >= -1.5 and disk_z_02[i]<= 1.5:
        #R_02_nobulge.append(R_02[i])
        #vel_r_02_nobulge.append(vel_r_02[i])
 

R_f_sorted_02,vel_r_d_02,vel_circ_d_02, vel_z_d_02 = zip(*sorted(zip(R_02,vel_r_d_02,vel_circ_d_02,vel_z_d_02)))
R_f_sorted_02 = list(R_f_sorted_02)
vel_r_d_02 = list(vel_r_d_02)
vel_circ_d_02 = list(vel_circ_d_02)
vel_z_d_02 = list(vel_z_d_02)
#R_02_sorted.sort()
index = 0
rad_median_02 = []

for r in vel_r_d_02:
    rad_median_02.append(median(R_f_sorted_02[index:index+100]))
    veldisp_r_d_02.append((np.std(vel_r_d_02[index:index+100]))**2)
    veldisp_azi_d_02.append((np.std(vel_circ_d_02[index:index+100]))**2)
    veldisp_z_d_02.append((np.std(vel_z_d_02[index:index+100]))**2)
    index += 1
    
    if index == len(vel_r_d_02)-100:
        break
        
        

z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]
for i in range(0,len(z_cuts)-1):
    zdisk_list_02.append([])
    indices_list_02.append([])
    Rdisk_list_02.append([])
    density_list_02.append([])
    #density_model_list_02.append([])
    #density_model_approx_list_02.append([])
    medianradius_list_02.append([])
    

for j in range(0,len(disk_z_02)):
    for i in range(0,len(z_cuts)-1):
        if disk_z_02[j] >= z_cuts[i] and disk_z_02[j] <= z_cuts[i+1]:
            indices_list_02[i].append(j)
            zdisk_list_02[i].append(disk_z_02[j])
            Rdisk_list_02[i].append(R_02[j])

            
for i in range(0,3):
    Rdisk_list_02[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist() 
    #for r in rmod:
        #density_model_approx_list_02[i].append(disk_density_approx(r,z_m))
       # density_model_list_02[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_02[i]:
        
        r_1 = Rdisk_list_02[i][index]
        r_2 = Rdisk_list_02[i][index + 10]
        r_m = median(Rdisk_list_02[i][index:index+10])
        
        
    
        density = 10 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1020
        density_list_02[i].append( density )
        medianradius_list_02[i].append(r_m)
    
        index+=1
    
        if index == len(Rdisk_list_02[i])-10:
            index = 0
            break
        
for i in range(3,15):
    Rdisk_list_02[i].sort()
    index = 0  
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_02[i].append(disk_density_approx(r,z_m))
        #density_model_list_02[i].append(disk_density(r, z_m))
    
    for r in Rdisk_list_02[i]:
        
        r_1 = Rdisk_list_02[i][index]
        r_2 = Rdisk_list_02[i][index + 500]
        r_m = median(Rdisk_list_02[i][index:index+500])
        
        density = 500 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1020:
        density_list_02[i].append( density )
        medianradius_list_02[i].append(r_m)
       
        
        index+=1
    
        if index == len(Rdisk_list_02[i])-500:
            index = 0
            break

for i in range(15,18):
    Rdisk_list_02[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
       # density_model_approx_list_02[i].append(disk_density_approx(r,z_m))
        #density_model_list_02[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_02[i]:
        
        r_1 = Rdisk_list_02[i][index]
        r_2 = Rdisk_list_02[i][index + 10]
        r_m = median(Rdisk_list_02[i][index:index+10])
        
        density = 10 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_02[i].append( density )
        medianradius_list_02[i].append(r_m)
        
               
        index+=1

        if index == len(Rdisk_list_02[i])-10:
            index = 0
            break

z_density_disk_02 = []
z_median_02 = []
zdisk_02 = disk_z_02
Rsort_02 = R_02
zdisk_02,Rsort_02 = zip(*sorted(zip(zdisk_02,Rsort_02)))

zdisk_02 = list(zdisk_02)
Rsort_02 = list(Rsort_02)

r_mean_d_02 = np.mean(Rsort_02)
index = 0
for z in zdisk_02:
    z_1 = zdisk_02[index]
    z_2 = zdisk_02[index+100]
    z_m = median(zdisk_02[index:index+100])
    z_median_02.append(z_m)
    r_m = median(Rsort_02[index:index+100])
    density = 100 / (np.pi*r_mean_d_02**2)/(z_2 - z_1)
    
    z_density_disk_02.append(density)
    index += 1
    if index == (len(zdisk_02)-100):
        break
    
R_density_disk_02 = []
R_density_disk_model_02 = []
R_density_disk_model_approx_02 = []
R_median_02 = []
Rsort_02 = R_02
z_02 = disk_z_02
Rsort_02,z_02= zip(*sorted(zip(Rsort_02, z_02)))
z_02 = list(z_02)
Rsort_02 = list(Rsort_02)

index = 0
for r in Rsort_02:
    r_1 = Rsort_02[index]
    r_2 = Rsort_02[index+1000]
    r_m = np.median(Rsort_02[index:index+1000])
    R_median_02.append(r_m)
    density = 1000 / (np.pi*(r_2**2 - r_1**2))
    R_density_disk_02.append(density)
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    R_density_disk_model_02.append(c*np.exp(-r_m/r_d))
    R_density_disk_model_approx_02.append(c*np.exp(-r_m/r_d)/r)
    index += 1
    if index == (len(Rsort_02)-1000):
        break
    
anisotropy_d_02 = []
for i in range(0,len(veldisp_r_d_02)):
    anisotropy_d_02.append(1 - (veldisp_azi_d_02[i] + veldisp_z_d_02[i])/ veldisp_r_d_02[i])
#####################################
#3rd snapshot

halo_xyz_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'pos', 'dm')
disk_xyz_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'pos', 'disk')
bulge_xyz_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'pos', 'bulge')

halo_vel_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'vel', 'dm')
disk_vel_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'vel', 'disk')
bulge_vel_03 = readsnap('C:\\Users\\doubl\\OneDrive\\Desktop\\snapshots\\snapshot_180', 'vel', 'bulge')

halo_x_03 = []
halo_y_03 = []
halo_z_03 = []
halo_vx_03 = []
halo_vy_03 = []
halo_vz_03 = []

disk_x_03 = []
disk_y_03 = []
disk_z_03 = []
disk_vx_03 = []
disk_vy_03 = []
disk_vz_03 = []

bulge_x_03 = []
bulge_y_03 = []
bulge_z_03 = []
bulge_vx_03 = []
bulge_vy_03 = []
bulge_vz_03 = []

for i in range(0,len(halo_xyz_03)):
    halo_x_03.append(halo_xyz_03[i][0])
    halo_y_03.append(halo_xyz_03[i][1])
    halo_z_03.append(halo_xyz_03[i][2])
    halo_vx_03.append(halo_vel_03[i][0])
    halo_vy_03.append(halo_vel_03[i][1])
    halo_vz_03.append(halo_vel_03[i][2])
        
for i in range(0,len(disk_xyz_03)):

    disk_x_03.append(disk_xyz_03[i][0])
    disk_y_03.append(disk_xyz_03[i][1])
    disk_z_03.append(disk_xyz_03[i][2])
    disk_vx_03.append(disk_vel_03[i][0])
    disk_vy_03.append(disk_vel_03[i][1])
    disk_vz_03.append(disk_vel_03[i][2])

for i in range(0,len(bulge_xyz_03)):
    bulge_x_03.append(bulge_xyz_03[i][0])
    bulge_y_03.append(bulge_xyz_03[i][1])
    bulge_z_03.append(bulge_xyz_03[i][2])
    bulge_vx_03.append(bulge_vel_03[i][0])
    bulge_vy_03.append(bulge_vel_03[i][1])
    bulge_vz_03.append(bulge_vel_03[i][2])


#############################
#halo density plots

#def veldisp_radial_halo_model(r):
    #integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    #return integral / halo_density(r)
#rhalo = np.linspace(min(r_h), max(r_h), 25030).tolist()
#veldisp_r_h_model=[]
#for r in rhalo:
    #veldisp_r_h_model.append(veldisp_radial_halo_model(r))
#plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
#plt.legend(fontsize=20)


#density plots
r_h_03 = []
density_halo_03 = []
density_halo_model_03 = []
radius_median_halo_03 = []
vel_r_h_03 = []
veldisp_r_h_03 = []
vel_circ_h_03 = []
R_h_03 = []

index = 0
for i in range(0,len(halo_x_03)):
    r_h_03.append(np.sqrt(halo_x_03[i]**2 + halo_y_03[i]**2 + halo_z_03[i]**2))
    R_h_03.append(np.sqrt(halo_x_03[i]**2 + halo_y_03[i]**2))
    vel_r_h_03.append(((halo_vx_03[i] * halo_x_03[i]) + (halo_vy_03[i] * halo_y_03[i]) + (halo_vz_03[i] * halo_z_03[i]))/(r_h_03[i]))
    vel_circ_h_03.append(((halo_x_03[i]*halo_vy_03[i]) - (halo_y_03[i]*halo_vx_03[i]))/r_h_03[i])
    
r_h_03,vel_r_h_03,vel_circ_h_03,R_h_03 = zip(*sorted(zip(r_h_03,vel_r_h_03,vel_circ_h_03,R_h_03)))
r_h_03 = list(r_h_03)
vel_r_h_03 = list(vel_r_h_03)
vel_circ_h_03 = list(vel_circ_h_03)
R_h_03 = list(R_h_03)

for r in r_h_03:
    r_1 = r
    r_2 = r_h_03[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h_03[index:index+1000])
    veldisp_r_h_03.append((np.std(vel_r_h_03[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_halo_03.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo_03.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h_03)-1000:
        break


##############################
# bulge density plots


density_bulge_03 = []
density_bulge_model_03 = []
radius_median_bulge_03 = []

vel_r_b_03 = []
veldisp_r_b_03 = []
vel_circ_b_03 = []
r_b_03 = []
R_b_03 = []
for i in range(0,len(bulge_x_03)):
    r_b_03.append(np.sqrt(bulge_x_03[i]**2 + bulge_y_03[i]**2 + bulge_z_03[i]**2))
    R_b_03.append(np.sqrt(bulge_x_03[i]**2 + bulge_y_03[i]**2))
    vel_r_b_03.append(((bulge_vx_03[i] * bulge_x_03[i]) + (bulge_vy_03[i] * bulge_y_03[i])+(bulge_vz_03[i] * bulge_z_03[i]))/(r_b_03[i]))
    vel_circ_b_03.append(((bulge_x_03[i]*bulge_vy_03[i]) - (bulge_y_03[i]*bulge_vx_03[i]))/r_b_03[i])

r_b_03,vel_r_b_03,vel_circ_b_03,R_b_03 = zip(*sorted(zip(r_b_03,vel_r_b_03,vel_circ_b_03,R_b_03)))
r_b_03 = list(r_b_03)
vel_r_b_03 = list(vel_r_b_03)
vel_circ_b_03 = list(vel_circ_b_03)
R_b_03 = list(R_b_03)

index = 0
for r in r_b_03:
    r_1 = r
    r_2 = r_b_03[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b_03[index:index+1000])
    veldisp_r_b_03.append((np.std(vel_r_b_03[index:index+1000]))**2)
    #r_m = r + 0.5   
    radius_median_bulge_03.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge_03.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    #density_bulge_model_03.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b_03)-1000:
        break


############## disk density plots

vel_r_d_03 = []
vel_circ_d_03 = []
vel_z_d_03 = disk_vz_03
R_03 = []
for i in range(0,len(disk_x_03)):
    R_03.append(np.sqrt(disk_x_03[i]**2 + disk_y_03[i]**2))
    vel_r_d_03.append(((disk_vx_03[i] * disk_x_03[i]) + (disk_vy_03[i] * disk_y_03[i]))/(R_03[i]))
    vel_circ_d_03.append(((disk_x_03[i]*disk_vy_03[i]) - (disk_y_03[i]*disk_vx_03[i]))/R_03[i])

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_03 = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list_03 = []
density_list_03 = []
density_model_list_03 = []
zdisk_list_03 = []
Rdisk_list_03 = []
indices_list_03 = []
density_model_approx_list_03 = []
veldisp_r_d_03 = []
veldisp_azi_d_03 = []
veldisp_z_d_03 = []

#R_03_nobulge = []
# = []

#for i in range(0,len(R_03)):
    #if disk_z_03[i] >= -1.5 and disk_z_03[i]<= 1.5:
        #R_03_nobulge.append(R_03[i])
        #vel_r_03_nobulge.append(vel_r_03[i])
 

R_f_sorted_03,vel_r_d_03,vel_circ_d_03, vel_z_d_03 = zip(*sorted(zip(R_03,vel_r_d_03,vel_circ_d_03,vel_z_d_03)))
R_f_sorted_03 = list(R_f_sorted_03)
vel_r_d_03 = list(vel_r_d_03)
vel_circ_d_03 = list(vel_circ_d_03)
vel_z_d_03 = list(vel_z_d_03)
#R_03_sorted.sort()
index = 0
rad_median_03 = []
for r in vel_r_d_03:
    rad_median_03.append(median(R_f_sorted_03[index:index+100]))
    veldisp_r_d_03.append((np.std(vel_r_d_03[index:index+100]))**2)
    veldisp_azi_d_03.append((np.std(vel_circ_d_03[index:index+100]))**2)
    veldisp_z_d_03.append((np.std(vel_z_d_03[index:index+100]))**2)
    index += 1
    
    if index == len(vel_r_d_03)-100:
        break
        
        

z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]
for i in range(0,len(z_cuts)-1):
    zdisk_list_03.append([])
    indices_list_03.append([])
    Rdisk_list_03.append([])
    density_list_03.append([])
    #density_model_list_03.append([])
    #density_model_approx_list_03.append([])
    medianradius_list_03.append([])
    

for j in range(0,len(disk_z_03)):
    for i in range(0,len(z_cuts)-1):
        if disk_z_03[j] >= z_cuts[i] and disk_z_03[j] <= z_cuts[i+1]:
            indices_list_03[i].append(j)
            zdisk_list_03[i].append(disk_z_03[j])
            Rdisk_list_03[i].append(R_03[j])

            
for i in range(0,3):
    Rdisk_list_03[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
   # rmod = np.linspace(0,20,100)
    #rmod.tolist() 
    #for r in rmod:
       # density_model_approx_list_03[i].append(disk_density_approx(r,z_m))
       # density_model_list_03[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_03[i]:
        
        r_1 = Rdisk_list_03[i][index]
        r_2 = Rdisk_list_03[i][index + 10]
        r_m = median(Rdisk_list_03[i][index:index+10])
        
        
    
        density = 10 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1030
        density_list_03[i].append( density )
        medianradius_list_03[i].append(r_m)
    
        index+=1
    
        if index == len(Rdisk_list_03[i])-10:
            index = 0
            break
        
for i in range(3,15):
    Rdisk_list_03[i].sort()
    index = 0  
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_03[i].append(disk_density_approx(r,z_m))
        #density_model_list_03[i].append(disk_density(r, z_m))
    
    for r in Rdisk_list_03[i]:
        
        r_1 = Rdisk_list_03[i][index]
        r_2 = Rdisk_list_03[i][index + 500]
        r_m = median(Rdisk_list_03[i][index:index+500])
        
        density = 500 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1030:
        density_list_03[i].append( density )
        medianradius_list_03[i].append(r_m)
       
        
        index+=1
    
        if index == len(Rdisk_list_03[i])-500:
            index = 0
            break

for i in range(15,18):
    Rdisk_list_03[i].sort()
    index = 0
    z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
    dz = abs(z_cuts[i]-z_cuts[i+1])
    #rmod = np.linspace(0,20,100)
    #rmod.tolist()
    
    #for r in rmod:
        #density_model_approx_list_03[i].append(disk_density_approx(r,z_m))
        #density_model_list_03[i].append(disk_density(r, z_m))
        
    for r in Rdisk_list_03[i]:
        
        r_1 = Rdisk_list_03[i][index]
        r_2 = Rdisk_list_03[i][index + 10]
        r_m = median(Rdisk_list_03[i][index:index+10])
        
        density = 10 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list_03[i].append( density )
        medianradius_list_03[i].append(r_m)
        
               
        index+=1

        if index == len(Rdisk_list_03[i])-10:
            index = 0
            break

z_density_disk_03 = []
z_median_03 = []
zdisk_03 = disk_z_03
Rsort_03 = R_03
zdisk_03,Rsort_03 = zip(*sorted(zip(zdisk_03,Rsort_03)))

zdisk_03 = list(zdisk_03)
Rsort_03 = list(Rsort_03)

r_mean_d_03 = np.mean(Rsort_03)
index = 0
for z in zdisk_03:
    z_1 = zdisk_03[index]
    z_2 = zdisk_03[index+100]
    z_m = median(zdisk_03[index:index+100])
    z_median_03.append(z_m)
    r_m = median(Rsort_03[index:index+100])
    density = 100 / (np.pi*r_mean_d_03**2)/(z_2 - z_1)
    
    z_density_disk_03.append(density)
    index += 1
    if index == (len(zdisk_03)-100):
        break
    
R_density_disk_03 = []
R_density_disk_model_03 = []
R_density_disk_model_approx_03 = []
R_median_03 = []
Rsort_03 = R_03
z_03 = disk_z_03
Rsort_03,z_03= zip(*sorted(zip(Rsort_03, z_03)))
z_03 = list(z_03)
Rsort_03 = list(Rsort_03)

index = 0
for r in Rsort_03:
    r_1 = Rsort_03[index]
    r_2 = Rsort_03[index+1000]
    r_m = np.median(Rsort_03[index:index+1000])
    R_median_03.append(r_m)
    density = 1000 / (np.pi*(r_2**2 - r_1**2))
    R_density_disk_03.append(density)
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    R_density_disk_model_03.append(c*np.exp(-r_m/r_d))
    R_density_disk_model_approx_03.append(c*np.exp(-r_m/r_d)/r)
    index += 1
    if index == (len(Rsort_03)-1000):
        break
    
anisotropy_d_03 = []
for i in range(0,len(veldisp_r_d_03)):
    anisotropy_d_03.append(1 - (veldisp_azi_d_03[i] + veldisp_z_d_03[i])/ veldisp_r_d_03[i])
#####################################
#plotting positions
fig = plt.figure(1, figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(halo_x_00, halo_y_00, halo_z_00, color = 'green', s=1)
plot2 = ax.scatter(disk_x_00, disk_y_00, disk_z_00, s=1)
plot3 = ax.scatter(bulge_x_00, bulge_y_00, bulge_z_00, color = 'red', s=1)
plt.title('Initial Positions')

#plotting positions
fig = plt.figure(2, figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(halo_x_01, halo_y_01, halo_z_01, color = 'green', s=1)
plot2 = ax.scatter(disk_x_01, disk_y_01, disk_z_01, s=1)
plot3 = ax.scatter(bulge_x_01, bulge_y_01, bulge_z_01, color = 'red', s=1)
plt.title('1st Snapshot')

#plotting positions
fig = plt.figure(3, figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(halo_x_02, halo_y_02, halo_z_02, color = 'green', s=1)
plot2 = ax.scatter(disk_x_02, disk_y_02, disk_z_02, s=1)
plot3 = ax.scatter(bulge_x_02, bulge_y_02, bulge_z_02, color = 'red', s=1)
plt.title('2nd Snapshot')

#plotting positions
fig = plt.figure(4, figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(halo_x_03, halo_y_03, halo_z_03, color = 'green', s=1)
plot2 = ax.scatter(disk_x_03, disk_y_03, disk_z_03, s=1)
plot3 = ax.scatter(bulge_x_03, bulge_y_03, bulge_z_03, color = 'red', s=1)
plt.title('3rd Snapshot')

#plotting positions
fig = plt.figure(5, figsize = (20,20))

ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(halo_x_f, halo_y_f, halo_z_f, color = 'green', s=1)
plot2 = ax.scatter(disk_x_f, disk_y_f, disk_z_f, s=1)
plot3 = ax.scatter(bulge_x_f, bulge_y_f, bulge_z_f, color = 'red', s=1)
plt.title('Final Positions')


#########
fig = plt.figure(6,figsize=(20,20))
plt.title('Density of the Halo', fontsize=25)
ax = plt.gca()
ax.set_yscale('log')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel('r', fontsize=20)
plt.ylabel('$ \log \\left[ \\rho_h(r) \\right ] $', fontsize=20)

rhalo = np.linspace(min(r_h_00), max(r_h_f), 1000).tolist()
for r in rhalo:
    density_halo_model_00.append(halo_density(r))
    
ax.scatter(radius_median_halo_00, density_halo_00,s = 10, label = 'Initial Snapshot')
ax.scatter(radius_median_halo_01, density_halo_01,s = 10, color = 'orange',label = '1st Snapshot')
ax.scatter(radius_median_halo_02, density_halo_02,s = 10, color = 'magenta',label = '2nd Snapshot')
ax.scatter(radius_median_halo_03, density_halo_03,s = 10, color = 'green', label = '3rd Snapshot')
ax.scatter(radius_median_halo, density_halo, color = 'black', label = 'Final Snapshot')

plt.plot(rhalo, density_halo_model_00, color = 'red', label = 'Analytical')
plt.legend(fontsize=20)

#############
fig = plt.figure(7, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax.scatter(radius_median_halo, veldisp_r_h_00,s = 10, label = 'Initial Snapshot')
ax.scatter(radius_median_halo, veldisp_r_h_01,s = 10, c = 'orange', label = '1st Snapshot')
ax.scatter(radius_median_halo, veldisp_r_h_02,s = 10, c = 'magenta', label = '2nd Snapshot')
ax.scatter(radius_median_halo, veldisp_r_h_03,s = 10, c = 'green', label = '3rd Snapshot')
ax.scatter(radius_median_halo, veldisp_r_h_f,s = 10, c = 'black', label = 'Final Snapshot')

plt.xlabel('r', fontsize=20)
plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)
plt.title('Radial Velocity Dispersion of the Halo', fontsize=25)
def veldisp_radial_halo_model(r):
    integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    return integral / halo_density(r)
rhalo = np.linspace(4.5, max(r_h_f), 10000).tolist()
veldisp_r_h_model=[]
for r in rhalo:
    veldisp_r_h_model.append(veldisp_radial_halo_model(r))
ax.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)


######################
fig = plt.figure(8, figsize=(20,20))
plt.title('Density of the Bulge', fontsize=25)
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('r', fontsize=20)
plt.ylabel('$ \log \\left[ \\rho_b(r) \\right ] $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

rbulge = np.linspace(0.1, 7, 1000).tolist()
for r in rbulge:
    density_bulge_model_00.append(bulge_density(r))

ax.scatter(radius_median_bulge_00, density_bulge_00,s = 10, label = 'Initial Snapshot')
ax.scatter(radius_median_bulge_01, density_bulge_01,s = 10, color = 'orange',label = '1st Snapshot')
ax.scatter(radius_median_bulge_02, density_bulge_02,s = 10, color = 'magenta',label = '2nd Snapshot')
ax.scatter(radius_median_bulge_03, density_bulge_03,s = 10, color = 'green', label = '3rd Snapshot')
ax.scatter(radius_median_bulge, density_bulge, color = 'black', label = 'Final Snapshot')
ax.plot(rbulge, density_bulge_model_00, color = 'red', label = 'Analytical')
plt.legend(fontsize=20)

########################
fig = plt.figure(9, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('r', fontsize=20)
plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
ax.scatter(radius_median_bulge_00, veldisp_r_b_00,s = 10, label = 'Initial Snapshot')
ax.scatter(radius_median_bulge_01, veldisp_r_b_01,s = 10, c = 'orange', label = '1st Snapshot')
ax.scatter(radius_median_bulge_02, veldisp_r_b_02,s = 10, c = 'magenta', label = '2nd Snapshot')
ax.scatter(radius_median_bulge_03, veldisp_r_b_03,s = 10, c = 'green', label = '3rd Snapshot')
ax.scatter(radius_median_bulge, veldisp_r_b_f, c = 'black', label = 'Final Snapshot')

plt.title('Radial Velocity Dispersion of the Bulge')
def veldisp_radial_bulge_model(r):
    integral, error = integrate.quad(hernquist_dispersion_equation_bulge, r, np.infty)
    return integral / bulge_density(r)
rbulge = np.linspace(0.1, max(r_b_f), 1000).tolist()
veldisp_r_b_model=[]
for r in rbulge:
    veldisp_r_b_model.append(veldisp_radial_bulge_model(r))
ax.plot(rbulge, veldisp_r_b_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)

###########

for i in range(0,len(z_cuts)-1):
    fig = plt.figure(i+10, figsize=(20,20))
    ax = plt.gca()
    plt.xlabel('R', fontsize=20)
    plt.ylabel('$ \log \\left[ \Sigma(R) \\right ] $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_yscale('log')
    
    ax.scatter(medianradius_list_00[i], density_list_00[i],s = 10, label = 'Initial Snapshot (' + str(z_cuts[i]) + '$ \leq z \leq $' + str(z_cuts[i+1]) + ')')
    ax.scatter(medianradius_list_01[i], density_list_01[i], s = 10, color = 'orange', label = '1st Snapshot (' + str(z_cuts[i]) + '$ \leq z \leq $' + str(z_cuts[i+1]) + ')')
    ax.scatter(medianradius_list_02[i], density_list_02[i], s = 10, color = 'magenta', label = '2nd Snapshot (' + str(z_cuts[i]) + '$ \leq z \leq $' + str(z_cuts[i+1]) + ')')
    ax.scatter(medianradius_list_03[i], density_list_03[i], s = 10,color = 'green', label = '3rd Snapshot (' + str(z_cuts[i]) + '$ \leq z \leq $' + str(z_cuts[i+1]) + ')')
    ax.scatter(medianradius_list_f[i], density_list_f[i], s = 10, color = 'black', label = 'Final Snapshot (' + str(z_cuts[i]) + '$ \leq z \leq $' + str(z_cuts[i+1]) + ')')
    ax.plot(rmod[i], density_model_list_00[i], color = 'red', label = 'Analytical Disk')
    #ax.plot(rad, density_model_approx_list_00[i], color = 'blue', label = 'Spherical Approxmation')
    plt.legend(fontsize = 20)



#############
fig = plt.figure(28, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('R', fontsize=20)
plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax.scatter(rad_median_00, veldisp_r_d_00,s = 5, label = 'Initial Snapshot')
ax.scatter(rad_median_01, veldisp_r_d_01,s = 5, c = 'orange', label = '1st Snapshot')
ax.scatter(rad_median_02, veldisp_r_d_02,s = 5, c = 'magenta', label = '2nd Snapshot')
ax.scatter(rad_median_03, veldisp_r_d_03,s = 5, c = 'green', label = '3rd Snapshot')
ax.scatter(rad_median_f, veldisp_r_d_f,s = 5, c = 'black', label = 'Final Snapshot')

plt.title('Radial Velocity Dispersion of the Disk')
def veldisp_radial_disk_model(r):
    c = 0.01
    return c* np.exp(-np.sqrt(r**2 + (r_d**2)/8 )/r_d)
rdisk = np.linspace(min(R_f), 23, 1000).tolist()
veldisp_r_d_model=[]
for r in rdisk:
    veldisp_r_d_model.append(veldisp_radial_disk_model(r))
ax.plot(rdisk, veldisp_r_d_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)

###########################
fig = plt.figure(29, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('R', fontsize=20)
plt.ylabel('$ \log \\left( \sigma_\phi ^2 \\right ) $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax.scatter(rad_median_00, veldisp_azi_d_00,s = 5, label = 'Initial Snapshot')
ax.scatter(rad_median_01, veldisp_azi_d_01,s = 5, c = 'orange', label = '1st Snapshot')
ax.scatter(rad_median_02, veldisp_azi_d_02,s = 5, c = 'magenta', label = '2nd Snapshot')
ax.scatter(rad_median_03, veldisp_azi_d_03,s = 5, c = 'green', label = '3rd Snapshot')
ax.scatter(rad_median_f, veldisp_azi_d_f,s = 5, c = 'black', label = 'Final Snapshot')

plt.title('Azimuthal Velocity Dispersion of the Disk')
def veldisp_azi_disk_model(r):
     v_circ = np.sqrt( r * misc.derivative(total_potential_approx, r, dx = 1e-3))
     omega_squared = (v_circ / r)**2 
     K_squared = 3 * omega_squared + misc.derivative(total_potential_approx, r, dx = 1e-3, n=2)
     azimuthal_veldisp = veldisp_radial_disk_model(r) * K_squared/(4*omega_squared)
     return azimuthal_veldisp
rdisk = np.linspace(min(R_f),23, 10000).tolist()
veldisp_r_azi_model=[]
for r in rdisk:
    veldisp_r_azi_model.append(veldisp_azi_disk_model(r))
ax.plot(rdisk, veldisp_r_azi_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)

 ###########################
fig = plt.figure(30, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('z', fontsize=20)
plt.ylabel('$ \log \\left[ \zeta(z)  \\right ] $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax.scatter(z_median_00, z_density_disk_00, s=5, label = 'Initial Snapshot')
ax.scatter(z_median_01, z_density_disk_01, s=5, c='orange', label = '1st Snapshot')
ax.scatter(z_median_02, z_density_disk_02, s=5, c='magenta', label = '2nd Snapshot')
ax.scatter(z_median_03, z_density_disk_03, s=5, c='green', label = '3rd Snapshot')
ax.scatter(z_median_f, z_density_disk_f, s=5, c='black', label = 'Final Snapshot')

ax.plot(z_median_00, z_density_disk_model_00, c = 'red', label = 'Analytical')
plt.title('Vertical Density of the Disk')
plt.legend(fontsize=20)

#######################
fig = plt.figure(31, figsize=(20,20))
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('R', fontsize=20)
plt.ylabel('$ \log \\left[ \Sigma(R)  \\right ] $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax.scatter(R_median_00, R_density_disk_00, s=5, label = 'Initial Snapshot')
ax.scatter(R_median_01, R_density_disk_01, s=5, c='orange', label = '1st Snapshot')
ax.scatter(R_median_02, R_density_disk_02, s=5, c='magenta', label = '2nd Snapshot')
ax.scatter(R_median_03, R_density_disk_03, s=5, c='green', label = '3rd Snapshot')
ax.scatter(R_median_f, R_density_disk_f, s=5, c='black', label = 'Final Snapshot')

ax.plot(Rmod, R_density_disk_model_00, c = 'red', label = 'Analytical')
ax.plot(Rmod, R_density_disk_model_approx_00, c = 'purple', label = 'Spherical Approximation')
plt.title('Radial Density of the Disk')
plt.legend(fontsize=20)

########################
fig = plt.figure(32, figsize=(20,20))
ax = plt.gca()
plt.xlabel('R', fontsize=20)
plt.ylabel('$ v_c $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

absvel_circ_b_f, absvel_circ_b_00, absvel_circ_b_01, absvel_circ_b_02, absvel_circ_b_03 = np.abs(vel_circ_b_f), np.abs(vel_circ_b_00), np.abs(vel_circ_b_01), np.abs(vel_circ_b_02), np.abs(vel_circ_b_03)
absvel_circ_h_f, absvel_circ_h_00, absvel_circ_h_01, absvel_circ_h_02, absvel_circ_h_03 = np.abs(vel_circ_h_f), np.abs(vel_circ_h_00), np.abs(vel_circ_h_01), np.abs(vel_circ_h_02), np.abs(vel_circ_h_03)
absvel_circ_d_f, absvel_circ_d_00, absvel_circ_d_01, absvel_circ_d_02, absvel_circ_d_03 = np.abs(vel_circ_d_f), np.abs(vel_circ_d_00), np.abs(vel_circ_d_01), np.abs(vel_circ_d_02), np.abs(vel_circ_d_03)
ax.scatter(R_f_sorted_00, absvel_circ_d_00, s=5, label = 'Initial Snapshot: Disk')
#ax.scatter(R_01, absvel_circ_d_01, s=5, c='orange', label = '1st Snapshot: Disk')
#ax.scatter(R_02, absvel_circ_d_02, s=5, c='magenta', label = '2nd Snapshot: Disk')
#ax.scatter(R_03, absvel_circ_d_03, s=5, c='green', label = '3rd Snapshot: Disk')
#ax.scatter(R_f, absvel_circ_d_f, s=5, c='black', label = 'Final Snapshot: Disk')

ax.scatter(R_b_00, absvel_circ_b_00, s=5, label = 'Initial Snapshot: Bulge')
#ax.scatter(R_b_01, absvel_circ_b_01, s=5, c='orange', label = '1st Snapshot: Bulge')
#ax.scatter(R_b_02, absvel_circ_b_02, s=5, c='magenta', label = '2nd Snapshot: Bulge')
#ax.scatter(R_b_03, absvel_circ_b_03, s=5, c='green', label = '3rd Snapshot: Bulge')
#ax.scatter(R_b_f, absvel_circ_b_f, s=5, c='black', label = 'Final Snapshot: Bulge')

ax.scatter(R_h_00, absvel_circ_h_00, s=5, label = 'Initial Snapshot: Halo')
#ax.scatter(R_h_00, absvel_circ_h_01, s=5, c='orange', label = '1st Snapshot: Halo')
#ax.scatter(R_h_02, absvel_circ_h_02, s=5, c='magenta', label = '2nd Snapshot: Halo')
#ax.scatter(R_h_03, absvel_circ_h_03, s=5, c='green', label = '3rd Snapshot: Halo')
#ax.scatter(R_h_f, absvel_circ_h_f, s=5, c='black', label = 'Final Snapshot: Halo')

#ax.plot(Rmod, R_density_disk_model_00, c = 'red', label = 'Analytical')
def v_circ_total(r):
    v_circ = np.sqrt( r * misc.derivative(total_potential_approx, r, dx = 1e-4))
    return v_circ

def v_circ_disk(r):
    v_circ = np.sqrt( r * misc.derivative(disk_potential_approx, r, dx= 1e-4))
    return v_circ

def v_circ_bulge(r):
    v_circ = np.sqrt( r * misc.derivative(bulge_potential, r, dx= 1e-4))
    return v_circ

def v_circ_halo(r):
    v_circ = np.sqrt( r * misc.derivative(halo_potential, r, dx= 1e-4))
    return v_circ

rmodel = np.linspace(0.1,20,1000).tolist()
v_circ_tot_model = []
v_circ_d_model = []
v_circ_b_model = []
v_circ_h_model = []
for r in rmodel:
    v_circ_tot_model.append(v_circ_total(r))
    v_circ_d_model.append(v_circ_disk(r))
    v_circ_b_model.append(v_circ_bulge(r))
    v_circ_h_model.append(v_circ_halo(r))
    
ax.plot(rmodel, v_circ_tot_model, c = 'black', label = 'Analytical Total')
ax.plot(rmodel, v_circ_d_model, c = 'blue', label = 'Analytical Disk')
ax.plot(rmodel, v_circ_b_model, c = 'red', label = 'Analytical Bulge')
ax.plot(rmodel, v_circ_h_model, c = 'purple', label = 'Analytical Halo')
plt.title('Circular Velocity Contributions')
plt.legend(fontsize=20)

######################
fig = plt.figure(33, figsize=(20,20))
ax = plt.gca()
plt.xlabel('R', fontsize=20)
#plt.ylabel('$ \\beta $', fontsize=20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)


    
ax.scatter(rad_median_00, anisotropy_d_00,s = 5, label = 'Initial Snapshot')
ax.scatter(rad_median_01, anisotropy_d_01,s = 5, c = 'orange', label = '1st Snapshot')
ax.scatter(rad_median_02, anisotropy_d_02,s = 5, c = 'magenta', label = '2nd Snapshot')
ax.scatter(rad_median_03, anisotropy_d_03,s = 5, c = 'green', label = '3rd Snapshot')
ax.scatter(rad_median_f, anisotropy_d_f,s = 5, c = 'black', label = 'Final Snapshot')

#ax.plot(Rmod, R_density_disk_model_approx_00, c = 'purple', label = 'Spherical Approximation')
plt.title('Anisotropy of the Disk')
plt.legend(fontsize=20)
