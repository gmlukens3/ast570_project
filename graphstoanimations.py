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

G=4.30091*10**-6 #G in terms of km/s ^2 M_sun ^-1 kpc

Z_d = 0.15 #scale height in kpc
r_d = 2 #scale length in kpc
a_h = 5
a_b = 0.25
Sigma_0 = 1
m_star = 1 # one solar mass to get it into SI units

N_disk = 100000#use 10**8 when actually running
N_halo = 10000 
N_bulge = 18790 #18790 for bulge halo disk, 0 for halo disk, 0 for disk

#have to be careful with velocities, since positions are in kpc, so velocities should be kpc/s
m_to_kpc = 3.08567758128e19 #meters per kpc

M_d = m_star * N_disk 
M_b = m_star * N_bulge 
M_h = m_star * N_halo 

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

def halo_density(r):
    density = M_h  / (4*np.pi*a_h**3) / ((r/a_h) * (1+r/a_h)**2)
    return density

def bulge_density(r):
    density = M_b  / (4*np.pi*a_b**3) / ((r/a_b)**1 * (1+r/a_b)**3)
    return density

def disk_density(r,z):
    #r2 = r**2 + z**2
    density = M_d  / (4*np.pi*r_d**2 * Z_d) * np.exp(-r/r_d) / (np.cosh(z/Z_d))**2
    return density

def disk_density_approx(r,z):
    #r2 = r**2 + z**2
    density = M_d  / (4*np.pi*r_d**2 * Z_d * r) * np.exp(-r/r_d) / (np.cosh(z/Z_d))**2
    return density

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


 
#density plots
##############################
# bulge density plots


############## disk density plots


R_00 = []
for i in range(0,len(disk_x_00)):
    R_00.append(np.sqrt(disk_x_00[i]**2 + disk_y_00[i]**2))
        
density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts_00 = []
#radius_master_list = []
#indices_master_list = []
#z

               
z_density_disk_model = []
z_median = []
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
    z_median.append(z_m)
    r_m = median(Rsort_00[index:index+100])
        
    
    c = 3.85 ##normalization
    z_density_disk_model.append(c * disk_density(r_mean_d_00, z_m))
    index += 1
    if index == (len(zdisk_00)-100):
        break

 
R_density_disk_model = []
R_density_disk_model_approx = []
R_median = []
Rsort_00 = R_00
z_00 = disk_z_00
Rsort_00,z_00= zip(*sorted(zip(Rsort_00, z_00)))
z_00 = list(z_00)
Rsort_00 = list(Rsort_00)

index = 0
for r in Rsort_00:
    r_1 = Rsort_00[index]
    r_2 = Rsort_00[index+1000]
    dz = max(z_00[index:index+1000]) - min(z_00[index:index+1000])
    r_m = np.median(Rsort_00[index:index+1000])
    R_median.append(r_m)
    c= M_d / (4*np.pi*r_d**2 *Z_d)
    z_m = np.mean(z_00[index:index+1000])
    R_density_disk_model.append(c*np.exp(-r_m/r_d)/(np.cosh(z_m/Z_d)**2))
    R_density_disk_model_approx.append(c*np.exp(-r_m/r_d)/r_m/(np.cosh(z_m/Z_d)**2))
    
    
    index += 1
    if index == (len(Rsort_00)-1000):
        break
    
    
###################################### 
#after initialized 000 snapshot w/ models, use for loop over each snapshot saving images as going alone
fname = []
root_dir = 'C:\\Users\\doubl\\OneDrive\\Desktop\\'
for i in range(0,241):
    
    if i < 10:
        fname.append(root_dir + 'snapshots\\snapshot_00' + str(i))
    
    if i >= 10 and i < 100:
        fname.append(root_dir + 'snapshots\\snapshot_0' + str(i))
    
    if i >= 100:
        fname.append(root_dir + 'snapshots\\snapshot_' + str(i))
figindex = 0
for P in range(0,241): #start at 125, end at 240 next time!
    time_snap = np.round(0.05*P, 2)
    
    if time_snap < 10:
        time_snap = str(0) + f'{time_snap:.2f}'
    else: time_snap = f'{time_snap:.2f}'
    
    
    halo_xyz_00 = readsnap(fname[P], 'pos', 'dm')
    disk_xyz_00 = readsnap(fname[P], 'pos', 'disk')
    bulge_xyz_00 = readsnap(fname[P], 'pos', 'bulge')
    
    halo_vel_00 = readsnap(fname[P], 'vel', 'dm')
    disk_vel_00 = readsnap(fname[P], 'vel', 'disk')
    bulge_vel_00 = readsnap(fname[P], 'vel', 'bulge')
    
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
    density_bulge_model = []
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
                   
    z_density_disk_00 = []
    
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
        index += 1
        if index == (len(zdisk_00)-100):
            break
    
    R_density_disk_00 = []
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
        dz = max(z_00[index:index+1000]) - min(z_00[index:index+1000])
        r_m = np.median(Rsort_00[index:index+1000])
        R_median_00.append(r_m)
        density = 1000 / (dz* np.pi*(r_2**2 - r_1**2))
        R_density_disk_00.append(density)
        index += 1
        if index == (len(Rsort_00)-1000):
            break
        
         
    anisotropy_d_00 = []
    for i in range(0,len(veldisp_r_d_00)):
        anisotropy_d_00.append(1 - (veldisp_azi_d_00[i] + veldisp_z_d_00[i])/ veldisp_r_d_00[i])
        
        
        
##############plotting positions
    fig = plt.figure(figindex+1, figsize = (10,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(0, 45) #change viewing angle to cross-sectional
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    plot = ax.scatter(halo_x_00, halo_y_00, halo_z_00, color = 'green', s=0.2, alpha=0.5)
    plot2 = ax.scatter(disk_x_00, disk_y_00, disk_z_00, s=0.2, alpha=0.5)
    plot3 = ax.scatter(bulge_x_00, bulge_y_00, bulge_z_00, color = 'red', s=0.2, alpha=0.5)
    plt.title('t = ' + time_snap)
    plt.savefig(root_dir + 'a570project//xyzside//xyzside_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    #########
    
    fig = plt.figure(figindex+2, figsize = (10,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(30, 45) #change viewing angle to cross-sectional
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    plot = ax.scatter(halo_x_00, halo_y_00, halo_z_00, color = 'green', s=0.2, alpha=0.5)
    plot2 = ax.scatter(disk_x_00, disk_y_00, disk_z_00, s=0.2, alpha=0.2)
    plot3 = ax.scatter(bulge_x_00, bulge_y_00, bulge_z_00, color = 'red', s=0.2, alpha=0.5)
    plt.title('t = ' + time_snap)
    plt.savefig(root_dir + 'a570project//xyzmid//xyzmid_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ########################
    fig = plt.figure(figindex+3, figsize = (10,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.view_init(80, 45) #change viewing angle to cross-sectional
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)
    ax.set_xlabel('x [kpc]')
    ax.set_ylabel('y [kpc]')
    ax.set_zlabel('z [kpc]')
    plot = ax.scatter(halo_x_00, halo_y_00, halo_z_00, color = 'green', s=0.2, alpha=0.5)
    plot2 = ax.scatter(disk_x_00, disk_y_00, disk_z_00, s=0.2, alpha=0.5)
    plot3 = ax.scatter(bulge_x_00, bulge_y_00, bulge_z_00, color = 'red', s=0.2, alpha=0.5)
    plt.title('t = ' + time_snap)
    plt.savefig(root_dir + 'a570project//xyztop//xyztop_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ##########################
    fig = plt.figure(figindex+4,figsize=(10,10))
    plt.title('Density of the Halo', fontsize=20)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('r', fontsize=20)
    plt.ylabel('$ \log \\left[ \\rho_h(r) \\right ] $', fontsize=20)
    ax.set_xlim(-0.4,27)
    ax.set_ylim(10**-2, 10**3)
    density_halo_model_00 = []
    rhalo = np.linspace(0.1, 26, 1000).tolist()
    for r in rhalo:
        density_halo_model_00.append(halo_density(r))
        
    ax.scatter(radius_median_halo_00, density_halo_00,s = 10, label = 't = ' + time_snap)
       
    plt.plot(rhalo, density_halo_model_00, color = 'red', label = 'Analytical')
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//halodensities//halodensity_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    #############
    fig = plt.figure(figindex+5, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_ylim(10**-2.5, 10**-1)
    ax.set_xlim(-0.4,26.5)
    ax.scatter(radius_median_halo_00, veldisp_r_h_00,s = 10, label = 't = ' + time_snap)
       
    plt.xlabel('r', fontsize=20)
    plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)
    plt.title('Radial Velocity Dispersion of the Halo', fontsize=20)
    def veldisp_radial_halo_model(r):
        integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
        return integral / halo_density(r)
    rhalo = np.linspace(0.5, 26, 1000).tolist()
    veldisp_r_h_model=[]
    for r in rhalo:
        veldisp_r_h_model.append(veldisp_radial_halo_model(r))
    ax.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//haloradveldisp//haloradveldisp_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ######################
    fig = plt.figure(figindex+6, figsize=(10,10))
    plt.title('Density of the Bulge', fontsize=20)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('r', fontsize=20)
    plt.ylabel('$ \log \\left[ \\rho_b(r) \\right ] $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_xlim(-0.2,8)
    ax.set_ylim(0.5*10**-1, 10**6.1)
    rbulge = np.linspace(0.1, 7, 1000).tolist()
    for r in rbulge:
        density_bulge_model.append(bulge_density(r))
    
    ax.scatter(radius_median_bulge_00, density_bulge_00,s = 10, label = 't = ' + time_snap)
    ax.plot(rbulge, density_bulge_model, color = 'red', label = 'Analytical')
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//bulgedensities//bulgedensity_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ########################
    fig = plt.figure(figindex+7, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('r', fontsize=20)
    plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)
    
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.scatter(radius_median_bulge_00, veldisp_r_b_00,s = 10, label = 't = ' + time_snap)
    
    plt.title('Radial Velocity Dispersion of the Bulge', fontsize =20)
    def veldisp_radial_bulge_model(r):
        integral, error = integrate.quad(hernquist_dispersion_equation_bulge, r, np.infty)
        return integral / bulge_density(r)
    rbulge = np.linspace(0.01, 7.5, 1000).tolist()
    veldisp_r_b_model=[]
    for r in rbulge:
        veldisp_r_b_model.append(veldisp_radial_bulge_model(r))
    ax.plot(rbulge, veldisp_r_b_model, label = 'Analytical', color = 'red')
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//bulgeradveldisp//bulgeradveldisp_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ###########
   
    
    #############
    fig = plt.figure(figindex+8, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('R', fontsize=20)
    plt.ylabel('$ \log  \\left(  \sigma_r^2  \\right )$', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_xlim(-0.5,24)
    ax.set_ylim(10**-7.3, 10**-1.8)
    ax.scatter(rad_median_00, veldisp_r_d_00,s = 5, label = 't = ' + time_snap)
        
    plt.title('Radial Velocity Dispersion of the Disk', fontsize =20)
    def veldisp_radial_disk_model(r):
        c = 0.01
        return c* np.exp(-np.sqrt(r**2 + (r_d**2)/8 )/r_d)
    rdisk = np.linspace(0.01, 23, 1000).tolist()
    veldisp_r_d_model=[]
    for r in rdisk:
        veldisp_r_d_model.append(veldisp_radial_disk_model(r))
    ax.plot(rdisk, veldisp_r_d_model, label = 'Analytical', color = 'red')
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//diskradveldisp//diskradveldisp_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ###########################
    fig = plt.figure(figindex+9, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('R', fontsize=20)
    plt.ylabel('$ \log \\left( \sigma_\phi ^2 \\right ) $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_xlim(-0.2,25)
    ax.set_ylim(10**-8, 10**-2)
    ax.scatter(rad_median_00, veldisp_azi_d_00,s = 5, label = 't = ' + time_snap)
        
    plt.title('Azimuthal Velocity Dispersion of the Disk', fontsize =20)
    
    def veldisp_azi_disk_model(r):
         v_circ = np.sqrt( r * misc.derivative(total_potential_approx, r, dx = 1e-3))
         omega_squared = (v_circ / r)**2 
         K_squared = 3 * omega_squared + misc.derivative(total_potential_approx, r, dx = 1e-3, n=2)
         azimuthal_veldisp = veldisp_radial_disk_model(r) * K_squared/(4*omega_squared)
         return azimuthal_veldisp
    rdisk = np.linspace(0.1,23, 1000).tolist()
    veldisp_r_azi_model=[]
    for r in rdisk:
        veldisp_r_azi_model.append(veldisp_azi_disk_model(r))
    ax.plot(rdisk, veldisp_r_azi_model, label = 'Analytical', color = 'red')
    plt.legend(fontsize=20)
    
    plt.savefig(root_dir + 'a570project//diskaziveldisp//diskaziveldisp_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
     ###########################
    fig = plt.figure(figindex+10, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_ylim(1,10**4)
    ax.set_xlim(-0.7,0.7)
    plt.xlabel('z', fontsize=20)
    plt.ylabel('$ \log \\left[ \zeta(z)  \\right ] $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    
    ax.scatter(z_median_00, z_density_disk_00, s=5, label = 't = ' + time_snap)
        
    ax.plot(z_median, z_density_disk_model, c = 'red', label = 'Analytical')
    plt.title('Vertical Density of the Disk', fontsize =25)
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//diskzdensities//diskzdensity_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    #######################
    fig = plt.figure(figindex+11, figsize=(10,10))
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlabel('R', fontsize=20)
    plt.ylabel('$ \log \\left[ \Sigma(R)  \\right ] $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_xlim(-0.5,18)
    ax.set_ylim(0.05, 10**6)
    
    ax.scatter(R_median_00, R_density_disk_00, s=5, label = 't = ' + time_snap)
      
    ax.plot(R_median, R_density_disk_model, c = 'red', label = 'Analytical')
    ax.plot(R_median, R_density_disk_model_approx, c = 'purple', label = 'Spherical Approximation')
    plt.title('Radial Density of the Disk', fontsize =25)
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//diskraddensities//diskraddensity_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ######################## circular velocity
    fig = plt.figure(figindex+12, figsize=(10,10))
    ax = plt.gca()
    plt.xlabel('R', fontsize=20)
    plt.ylabel('$ v_c $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.set_xlim(-0.3,20)
    ax.set_ylim(-0.01,0.7)
    absvel_circ_b_00=  np.abs(vel_circ_b_00)
    absvel_circ_h_00=  np.abs(vel_circ_h_00)
    absvel_circ_d_00=  np.abs(vel_circ_d_00)
    
    
    ax.scatter(R_f_sorted_00, absvel_circ_d_00, s=5, label = 'Disk', alpha=0.5)
  
    ax.scatter(R_b_00, absvel_circ_b_00, s=5, label = 'Bulge', alpha=0.5)
        
    ax.scatter(R_h_00, absvel_circ_h_00, s=5, label = 'Halo', alpha=0.5)
        
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
    
    #rmodel = np.linspace(0.1,20,1000).tolist()
    #v_circ_tot_model = []
    #v_circ_d_model = []
    #v_circ_b_model = []
    #v_circ_h_model = []
    #for r in rmodel:
        #v_circ_tot_model.append(v_circ_total(r))
        #v_circ_d_model.append(v_circ_disk(r))
        #v_circ_b_model.append(v_circ_bulge(r))
        #v_circ_h_model.append(v_circ_halo(r))
        
    #ax.plot(rmodel, v_circ_tot_model, c = 'black', label = 'Total (no dark matter)')
    #ax.plot(rmodel, v_circ_d_model, c = 'blue', label = 'Disk')
    #ax.plot(rmodel, v_circ_b_model, c = 'red', label = 'Bulge')
    #ax.plot(rmodel, v_circ_h_model, c = 'purple', label = 'Halo')
    plt.title('Circular Velocity (t = ' + time_snap + ')', fontsize =20)
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//vcirc//vcirc_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    ######################
    fig = plt.figure(figindex+13, figsize=(10,10))
    ax = plt.gca()
    plt.xlabel('R', fontsize=20)
    plt.ylabel(r'$ \beta $', fontsize=20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)    
   
    
    

    ax.scatter(rad_median_00[:len(rad_median_00)-150], anisotropy_d_00[:len(rad_median_00)-150],s = 5, label = 't = ' + time_snap)
    
    #ax.plot(Rmod, R_density_disk_model_approx_00, c = 'purple', label = 'Spherical Approximation')
    plt.title('Anisotropy of the Disk', fontsize =20)
    plt.legend(fontsize=20)
    plt.savefig(root_dir + 'a570project//anisotropy//anisotropy_' + str(P) + '.png' , format = 'png', dpi = 500, transparent = True)
    plt.close(fig)
    figindex += 14