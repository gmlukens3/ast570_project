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
from galpy import potential
from sympy import symbols, diff

G=4.30091*10**-6 #G in terms of km/s ^2 M_sun ^-1 kpc

Z_d = 0.15 #scale height in kpc
r_d = 2 #scale length in kpc
a_h = 5
a_b = 0.25
Sigma_0 = 1
m_star = 1 # one solar mass to get it into SI units

N_disk = 100000 #use 10**8 when actually running
N_halo = 10000
N_bulge = 20000 #use 10**5 when actually running

#have to be careful with velocities, since positions are in kpc, so velocities should be kpc/s
m_to_kpc = 3.08567758128e19 #meters per kpc

M_d = m_star * N_disk 
M_b = m_star * N_bulge 
M_h = m_star * N_halo 

n = []
R = []

theta = []
R = []

x_d = []
y_d = []
z_d = []

x_b = []
y_b = []
z_b = []

x_h = []
y_h = []
z_h = []

phi_b = []
phi_h = []
theta_h = []
theta_b = []
theta_d = []

r_b = []
r_h = []

veldisp_r_b = []
veldisp_r_d = []
veldisp_azi_d = []
veldisp_z_d = []
veldisp_r_h = []

v_R_d = []
v_circ_d=[]
v_azi_d = []
anisotropy_d = []

vel_x_h = []
vel_y_h = []
vel_z_h = []

vel_x_b = []
vel_y_b = []
vel_z_b = []

vel_x_d = []
vel_y_d = []
vel_z_d = []

v_esc_h = []
v_esc_b = []
v_esc_d = []

speed_h = []
speed_b = []
speed_d = []

#density profiles:
    
def disk_density(r,z):
    r2 = r**2 + z**2
    density = M_d  / (4*np.pi*r_d**2 * Z_d) * np.exp(-r/r_d) / (np.cosh(z/Z_d))**2
    return density

def bulge_density(r):
    density = M_b  / (2*np.pi*a_b**3) / ((r/a_b)**1 * (1+r/a_b)**3)
    return density

def halo_density(r):
    density = M_h  / (4*np.pi*a_h**3) / ((r/a_h) * (1+r/a_h)**2)
    return density

#mass profiles:
    
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
#potential profiles

#use the spherical approximation for the disk potential, we assume the potential does not differ drastically with z
def disk_potential_approx(r):
    potential = -G*M_d/ r * (1-np.exp(-r/r_d))
    return potential

def disk_potential(R,z):
    A = M_d  / (4*np.pi*r_d**2 * Z_d)
    diskpotential = potential.MN3ExponentialDiskPotential(amp = A, hr = r_d, hz = Z_d, sech=True)
    return potential.evaluate(R,z,diskpotential)

def bulge_potential(r): 
    potential = -G * M_b * 1/(r+a_b)
    return potential 

def halo_potential(r):
    potential = -G * M_h * np.log(1+r/a_h)/(r)
    return potential

def total_potential_approx(r):
    return disk_potential_approx(r) + bulge_potential(r) + halo_potential(r)

def total_potential(R):
    return disk_potential(R) + bulge_potential(R) + halo_potential(R)


#disk positions (using potential w/ exponential disk, sech^2 z)
for i in range(0,N_disk):
    
   
    q_t = random.uniform(0, 1)
    q_r = random.uniform(0, 8.389) #279.635 determined by finding where R function hits 0 so as to avoid negative R values
    q_z = random.uniform(-1, 1)
    #derived random values for all 3 spherical coordinates
    theta_d.append(2*np.pi*q_t)
    
    #converting to Cartesian coordinates for plotting purposes (all in terms of r/a)

    z_d.append(0.5*Z_d*np.log((q_z+1)/(1-q_z)))
    R.append(5*np.real(-r_d * sci.lambertw((q_r- 1)/np.e) + r_d ))
    x_d.append(R[i]*np.cos(theta_d[i])) 
    y_d.append(R[i]*np.sin(theta_d[i]))
    
    r_disk = np.real(R[i])
    
    #def rotvel(r):
        #rad = r/(2*r_d)
        #velocitymodel = np.sqrt(M_d*r**2/(4*r_d**3) * ((sci.kv(0,rad)*sci.iv(0,rad))-(sci.kv(1,rad)*sci.iv(1,rad) )))
        #return velocitymodel
    v_circ = np.sqrt( r_disk * misc.derivative(total_potential_approx, r_disk, dx = 1e-3))
    #v_circ = rotvel(r_disk)
    v_circ_d.append(v_circ)
    #circular frequency
    omega_squared = (v_circ / r_disk)**2 
    #epicycle freq. squared (hernquist)
    K_squared = 3 * omega_squared + misc.derivative(total_potential_approx, r_disk, dx = 1e-3, n=2)
    
    #hernquist 2.29 softened radial dispersion equation
    c = 0.01
    radial_veldisp = c * np.exp(-np.sqrt(r_disk**2 + (r_d**2)/8 )/r_d)
    veldisp_r_d.append(radial_veldisp)
    
    
    #azimuthal dispersion given by hernquist 2.26 and mean azimuthal velocity given by 2.28
    v_mean_azi = np.sqrt(abs(radial_veldisp * (1 - K_squared/(4*omega_squared) - 2*r_disk/r_d) + v_circ**2))
    
    
    azimuthal_veldisp = radial_veldisp * K_squared/(4*omega_squared)
    
    veldisp_azi_d.append(azimuthal_veldisp)
    
    #z direction dispersion given by herqnuist 2.22
    z_veldisp = np.pi * G * Sigma_0 * np.exp(-r_disk/r_d) * Z_d
    veldisp_z_d.append(z_veldisp)
    
    noise_R = np.random.normal(0, np.sqrt(radial_veldisp))
    v_R = 0 + noise_R
    v_R_d.append(v_R)
    
    v_azi = v_mean_azi + np.random.normal(0, np.sqrt(azimuthal_veldisp)) #assuming mean 0 velocity in azimuthal and z directions
    v_azi_d.append(v_azi)
    v_z = 0 + np.random.normal(0, np.sqrt(z_veldisp))
    
    vel_x_d.append(v_R*np.cos(theta_d[i]) - v_azi*np.sin(theta_d[i]))
    vel_y_d.append(v_R*np.sin(theta_d[i]) + v_azi*np.cos(theta_d[i]))
    vel_z_d.append(v_z)
    speed_d.append(np.sqrt(vel_x_d[i]**2 +vel_y_d[i]**2+vel_z_d[i]**2 ))
    
    anisotropy_d.append(1 - (azimuthal_veldisp + z_veldisp )/radial_veldisp)
#bulge positions (Hernquist 1990 profile)
for i in range(0,N_bulge):
   
    #random values cooresponding to each coordinate in spherical coord., from 0 to 1
    q_p = random.uniform(0, 1)
    q_t = random.uniform(0, 1)
    q_r = random.uniform(0, 1)
    
    #derived random values for all 3 spherical coordinates
    phi_bulge = 2*np.pi*q_p
    theta_bulge = np.arccos(1-2*q_t)
    r_bulge = a_b*(q_r + np.sqrt(q_r))/(1-q_r)
    
    #converting to Cartesian coordinates for plotting purposes (all in terms of r/a)
    if r_bulge < 30*a_b: #cutoff at 30*0.25 = 7.5 kpc
        r_b.append(r_bulge)
        x_b.append(r_bulge*np.sin(phi_bulge)*np.cos(theta_bulge)) 
        y_b.append(r_bulge*np.sin(phi_bulge)*np.sin(theta_bulge))
        z_b.append(r_bulge*np.cos(phi_bulge))

        integral, error = integrate.quad(hernquist_dispersion_equation_bulge, r_bulge, np.infty)
        radial_veldisp = 1/bulge_density(r_bulge) * integral
        veldisp_r_b.append(radial_veldisp)
        
        potential_b = total_potential_approx(r_bulge)
        
        v_esc = np.sqrt(2*(abs((potential_b))))
        
        #normalize the distribution function to 1
            
        q_a = random.uniform(0, 1)
        q_b = random.uniform(0, 1)
        
        alpha = np.arccos(1-2*q_a)
        beta = 2*np.pi*q_b
        
        #v_circ = np.sqrt( r_bulge * misc.derivative(total_potential_approx, r_bulge, dx = 1e-3))
        #noise = np.random.normal(0, np.sqrt(radial_veldisp))
        #v_r_b = 0 + noise
        
        #veldisp_beta_b = radial_veldisp
        #veldisp_alpha_b = radial_veldisp
        #while abs(v) > 0.95 * v_esc:
                #v = np.random.normal(v_circ, radial_veldisp)
                #if abs(v) <= 0.95 * v_esc:
                    #break
        #v_beta = np.random.uniform(-v_circ,v_circ)
        
        #does sign matter for these?????
        #v_alpha_sign = random.random()
        #if v_alpha_sign < 0.5: v_alpha_sign = -1
        #else: v_alpha_sign = 1
        
        #v_alpha = np.sqrt((v_circ)**2 - v_beta**2)
        
        #noise_a = np.random.normal(0, np.sqrt(veldisp_alpha_b))
        #noise_b = np.random.normal(0, np.sqrt(veldisp_beta_b))
        
        #v_alpha = v_alpha + noise_a
        #v_beta = v_beta + noise_b
        
        #v_esc_b.append(v_esc)
        
        v = np.asscalar(st.maxwell.rvs(size = 1))
    
   
        while abs(v) > 1 * (v_esc/radial_veldisp**0.5):
                v = np.asscalar(st.maxwell.rvs(size = 1))
                if abs(v) <= 1 * (v_esc/radial_veldisp**0.5):
                    break
                    
        speed = v * radial_veldisp**0.5
        speed_b.append(speed)
        v_esc_b.append(v_esc)
        
        vel_x_b.append(speed*np.sin(alpha)*np.cos(beta)) 
        vel_y_b.append(speed*np.sin(alpha)*np.sin(beta))
        vel_z_b.append(speed*np.cos(alpha))
            #check units on these
    
    
#halo positions (NFW)

for i in range(0,N_halo):
   
    #random values cooresponding to each coordinate in spherical coord., from 0 to 1
    q_p = random.uniform(0, 1)
    q_t = random.uniform(0, 1)
    q_r = random.uniform(0, 1)
    
    #derived random values for all 3 spherical coordinates
    phi_halo = 2*np.pi*q_p
    theta_halo = np.arccos(1-2*q_t)
    r_halo = -a_h*(((sci.lambertw(-np.exp((-q_r-1)/1))+1 )/ sci.lambertw(-np.exp(-(q_r+1)/1))))
    r_halo = r_halo.real
    
    #converting to Cartesian coordinates for plotting purposes (all in terms of r/a)
    #if r_halo < 5*a_h:
    r_h.append(r_halo)
    x_h.append(r_halo*np.sin(phi_halo)*np.cos(theta_halo)) 
    y_h.append(r_halo*np.sin(phi_halo)*np.sin(theta_halo))
    z_h.append(r_halo*np.cos(phi_halo))
    
    
    #velocity assignment (following hernquist procedure for distribution function)
    integral, error = integrate.quad(hernquist_dispersion_equation_halo, r_halo, np.infty)
    radial_veldisp = 1/halo_density(r_halo) * integral
    veldisp_r_h.append(radial_veldisp)
    
    potential_h = total_potential_approx(r_halo)
    
    v_esc = np.sqrt(2*(abs((potential_h))))
    
    #normalize the distribution function to 1
        
    q_a = random.uniform(0, 1)
    q_b = random.uniform(0, 1)
    
    alpha = np.arccos(1-2*q_a)
    beta = 2*np.pi*q_b
    
    
    v = np.asscalar(st.maxwell.rvs(size = 1))
    
   
    while abs(v) > 0.95 * (v_esc/radial_veldisp**0.5):
            v = np.asscalar(st.maxwell.rvs(size = 1))
            if abs(v) <= 0.95 * (v_esc/radial_veldisp**0.5):
                break
                
    speed = v * radial_veldisp**0.5
    speed_h.append(speed)
    v_esc_h.append(v_esc)
    
    vel_x_h.append(speed*np.sin(alpha)*np.cos(beta)) 
    vel_y_h.append(speed*np.sin(alpha)*np.sin(beta))
    vel_z_h.append(speed*np.cos(alpha))
    
    #check units on these

#####positions and velocities lists/export to .txt file (used for GADGET)

x_positions_0 = x_h + x_b
y_positions_0 = y_h + y_b
z_positions_0 = z_h + z_b

x_velocities_0 = vel_x_h + vel_x_b 
y_velocities_0 = vel_y_h + vel_y_b 
z_velocities_0 = vel_z_h + vel_z_b 

position_data_0 = zip(x_positions_0, y_positions_0, z_positions_0)
velocity_data_0 = zip(x_velocities_0, y_velocities_0, z_velocities_0)

x_positions_1 = x_d[0:9999]
y_positions_1 = y_d[0:9999]
z_positions_1 = z_d[0:9999]

x_velocities_1 = vel_x_d[0:9999]
y_velocities_1 = vel_y_d[0:9999] 
z_velocities_1 = vel_z_d[0:9999]
   
position_data_1 = zip(x_positions_1, y_positions_1, z_positions_1)
velocity_data_1 = zip(x_velocities_1, y_velocities_1, z_velocities_1)

x_positions_2 = x_d[10000:19999]
y_positions_2 = y_d[10000:19999]
z_positions_2 = z_d[10000:19999]

x_velocities_2 = vel_x_d[10000:19999]
y_velocities_2 = vel_y_d[10000:19999] 
z_velocities_2 = vel_z_d[10000:19999]
   
position_data_2 = zip(x_positions_2, y_positions_2, z_positions_2)
velocity_data_2 = zip(x_velocities_2, y_velocities_2, z_velocities_2)

x_positions_3 = x_d[20000:29999]
y_positions_3 = y_d[20000:29999]
z_positions_3 = z_d[20000:29999]

x_velocities_3 = vel_x_d[20000:29999]
y_velocities_3 = vel_y_d[20000:29999] 
z_velocities_3 = vel_z_d[20000:29999]
   
position_data_3 = zip(x_positions_3, y_positions_3, z_positions_3)
velocity_data_3 = zip(x_velocities_3, y_velocities_3, z_velocities_3)

x_positions_4 = x_d[30000:39999]
y_positions_4 = y_d[30000:39999]
z_positions_4 = z_d[30000:39999]

x_velocities_4 = vel_x_d[30000:39999]
y_velocities_4 = vel_y_d[30000:39999] 
z_velocities_4 = vel_z_d[30000:39999]
   
position_data_4 = zip(x_positions_4, y_positions_4, z_positions_4)
velocity_data_4 = zip(x_velocities_4, y_velocities_4, z_velocities_4)

x_positions_5 = x_d[40000:49999]
y_positions_5 = y_d[40000:49999]
z_positions_5 = z_d[40000:49999]

x_velocities_5 = vel_x_d[40000:49999]
y_velocities_5 = vel_y_d[40000:49999] 
z_velocities_5 = vel_z_d[40000:49999]
   
position_data_5 = zip(x_positions_5, y_positions_5, z_positions_5)
velocity_data_5 = zip(x_velocities_5, y_velocities_5, z_velocities_5)

x_positions_6 = x_d[50000:59999]
y_positions_6 = y_d[50000:59999]
z_positions_6 = z_d[50000:59999]

x_velocities_6 = vel_x_d[50000:59999]
y_velocities_6 = vel_y_d[50000:59999] 
z_velocities_6 = vel_z_d[50000:59999]
   
position_data_6 = zip(x_positions_6, y_positions_6, z_positions_6)
velocity_data_6 = zip(x_velocities_6, y_velocities_6, z_velocities_6)

x_positions_7 = x_d[60000:69999]
y_positions_7 = y_d[60000:69999]
z_positions_7 = z_d[60000:69999]

x_velocities_7 = vel_x_d[60000:69999]
y_velocities_7 = vel_y_d[60000:69999] 
z_velocities_7 = vel_z_d[60000:69999]
   
position_data_7 = zip(x_positions_7, y_positions_7, z_positions_7)
velocity_data_7 = zip(x_velocities_7, y_velocities_7, z_velocities_7)

x_positions_8 = x_d[70000:79999]
y_positions_8 = y_d[70000:79999]
z_positions_8 = z_d[70000:79999]

x_velocities_8 = vel_x_d[70000:79999]
y_velocities_8 = vel_y_d[70000:79999] 
z_velocities_8 = vel_z_d[70000:79999]
   
position_data_8 = zip(x_positions_8, y_positions_8, z_positions_8)
velocity_data_8 = zip(x_velocities_8, y_velocities_8, z_velocities_8)

x_positions_9 = x_d[80000:89999]
y_positions_9 = y_d[80000:89999]
z_positions_9 = z_d[80000:89999]

x_velocities_9 = vel_x_d[80000:89999]
y_velocities_9 = vel_y_d[80000:89999] 
z_velocities_9 = vel_z_d[80000:89999]
   
position_data_9 = zip(x_positions_9, y_positions_9, z_positions_9)
velocity_data_9 = zip(x_velocities_9, y_velocities_9, z_velocities_9)

x_positions_10 = x_d[90000:99999]
y_positions_10 = y_d[90000:99999]
z_positions_10 = z_d[90000:99999]

x_velocities_10 = vel_x_d[90000:99999]
y_velocities_10 = vel_y_d[90000:99999] 
z_velocities_10 = vel_z_d[90000:99999]
   
position_data_10 = zip(x_positions_10, y_positions_10, z_positions_10)
velocity_data_10 = zip(x_velocities_10, y_velocities_10, z_velocities_10)


df1_0 = pd.DataFrame(data=position_data_0)
df2_0 = pd.DataFrame(data=velocity_data_0)

df1_0.to_csv('positions_0.txt',  sep = ' ', index=False, header=None)
df2_0.to_csv('velocities_0.txt', sep = ' ', index=False, header=None)

df1_1 = pd.DataFrame(data=position_data_1)
df2_1 = pd.DataFrame(data=velocity_data_1)

df1_1.to_csv('positions_1.txt',  sep = ' ', index=False, header=None)
df2_1.to_csv('velocities_1.txt', sep = ' ', index=False, header=None)

df1_2 = pd.DataFrame(data=position_data_2)
df2_2 = pd.DataFrame(data=velocity_data_2)

df1_2.to_csv('positions_2.txt',  sep = ' ', index=False, header=None)
df2_2.to_csv('velocities_2.txt', sep = ' ', index=False, header=None)

df1_3 = pd.DataFrame(data=position_data_3)
df2_3 = pd.DataFrame(data=velocity_data_3)

df1_3.to_csv('positions_3.txt',  sep = ' ', index=False, header=None)
df2_3.to_csv('velocities_3.txt', sep = ' ', index=False, header=None)

df1_4 = pd.DataFrame(data=position_data_4)
df2_4 = pd.DataFrame(data=velocity_data_4)

df1_4.to_csv('positions_4.txt',  sep = ' ', index=False, header=None)
df2_4.to_csv('velocities_4.txt', sep = ' ', index=False, header=None)

df1_5 = pd.DataFrame(data=position_data_5)
df2_5 = pd.DataFrame(data=velocity_data_5)

df1_5.to_csv('positions_5.txt',  sep = ' ', index=False, header=None)
df2_5.to_csv('velocities_5.txt', sep = ' ', index=False, header=None)

df1_6 = pd.DataFrame(data=position_data_6)
df2_6 = pd.DataFrame(data=velocity_data_6)

df1_6.to_csv('positions_6.txt',  sep = ' ', index=False, header=None)
df2_6.to_csv('velocities_6.txt', sep = ' ', index=False, header=None)

df1_7 = pd.DataFrame(data=position_data_7)
df2_7 = pd.DataFrame(data=velocity_data_7)

df1_7.to_csv('positions_7.txt',  sep = ' ', index=False, header=None)
df2_7.to_csv('velocities_7.txt', sep = ' ', index=False, header=None)

df1_8 = pd.DataFrame(data=position_data_8)
df2_8 = pd.DataFrame(data=velocity_data_8)

df1_8.to_csv('positions_8.txt',  sep = ' ', index=False, header=None)
df2_8.to_csv('velocities_8.txt', sep = ' ', index=False, header=None)

df1_9 = pd.DataFrame(data=position_data_9)
df2_9 = pd.DataFrame(data=velocity_data_9)

df1_9.to_csv('positions_9.txt',  sep = ' ', index=False, header=None)
df2_9.to_csv('velocities_9.txt', sep = ' ', index=False, header=None)

df1_10 = pd.DataFrame(data=position_data_10)
df2_10 = pd.DataFrame(data=velocity_data_10)

df1_10.to_csv('positions_10.txt',  sep = ' ', index=False, header=None)
df2_10.to_csv('velocities_10.txt', sep = ' ', index=False, header=None)

#with open('positions.txt', 'w', newline='') as f_output:
    #tsv_output = csv.writer(f_output, delimiter = ' ')
    #tsv_output.writerow(position_data)
    
#with open('velocities.txt', 'w', newline='') as f_output:
    #tsv_output = csv.writer(f_output, delimiter = ' 44')
    #tsv_output.writerow(velocity_data)
################################################################


#plotting positions
fig = plt.figure(1, figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.view_init(0, 45) #change viewing angle to cross-sectional
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)
plot = ax.scatter(x_d, y_d, z_d, s=5)
plot2 = ax.scatter(x_h, y_h, z_h, color = 'green', s=5)
plot3 = ax.scatter(x_b, y_b, z_b, color = 'red', s=5)

#veldisp plots

plt.figure(2, figsize=(20,20))
plt.scatter(r_h, veldisp_r_h, label = 'Computational')
plt.xlabel('r  [kpc]', fontsize = 20)
plt.ylabel('$ \\bar{v_r ^2} \, \, \, \\left [km^2 s^{-2} \\right ] $', fontsize = 20)
plt.title('Radial Velocity Dispersion of the Halo', fontsize=25)
def veldisp_radial_halo_model(r):
    integral, error = integrate.quad(hernquist_dispersion_equation_halo, r, np.infty)
    return integral / halo_density(r)
rhalo = np.linspace(min(r_h), max(r_h), 10000).tolist()
veldisp_r_h_model=[]
for r in rhalo:
    veldisp_r_h_model.append(veldisp_radial_halo_model(r))
plt.plot(rhalo, veldisp_r_h_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)



plt.figure(3,figsize=(20,20))


#density plots
density_halo = []
density_halo_model = []
radius_median_halo = []

r_h.sort()
index = 0

for r in r_h:
    r_1 = r
    r_2 = r_h[index + 1000]
    #r_2 = r + dr
    r_m = median(r_h[index:index+1000])
    
    #r_m = r + 0.5   
    radius_median_halo.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_halo.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    density_halo_model.append(halo_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_h)-1000:
        break

plt.title('Density of the Halo', fontsize=25)
plt.scatter(radius_median_halo, density_halo, label = 'Computational')
plt.plot(radius_median_halo, density_halo_model, color = 'red', label = 'Analytical')
plt.legend(fontsize=20)

## radial velocity disp of bulge

plt.figure(4, figsize=(20,20))
plt.scatter(r_b, veldisp_r_b, label = 'Computational')
plt.xlabel('r  [kpc]', fontsize = 20)
plt.ylabel('$ \\bar{v_r ^2} \, \, \, \\left [km^2 s^{-2} \\right ] $', fontsize = 20)
plt.title('Radial Velocity Dispersion of the Bulge', fontsize=25)
def veldisp_radial_bulge_model(r):
    integral, error = integrate.quad(hernquist_dispersion_equation_bulge, r, np.infty)
    return integral / bulge_density(r)
rbulge = np.linspace(min(r_b), max(r_b), 10000).tolist()
veldisp_r_b_model=[]
for r in rbulge:
    veldisp_r_b_model.append(veldisp_radial_bulge_model(r))
plt.plot(rbulge, veldisp_r_b_model, label = 'Analytical', color = 'red')
plt.legend(fontsize=20)


##############################
# bulge density plots

plt.figure(5, figsize=(20,20))
density_bulge = []
density_bulge_model = []
radius_median_bulge = []

r_b.sort()
index = 0

for r in r_b:
    r_1 = r
    r_2 = r_b[index + 1000]
    #r_2 = r + dr
    r_m = median(r_b[index:index+1000])
    
    #r_m = r + 0.5   
    radius_median_bulge.append(r_m)
    
    #N = len( np.where((radius >=r_1) & (radius <= r_2)) )
        
    density_bulge.append( 1000 / (4/3 * np.pi * ((r_2)**3-(r_1)**3)))
    density_bulge_model.append(bulge_density(r_m))
    index+=1
    
    #breaking off the shell iteration when it reaches 5000th the last element
    if index == len(r_b)-1000:
        break

plt.title('Density of the Bulge', fontsize=25)
plt.scatter(radius_median_bulge, density_bulge, label = 'Computational')
plt.plot(radius_median_bulge, density_bulge_model, color = 'red', label = 'Analytical')
plt.legend(fontsize=20)

############## disk density plots

density_disk = []
density_disk_model = []
radius_mean_disk = []

z_cuts = []
#radius_master_list = []
#indices_master_list = []
#z

medianradius_list = []
density_list = []
density_model_list = []
zdisk_list = []
Rdisk_list = []
indices_list = []


##measuring disk density along cuts of z
z_cuts = [-1.5, -0.6, -0.5, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5]
for i in range(0,len(z_cuts)-2):
    zdisk_list.append([])
    indices_list.append([])
    Rdisk_list.append([])
    density_list.append([])
    density_model_list.append([])
    medianradius_list.append([])
    

for j in range(0,len(z_d)):
    for i in range(0,len(z_cuts)-2):
        if z_d[j] >= z_cuts[i] and z_d[j] <= z_cuts[i+1]:
            indices_list[i].append(j)
            zdisk_list[i].append(z_d[j])
            Rdisk_list[i].append(R[j])

            
for i in range(0,2):
    Rdisk_list[i].sort()
    index = 0
    for r in Rdisk_list[i]:
        
        r_1 = Rdisk_list[i][index]
        r_2 = Rdisk_list[i][index + 10]
        r_m = median(Rdisk_list[i][index:index+10])
        z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
        dz = abs(z_cuts[i]-z_cuts[i+1])
        
    
        density = 10 / (np.pi  * dz * ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list[i].append( density )
        medianradius_list[i].append(r_m)
        density_model_list[i].append(disk_density(r_m, z_m))
        index+=1
    
        if index == len(Rdisk_list[i])-10:
            index = 0
            break
        
for i in range(3,13):
    Rdisk_list[i].sort()
    index = 0     
    for r in Rdisk_list[i]:
        
        r_1 = Rdisk_list[i][index]
        r_2 = Rdisk_list[i][index + 1000]
        r_m = median(Rdisk_list[i][index:index+1000])
        z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
        dz = abs(z_cuts[i]-z_cuts[i+1])
        
        density = 1000 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list[i].append( density )
        medianradius_list[i].append(r_m)
        density_model_list[i].append(disk_density(r_m, z_m))
        index+=1
    
        if index == len(Rdisk_list[i])-1000:
            index = 0
            break

for i in range(14,16):
    Rdisk_list[i].sort()
    index = 0
    for r in Rdisk_list[i]:
        
        r_1 = Rdisk_list[i][index]
        r_2 = Rdisk_list[i][index + 10]
        r_m = median(Rdisk_list[i][index:index+10])
        z_m = 0.5*(z_cuts[i] + z_cuts[i+1])
        dz = abs(z_cuts[i]-z_cuts[i+1])
        
        density = 10 / (np.pi *dz* ((r_2)**2-(r_1)**2))
        #if abs(density) < 1000:
        density_list[i].append( density )
        medianradius_list[i].append(r_m)
        
        density_model_list[i].append(disk_density(r_m, z_m))
        index+=1

        if index == len(Rdisk_list[i])-10:
            index = 0
            break
        
        


for i in range(0,len(z_cuts)-2):
    plt.figure(i+6, figsize=(20,20))
    plt.scatter(medianradius_list[i], density_list[i], label = 'Computational')
    plt.plot(medianradius_list[i], density_model_list[i], color = 'red', label = 'Analytical')
    plt.legend(fontsize = 20)

plt.figure(26, figsize = (20,20))
plt.scatter(R, anisotropy_d)
plt.title('Anisotropy of Disk')
