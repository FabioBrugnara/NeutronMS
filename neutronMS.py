# IMPORT
import numpy as np
np.seterr(invalid='ignore', divide='ignore')

from matplotlib import pyplot as plt
from matplotlib import colors

from numpy.random import rand
from scipy.stats import truncexpon

import pandas as pd


#######################################################################################################
########################################## USEFULL FUNCTIONS ##########################################
#######################################################################################################

### k-E relations ###
k2E = lambda k:81.8*(k/(2*np.pi))**2
E2k = lambda E:2*np.pi*np.sqrt(E/81.8)

### Q-theta relations ###
theta2Q = lambda omega, ki, theta: ki * np.sqrt(2 - omega/k2E(ki) - 2*np.sqrt(1 - omega/k2E(ki))*np.cos(theta))
Q2theta = lambda omega, ki, Q: np.arccos((2-Q**2/ki**2-omega/k2E(ki)) / (2*np.sqrt(1-omega/k2E(ki))))

##### dQ-dtheta relation #####
def dqdtheta(omega, ki, theta):
    a = 2-omega/k2E(ki)
    b = 2*np.sqrt(1-omega/k2E(ki))
    return ki**2/2 * b*np.sin(theta) / np.sqrt(a-b*np.cos(theta))

def dthetadq(omega, ki, theta):
    a = 2-omega/k2E(ki)
    b = 2*np.sqrt(1-omega/k2E(ki))
    return (ki**2/2 * b*np.sin(theta) / np.sqrt(a-b*np.cos(theta)))**-1


########################################################################################################
####################################### LINE-SOLID INTERCEPTIONS #######################################
########################################################################################################

############### Line parametrization ###############
def line_param(t, p, v):
    v = v/np.linalg.norm(v, axis=1)[:,None]
    return np.array([p[:,0] + v[:,0]*t, p[:,1] + v[:,1]*t, p[:,2] + v[:,2]*t]).T


############### CYLINDER ###############

### Line-cylinder intersection with p0 outside ###
# The interception is guaranteed by the given p and v, and happens on the cylinder and not on the plain boundaries. That is, whatch out for random p, v
def CylLine_inter_fromout(geom, p, v):
    # the incoming neutrons are aligned along x, therefor can hit only the vertical walls!
    r, h = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]
  
    a = v[:,0]**2 + v[:,1]**2
    b = 2*(p[:,0]*v[:,0] + p[:,1]*v[:,1])
    c = p[:,0]**2 + p[:,1]**2 - r**2
    delta = b**2 - 4*a*c

    t1c, t2c = (-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a)

    return np.sort((t1c, t2c), axis=0).T

### Line-cylinder intersection with p0 inside ###
# Only one foreword intercept is guaranteed, p is for shure inside the body
def CylLine_inter_fromin(geom, p, v):
    r, h = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]

    ### cylinder
    a = v[:,0]**2 + v[:,1]**2
    b = 2*(p[:,0]*v[:,0] + p[:,1]*v[:,1])
    c = p[:,0]**2 + p[:,1]**2 - r**2
    delta = b**2 - 4*a*c

    tc = np.max([(-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a)], axis=0) # take the higher t, that is the only one positive solution!

    ### planes
    tz = np.max([(h/2 - p[:,2])/v[:,2], (-h/2 - p[:,2])/v[:,2]], axis=0) # take the higher t, that is the only one positive solution!
    
    return np.sort((tc, tz), axis=0)[0].T


############### CUBOID ###############

### Line-cuboid intersection with p0 outside ###
def CubLine_inter_fromout(geom, p, v):
    # the incoming neutrons are aligned along x, therefor can hit only the x walls!
    dx, dy, dz = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]
  
    # intercepts
    t1x, t2x = (-dx/2 - p[:,0])/v[:,0], (+dx/2 - p[:,0])/v[:,0]

    return np.sort([t1x, t2x])[:2].T

### Line-cuboid intersection with p0 inside ###
def CubLine_inter_fromin(geom, p, v):
    dx, dy, dz = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]

    # intercepts
    tx = np.max([(-dx/2 - p[:,0])/v[:,0], (+dx/2 - p[:,0])/v[:,0]], axis=0)
    ty = np.max([(-dy/2 - p[:,1])/v[:,1], (+dy/2 - p[:,1])/v[:,1]], axis=0)
    tz = np.max([(-dz/2 - p[:,2])/v[:,2], (+dz/2 - p[:,2])/v[:,2]], axis=0)
    
    return np.min((tx, ty, tz), axis=0).T


############### ANULAR ###############

### Line-anular intersection with p0 outside ###
# The interception is guaranteed by the given p and v, and happens on the cylinder and not on the plain boundaries. That is, whatch out for random p, v
def AnuLine_inter_fromout(geom, p, v):
    r1, r2, h = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]

    # the incoming neutrons are aligned along x, therefor can hit only the vertical walls!
    a = v[:,0]**2 + v[:,1]**2
    b = 2*(p[:,0]*v[:,0] + p[:,1]*v[:,1])

    c2 = p[:,0]**2 + p[:,1]**2 - r2**2
    delta2 = b**2 - 4*a*c2

    c1 = p[:,0]**2 + p[:,1]**2 - r1**2
    delta1 = b**2 - 4*a*c1

    t1c, t2c, t3c, t4c = (-b + np.sqrt(delta2))/(2*a), (-b - np.sqrt(delta2))/(2*a), (-b + np.sqrt(delta1))/(2*a), (-b - np.sqrt(delta1))/(2*a)
    # if no 4 intercepts, wh have nans!
    return np.sort((t1c, t2c, t3c, t4c), axis=0).T

### Line-anular intersection with p0 inside ###
# Only one foreword intercept is guaranteed, p is for shure inside the body
def AnuLine_inter_fromin(geom, p, v):
    r1, r2, h = geom[1:]
    v = v/np.linalg.norm(v, axis=1)[:,None]

    ### cylinders
    a = v[:,0]**2 + v[:,1]**2
    b = 2*(p[:,0]*v[:,0] + p[:,1]*v[:,1])

    c2= p[:,0]**2 + p[:,1]**2 - r2**2
    delta2 = b**2 - 4*a*c2

    c1 = p[:,0]**2 + p[:,1]**2 - r1**2
    delta1 = b**2 - 4*a*c1

    tc1 = np.max([(-b + np.sqrt(delta2))/(2*a), (-b - np.sqrt(delta2))/(2*a)], axis=0) # the external cylinder can have only one intercept!
    pc1 = line_param(tc1, p, v)
    tc1[(pc1[:,2]>h/2)|(pc1[:,2]<-h/2)] = np.nan  #check if it have!
    
    tc2, tc3 = (-b + np.sqrt(delta1))/(2*a), (-b - np.sqrt(delta1))/(2*a) # this can have 0, 1 or 2 intercept
    tc2[tc2<0] = np.nan # should be foreword
    tc3[tc3<0] = np.nan

    pc2 = line_param(tc2, p, v)
    tc2[(pc2[:,2]>h/2)|(pc2[:,2]<-h/2)] = np.nan # and should hapend in +-h/2
    pc3 = line_param(tc3, p, v)
    tc3[(pc3[:,2]>h/2)|(pc3[:,2]<-h/2)] = np.nan

    ### planes
    tz = np.max([(h/2 - p[:,2])/v[:,2], (-h/2 - p[:,2])/v[:,2]], axis=0)
    pz = line_param(tz, p, v)
    tz[((pz[:,0]**2 + pz[:,1]**2) < r1**2)|((pz[:,0]**2 + pz[:,1]**2) > r2**2)] = np.nan
    
    return  np.sort((tc1, tc2, tc3, tz), axis=0)[:3].T # in total, a maximum three intercepts|


########################################################################################################
########################################## NEUTRON GENERATION ##########################################
########################################################################################################

def ngen4Cyl(geom, x_init, N):
    r, h = geom[1:]
    return np.array([[-x_init]*N, rand(N) * 2*r - r, rand(N) * h - h/2]).T

def ngen4Cub(geom, x_init, N):
    dx, dy, dz = geom[1:]
    return np.array([[-x_init]*N, rand(N) * dy - dy/2, rand(N) * dz - dz/2]).T

def ngen4Anu(geom, x_init, N):
    r1, r2, h = geom[1:]
    return np.array([[-x_init]*N, rand(N) * 2*r2 - r2, rand(N) * h - h/2]).T


########################################################################################################
############################################## GEOMETRIES ##############################################
########################################################################################################

# example
# geom = ('cylinder', r, h)
# geom = ('cuboid', dx, dy, dz)
# geom = ('anular', r1, r2, h)

### "detector" fake apertures
dtheta = 0.01
dphi = 0.01
domega = .1

### x for neutron initial generation ###
x_init = 100 #cm


########################################################################################################
##################################### GEN SCATTERING GEOMETRIES ########################################
########################################################################################################

def GEN_constQgeom(type, kx, Q, omega_vec):
    if type=='inverse':
        kf = kx
        s = omega_vec.shape[0]
        Ei_vec = omega_vec + k2E(kf)
        theta_vec = Q2theta(omega_vec, E2k(Ei_vec), Q)

        return pd.DataFrame({ 'omega': omega_vec, 'Q': np.ones(s)*Q, 'ki': E2k(Ei_vec), 'kf': np.ones(s)*kf, 'Ei': E2k(Ei_vec), 'Ef': np.ones(s)*k2E(kf), 'theta': theta_vec})

    
    elif type=='direct':
        ki = kx
        s = omega_vec.shape[0]
        Ef_vec = k2E(ki) - omega_vec
        theta_vec = Q2theta(omega_vec, ki, Q)

        return pd.DataFrame({ 'omega': omega_vec, 'Q': np.ones(s)*Q, 'ki': np.ones(s)*ki, 'kf': E2k(Ef_vec), 'Ei': np.ones(s)*k2E(ki), 'Ef': E2k(Ef_vec), 'theta': theta_vec})
    

########################################################################################################
############################################ THE SIMULATION ############################################
########################################################################################################

class MS_sim:
    def __init__(self, geom, mus, S_files, ki:float, kf: float, theta: float):
        self.geom = geom
        self.ki = ki
        self.kf = kf
        self.theta = theta

        #####################
        ###### Geometry #####
        #####################
        self.mu_s = mus[0]
        self.mu_abs = mus[1]

        if geom[0]=='cuboid':
            self.geom_type = 'convex'
            self.ngen = lambda N: ngen4Cub(geom, x_init, N)
            self.inter_fromout, self.inter_fromin = lambda p, v: CubLine_inter_fromout(geom, p, v), lambda p, v: CubLine_inter_fromin (geom, p, v)
        elif geom[0]=='cylinder':
            self.geom_type = 'convex'
            self.ngen = lambda N: ngen4Cyl(geom, x_init, N)
            self.inter_fromout, self.inter_fromin = lambda p, v: CylLine_inter_fromout(geom, p, v), lambda p, v: CylLine_inter_fromin (geom, p, v)

        elif geom[0]=='anular':
            self.geom_type = 'concave'
            self.ngen = lambda N: ngen4Anu(geom, x_init, N)
            self.inter_fromout, self.inter_fromin = lambda p, v: AnuLine_inter_fromout(geom, p, v), lambda p, v: AnuLine_inter_fromin (geom, p, v)

        ####################################
        ##### Scattering configuration #####
        ####################################
        self.Ei = k2E(ki) #meV
        self.Ef = k2E(kf)
        self.omega = self.Ei - self.Ef
        self.Q = theta2Q(self.omega, ki, theta)

        ##################
        ##### S(Q,w) #####
        ##################

        # LOAD S(Q,E)
        self.S_mesh = np.load(S_files[0])
        self.Q_mesh = np.load(S_files[1])
        self.omega_mesh = np.load(S_files[2])

        # Usefull vectors
        self.omega_vec = self.omega_mesh[0,:]
        self.Q_vec = self.Q_mesh[:,0]
        self.dQ_vec = self.Q_vec[1]-self.Q_vec[0]
        self.domega_vec = self.omega_vec[1]-self.omega_vec[0]

        # Useful object for plots
        self.S_imshow_extent = [self.omega_vec.min(),self.omega_vec.max(),self.Q_vec.max(),self.Q_vec.min()]

        ###############################################
        ##### Dynamic range for direct scattering #####
        ###############################################
        self.Q_lowlim = lambda ki: ki * np.sqrt(2 - self.omega_vec/k2E(ki) - 2*np.sqrt(1 - self.omega_vec/k2E(ki)))
        self.Q_uplim  = lambda ki: ki * np.sqrt(2 - self.omega_vec/k2E(ki) + 2*np.sqrt(1 - self.omega_vec/k2E(ki)))

        ####################################
        ##### Cut of the S(Q,w) for Ei #####
        ####################################
        self.S_mesh_cutEi = ((self.Q_mesh>self.Q_lowlim(ki)) & (self.Q_mesh<self.Q_uplim(ki))) * self.S_mesh
        self.S_mesh_cutEi /= self.S_mesh_cutEi.sum() * self.dQ_vec*self.domega_vec

        ###########################
        ##### Lamber-Beer law #####
        ###########################

        # Trasmission
        self.T_s = lambda d: np.exp(-self.mu_s*d)
        self.T_abs = lambda d: np.exp(-self.mu_abs*d)

        # Random scattering extraction from Lamber-Beer law
        self.rand_LB = lambda b: truncexpon.rvs(b = b*self.mu_s)/self.mu_s



    def run(self, N: int=100000, N_bunch: int=100):
        ##########################
        ##### THE SIMULATION #####
        ##########################

        ##### RANDOMLY GENERATE N NEUTRONS #####
        # from uniform beam at x=-x_init, random z = [-h/2, h/2] and y[-r, r]
        p0 = self.ngen(N)
        k0 = np.array([[self.ki, 0, 0]]*N)
        w0 = np.array([1]*N)

        ##### EXTRACT THE 1ST SCATTERING POSITION #####

        # intercept calculation
        ts = self.inter_fromout(p0, k0)
        ts = np.nan_to_num(ts)
        if self.geom_type=='convex':
            ts = np.append(ts, np.zeros((N,2)), axis=1)
        if self.geom_type=='concave':
            pass

        d1 = (ts[:,1] - ts[:,0]) + (ts[:,3] - ts[:,2])

        # weigth update
        w1 = w0*(1-self.T_s(d1))*self.T_abs(d1)

        # dtp2 extraction (one for all)
        dtp2 = self.rand_LB(d1)

        # generate void vector
        void = np.where(dtp2<=(ts[:, 1]-ts[:, 0]), 0, ts[:,2] - ts[:,1])

        p1 = line_param(ts[:,0] + dtp2 + void, p0, k0)

        
        ####################################
        ###### SINGLE SCATTERING PATH ######
        ####################################

        # Find k1, omega1, Q1 imposing collimator angle
        k1s = self.kf * np.array([np.cos(self.theta), np.sin(self.theta), 0]) * np.ones((N, 3))
        Q1s = np.linalg.norm(k0 - k1s, axis=1)
        omega1s = (self.Ei - self.Ef) * np.ones(N)
        theta1s = self.theta

        # Get the scattering probavilities from S_mesh_cutEi (w1s)
        Q_idx = np.searchsorted(self.Q_vec, Q1s)
        omega_idx = np.searchsorted(self.omega_vec, omega1s)

        w1s = w1 * self.S_mesh_cutEi[Q_idx, omega_idx] * dqdtheta(omega1s, self.ki, theta1s) * dtheta * dphi * domega

        # intercept calculation
        ts = self.inter_fromin(p1, k1s)
        ts = np.nan_to_num(ts)
        if self.geom_type=='convex':
            ts = np.column_stack([ts, np.zeros((N,2))])
        if self.geom_type=='concave':
            pass

        d2s = ts[:,0] + (ts[:,2] - ts[:,1])

        # weigth update
        wfs = w1s*self.T_s(d2s)*self.T_abs(d2s)


        ####################################
        ##### MULTIPLE SCATTERING PATH #####
        ####################################

        ### EXTRACTION FROM S(q,W) ###

        # dQdtheta on the mesh
        dqdtheta_pdf = np.nan_to_num(dqdtheta(self.omega_mesh, self.ki, Q2theta(self.omega_mesh, self.ki, self.Q_mesh)))     

        # Probaility mesh for random choiches
        PDF = self.S_mesh * dqdtheta_pdf # * dtheta * domega
        PDF /= PDF.sum() # * dtheta * domega

        # Random extractions
        is_mesh = np.random.choice(np.arange(PDF.shape[0]*PDF.shape[1]), p=PDF.reshape(-1), size=N)

        # Extracted scattering variables
        Q1m = self.Q_mesh.reshape(-1)[is_mesh]
        omega1m = self.omega_mesh.reshape(-1)[is_mesh]
        theta1m = Q2theta(omega1m, self.ki, Q1m)
        E1m = self.Ei - omega1m
        w1m = w1

        ### NEW NEUTRON MOMENTA ###
        # Random angle on the scattering cone
        phi1m = rand(N)*2*np.pi

        # Neutrom momenta after scattering
        k1m = (np.array((np.cos(theta1m), np.sin(theta1m)*np.cos(phi1m), np.sin(theta1m)*np.sin(phi1m))) * E2k(E1m)).T
        k1m_mod = E2k(E1m)

        # intercept calculation
        ts = self.inter_fromin(p1, k1m)
        ts = np.nan_to_num(ts)
        if self.geom_type=='convex':
            ts = np.column_stack([ts, np.zeros((N,2))])
        if self.geom_type=='concave':
            pass

        d2m = ts[:,0] + (ts[:,2] - ts[:,1])

        # weigth update
        w2 = w1m*(1-self.T_s(d2m))*self.T_abs(d2m)

        # dtp2 extraction (one for all)
        dtp2 = self.rand_LB(d2m)

        # generate void vector
        void = np.where(dtp2<=ts[:, 0], 0,ts[:,1] - ts[:,0])

        p2 = line_param(dtp2 + void, p1, k1m)

        ### Find k2m, omega2m, Q2m imposing collimator angle ###
        k2m = self.kf * np.array([np.cos(self.theta), np.sin(self.theta), 0]) * np.ones((N, 3))
        omega2m = (E1m - self.Ef) * np.ones(N)
        Q2m = np.linalg.norm(k1m - k2m, axis=1)

        theta2m = Q2theta(omega2m, E2k(E1m), Q2m)

        ### Mask of impossible events
        mask2m = ~((Q2m>self.Q_vec.max()) | (Q2m<self.Q_vec.min()) | (omega2m>self.omega_vec.max()) | (omega2m<self.omega_vec.min()) | np.isnan(theta2m))

        ### SCATTERING PROBABILITIES FROM THE PDF $$$

        # Bunching the initial energies E1m
        E_min = E1m[mask2m].min()
        E_max = E1m[mask2m].max()
        bunch = np.linspace(E_min, E_max, N_bunch)[1:]
        dbunch = (bunch[1]-bunch[0])

        # find the bunch indexes
        bunch_idx = np.searchsorted(bunch, E1m, side='left')
        bunch_idx[~mask2m] = -1 # putting impossible events with idx=-1, i.e. in the next part P stay 0

        # indexing Q2m and omega2m on the mesh
        Q_idx = np.searchsorted(self.Q_vec, Q2m)
        omega_idx = np.searchsorted(self.omega_vec, omega2m)

        # calculating probabilities at bunches
        P = np.zeros(N)

        for b in range(N_bunch-1):
            S_mesh_cut = ((self.Q_mesh>self.Q_lowlim(E2k(bunch[b]-dbunch/2))) & (self.Q_mesh<self.Q_uplim(E2k(bunch[b]-dbunch/2)))) * self.S_mesh
            S_mesh_cut /= S_mesh_cut.sum() * self.dQ_vec * self.domega_vec

            P[bunch_idx==b] = S_mesh_cut[Q_idx[bunch_idx==b], omega_idx[bunch_idx==b]] * dqdtheta(omega2m[bunch_idx==b], k1m_mod[bunch_idx==b], theta2m[bunch_idx==b]) * dtheta * dphi * domega

        # updating the weights
        w2m = w2 * P

        # intercept calculation
        ts = self.inter_fromin(p1, k2m)
        ts = np.nan_to_num(ts)
        if self.geom_type=='convex':
            ts = np.column_stack([ts, np.zeros((N,2))])
        if self.geom_type=='concave':
            pass

        d3m = ts[:,0] + (ts[:,2] - ts[:,1])

        # weigth update
        wfm = w2m*self.T_s(d3m)*self.T_abs(d3m)

        

        ####################################
        ########### FINAL OUTPUT ###########
        ####################################
        self.p0 = p0
        self.w0 = w0
        self.k0 = k0

        self.d1 = d1
        self.w1 = w1
        self.p1 = p1

        # single
        self.k1s = k1s
        self.w1s = w1s

        self.d2s = d2s
        self.wfs = wfs

        self.Is = wfs.mean()

        # multiple
        self.k1m = k1m
        self.w1m = w1m
        self.theta1m = theta1m
        self.omega1m = omega1m

        self.d2m = d2m
        self.w2 = w2

        self.p2 = p2
        self.k2m = k2m
        self.w2m = w2m
        self.theta2m = np.arccos((k2m*k1m).sum(axis=1)/(np.linalg.norm(k2m, axis=1)*np.linalg.norm(k1m, axis=1)))
        self.omega2m = omega2m

        self.d3m = d3m
        self.wfm = wfm

        self.Im = wfm.mean()






