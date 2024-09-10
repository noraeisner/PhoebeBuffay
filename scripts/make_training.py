import numpy as np
import pandas as pd
import phoebe as phb
import astropy.units as au
import astropy.constants as ac
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

def stringify_float(f):
    return "{:.5f}".format(f)

def m_to_t(m,r):
    top = 1.4 * m**3.5 ## mass to lum
    bottom = (4*3.141592653*r**2 * ac.sigma_sb.to('Lsun/Rsun/Rsun/K/K/K/K')).value
    return (top/bottom)**0.25


def calculate_masses(q, sma, period):
    # Given the mass ratio (q), the semi-major axis (sma) and the period (in days),
    # calculate the masses of the primary and secondary stars (in solar masses)
    G = 6.67430e-11  # gravitational constant in m^3 kg^−1 s^−2
    M_sun = 1.989e30  # mass of the Sun in kg

    # Convert sma from Rsun to meters
    sma_m = sma * 6.957e8  # convert sma from Rsun to meters

    # Calculate the total mass of the binary system
    M_total = (4 * np.pi**2 * sma_m**3) / (G * (period*86400.)**2)

    # Calculate the mass of the primary star
    M1 = M_total / (1 + q)

    # Calculate the mass of the secondary star
    M2 = M_total - M1

    # Convert masses from kg to solar masses
    M1_sun = M1 / M_sun
    M2_sun = M2 / M_sun

    return M1_sun, M2_sun




def calculate_sma(M1,M2,period):
    # Given the masses (in solar masses) and the period (in days), 
    # calculate the semi-major axis (in solar radii

    G = 6.67430e-11  # gravitational constant in m^3 kg^−1 s^−2
    M_sun = 1.989e30  # mass of the Sun in kg

    M_total = (M1+M2)*M_sun

    sma = ( ( ( period*86400.)**2 * G * M_total) / (4.*np.pi**2) )**(1./3.)

    return sma / 6.957e8


def sort_on_x(x, y):
    """
    Sorts the x and y arrays based on the x values
    """
    # Create a list of tuples
    zipped = list(zip(x, y))

    # Sort the list of tuples based on the x values
    zipped.sort(key=lambda x: x[0])

    # Unzip the sorted list of tuples
    x, y = zip(*zipped)

    return np.array(x), np.array(y)


def r_from_logg(m, logg):
    # Given the mass (in solar masses) and the surface gravity (in cgs),
    # calculate the radius (in solar radii)

    G = 6.67430e-8  # gravitational constant in cm^3 g^−1 s^−2
    
    r_cm = 6.957e10  # convert radius from solar radii to centimeters
    m_g = 1.989e33  # convert mass from solar masses to grams

    g = 10**logg
    r = np.sqrt( (G*m*m_g) / g )

    return r / r_cm


def generate_lightcurve(vector, n_phase):
    """
    Generates a light curve given the parameters in the vector
    Interpolates the model to have n_phase points distributed between
    0 and 1.
    """

    # Instantiate default binary model
    b = phb.Bundle.default_binary()

    # Set the parameters
    b.set_value('period@binary', vector['period'])
    b.set_value('t0_supconj@binary', vector['t0_frac']*vector['period'])
    b.set_value('sma@binary', vector['sma'])
    b.set_value('incl@binary', vector['incl'])
    b.set_value('q@binary', vector['q'])
    b.set_value('ecc@binary', vector['ecc'])
    b.set_value('per0@binary', vector['per0'])

    # Primary
    b.set_value('teff@primary', vector['teff1'])
    b.set_value('requiv@primary', vector['reqv1'])
    b.set_value('gravb_bol@primary', vector['grb1'])
    b.set_value('irrad_frac_refl_bol@primary', vector['alb1'])

    # Secondary
    b.set_value('teff@secondary', vector['teff2'])
    b.set_value('requiv@secondary', vector['reqv2'])
    b.set_value('gravb_bol@secondary', vector['grb2'])
    b.set_value('irrad_frac_refl_bol@secondary', vector['alb2'])

    # Declare data set
    times = np.linspace(0., 1.1 * b.get_value('period@binary')-0.001, 300)
    b.add_dataset('lc', times=times, dataset='lc01')
    # b.set_value('l3_mode', 'fraction')
    # b.set_value('l3_frac', vector['l3'])
    b.set_value('passband@lc01', 'TESS:T')
    b.set_value('ld_mode@primary@lc01', 'lookup')
    b.set_value('ld_mode@secondary@lc01', 'lookup')

    mass1 = b.get_value('mass@primary@component')
    radius1 = b.get_value('requiv@primary@component')
    logg1 = b.get_value('logg@primary@component')

    mass2 = b.get_value('mass@secondary@component')
    radius2 = b.get_value('requiv@secondary@component')
    logg2 = b.get_value('logg@secondary@component')

    print('Period = {} \t A = {} \t incl = {}'.format(vector['period'], vector['sma'], vector['incl']))
    print('Mass 1 = {} \t Radius 1 = {} \t logg1 = {} \t teff1 = {}'.format(mass1, radius1, logg1, teff1_dist[i]))
    print('Mass 2 = {} \t Radius 2 = {} \t logg2 = {} \t teff2 = {}'.format(mass2, radius2, logg2, teff2_dist[i]))
    print('\n\n')


    # Check if the parameters don't produce a system with RLOF and are within the range of the lookup tables
    check = ( (b.get_value('requiv@primary') < b.get_value('requiv_max@primary@component')) & 
              (b.get_value('requiv@secondary') < b.get_value('requiv_max@secondary@component')) &
              (b.get_value('logg@primary@component') > 3.) & (b.get_value('logg@primary@component') < 5.) &
              (b.get_value('logg@secondary@component') > 3.) & (b.get_value('logg@secondary@component') < 5.)
            )

    if check:

        # Run the model

        b.add_compute('ellc')
        # b.set_value('atm@primary@compute', 'blackbody')
        # b.set_value('atm@secondary@compute', 'blackbody')
        try:
            b.run_compute( kind='ellc', irrad_method='none')
            model = b.get_value('fluxes@model')
    
            ph = ( (times-b.get_value('t0_supconj@binary')) / b.get_value('period@binary')) % 1
            ph_, model_ = sort_on_x(ph, model)
            ph_ = np.hstack([ph_, ph_+1.])
            model_ = np.hstack([model_, model_])

            if not np.all(ph_[1:]-ph_[:-1] > 0.):
                print('Phase problem')
            model = PchipInterpolator(ph_, model_, extrapolate=False)(np.linspace(0., 2., 2 * n_phase))

            model[0] = model[n_phase]
            model[1] = model[n_phase+1]
            model = model[:n_phase]
            # Rescaling to the median has two purposes:
            # 1. The model is normalized to the median of the model
            # 2. The effect of third light is accounted for, reducing the
            #    apparent depth of the eclipses when third light is present
            # We're taking care of the fractional third light here
            # in a slightly hacky, but essentially equivalent way
            l3 = np.nanmedian(model) * vector['l3']
            model = model + l3
            model /= np.nanmedian(model)

        except:
            print('Model failed')
            model = np.zeros(n_phase)

    else:
        print('Check failed')
        model = np.zeros(n_phase)


    # print(model[0], model[1], model[2], model[-2], model[-1])
    return model, mass1, radius1, logg1, mass2, radius2, logg2



if __name__ == '__main__':


    n_phase = 500

    ndist = 2000

    # Generate random parameters
    period_dist = np.random.uniform(0.1, 25., ndist)
    t0_frac_dist = np.random.uniform(-1., 1., ndist)
    q_dist = np.random.uniform(0.1, 1.5, ndist)

    # For the sake of simplicity, we will assume that the primary star is the more massive star
    mass1_dist = np.random.uniform(1.,1.4,ndist)
    mass2_dist = mass1_dist * q_dist

    # Because there are vastly more combinations of mass and radius that will lead to 
    # logg values outside the range of our LD table, we will generate random logg values
    # and calculate the radii from them
    logg1_dist = np.random.uniform(3.,5.,ndist)
    logg2_dist = np.random.uniform(3.,5.,ndist)

    reqv1_dist = np.array( [r_from_logg(m1, logg1) for m1, logg1 in zip(mass1_dist, logg1_dist) ] ) ## solar radii
    reqv2_dist = np.array( [r_from_logg(m2, logg2) for m2, logg2 in zip(mass2_dist, logg2_dist) ] )

    # Similarly, we're going to calculate the semi-major axis from the masses and period
    sma_dist = np.array([ calculate_sma(m1, m2, period) for m1, m2, period in zip(mass1_dist, mass2_dist, period_dist)])


    # General, non-circular case
    # ecc_dist = np.random.uniform(0., 0.5, ndist)
    # per0_dist = np.random.uniform(0., 360, ndist)

    # Circular case
    ecc_dist = np.zeros(ndist)
    per0_dist = np.zeros(ndist)

    teff2_dist = np.random.uniform(3501., 15000, ndist)
    teff1_dist = np.random.uniform(3501., 15000, ndist)

    incl_dist = np.random.uniform(70.,90,ndist)
    l3_dist = np.random.uniform(0.,0.99,ndist)
    #l3_dist = np.zeros(ndist)

    grb1_dist = np.random.uniform(0.,1.,ndist)
    grb2_dist = np.random.uniform(0.,1.,ndist)

    alb1_dist = np.random.uniform(0.,1.,ndist)
    alb2_dist = np.random.uniform(0.,1.,ndist)


    fmt_sim = '{:0>5d}'
    fmt_x = '{:0>4d}'

    x_num = np.array([fmt_x.format(i) for i in range(n_phase)])

    theta_names = [ 'simulation', 'period', 't0', 'ecc', 'per0', 'q', 'sma','incl', 
                    'reqv1', 'reqv2','teff1', 'teff2', 'grb1','grb2', 'alb1', 'alb2', 'l3']

    out_names = np.hstack( [ ['simulation'], x_num] )

    psi_names = [ 'simulation', 'mass1', 'radius1', 'logg1', 'mass2', 'radius2', 'logg2']

    try:
        print('Training data already exists')
        df = pd.read_csv('./training/psi.txt')
        n = df.shape[0]
 
    except:
        n = 0
        with open('./training/theta.txt','a') as ft:
            ft.write(','.join(theta_names)+'\n')

        with open('./training/simulations.txt', 'a') as f:
            f.write(','.join(out_names)+'\n')

        with open('./training/psi.txt', 'a') as f:
            f.write(','.join(psi_names)+'\n')

    # build a list of simulation numbers
    sim_num = np.array([fmt_sim.format(i+n) for i in range(ndist)])
 

    for j in range(n, n + ndist):
        i = j - n 
        vector = {'period': period_dist[i], 't0_frac': t0_frac_dist[i], 'ecc': ecc_dist[i], 'per0': per0_dist[i], 
                  'q': q_dist[i], 'sma': sma_dist[i], 'incl': incl_dist[i],
                  'reqv1': reqv1_dist[i], 'reqv2': reqv2_dist[i], 
                  'teff1': teff1_dist[i], 'teff2': teff2_dist[i], 
                  'grb1': grb1_dist[i], 'grb2': grb2_dist[i], 'alb1': alb1_dist[i], 'alb2': alb2_dist[i],
                  'l3': l3_dist[i], 
                 }

        y, mass1, radius1, logg1, mass2, radius2, logg2  = generate_lightcurve(vector, n_phase)


        # Save the data
        # This is a very inefficient way to save the data, but it makes sure that if we have to stop the
        # program for some reason, we don't lose all the data, and can continue from where we left off
        with open('./training/theta.txt','a') as ft:
            ft.write(','.join([sim_num[i], stringify_float(period_dist[i]), stringify_float(t0_frac_dist[i]), 
                               stringify_float(ecc_dist[i]), stringify_float(per0_dist[i]), stringify_float(q_dist[i]), 
                               stringify_float(sma_dist[i]), stringify_float(incl_dist[i]), 
                               stringify_float(reqv1_dist[i]), stringify_float(reqv2_dist[i]), 
                               stringify_float(teff1_dist[i]), stringify_float(teff2_dist[i]), 
                               stringify_float(grb1_dist[i]), stringify_float(grb2_dist[i]), 
                               stringify_float(alb1_dist[i]), stringify_float(alb2_dist[i]), 
                               stringify_float(l3_dist[i])])+'\n')

        out = [stringify_float(i) for i in y]

        with open('./training/simulations.txt', 'a') as f:
            f.write(','.join(np.hstack([[sim_num[i]], out]))+'\n')

        with open('./training/psi.txt', 'a') as f:
            f.write(','.join([sim_num[i], stringify_float(mass1), stringify_float(radius1), stringify_float(logg1),
                              stringify_float(mass2), stringify_float(radius2), stringify_float(logg2)])+'\n')

