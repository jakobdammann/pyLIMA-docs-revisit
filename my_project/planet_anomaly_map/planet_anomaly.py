from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.tri as mtri

import multiprocessing

import lintsampler
from scipy.interpolate import RegularGridInterpolator

from pyLIMA.fits import TRF_fit
from pyLIMA.models import PSPL_model, PSBL_model
from pyLIMA.outputs import pyLIMA_plots

import VBBinaryLensing
VBB=VBBinaryLensing.VBBinaryLensing()
VBB.Tol=float(0.001)
VBB.satellite=int(0)

############################
# PLANET ANOMALY MAP CLASS #
############################

class PlanetAnomalyDetector(object):
    """
    This class inhabits all the functionality for creating and sampling a 
    planet anomaly map.

    Attributes
    ----------

    event : pyLIMA event object
    window_limits: ((float,float),(float,float)), x and y limits of lightcurve plot
    """

    def __init__(self, event, window_limits=None):
        self.event = event
        self.window_limits = window_limits

        self.planetAnomalyMap = None

    #########
    # UTILS #
    #########

    def _set_window_limits(self, ax):
        if self.window_limits is not None:
            ax.set_xlim(*self.window_limits[0])
            ax.set_ylim(*self.window_limits[1])

    #############
    # PSPL STEP #
    #############

    # PSPL FIT
    def run_pspl_fit(self, parameter_guess=None):
        # PSPL model
        self.pspl = PSPL_model.PSPLmodel(self.event)
        # Robust fit
        self.pspl_fit = TRF_fit.TRFfit(self.pspl, loss_function='soft_l1')
        if parameter_guess is not None:
            self.pspl_fit.model_parameters_guess = parameter_guess
        self.pspl_fit.fit()
        self.pspl_params =  OrderedDict(zip(self.pspl_fit.priors_parameters.keys(), 
                                            self.pspl_fit.fit_results['best_model']))

        # get parameters from pspl fit
        self.t0 = self.pspl_params['t0']
        self.u0 = self.pspl_params['u0']
        self.te = self.pspl_params['tE']
        self.tE = self.te

    # PLOT
    def plot_pspl_model(self, show=True):
        # Plot
        fig, _ = pyLIMA_plots.plot_lightcurves(self.pspl, 
                                                self.pspl_fit.fit_results['best_model'])
        # Set the window dimension.
        self._set_window_limits(fig.axes[0])

        # Line at t0
        fig.axes[0].axvline(self.pspl_params['t0'], color='gray', linestyle='--')

        # show
        if show:
            fig.show()
        return fig

    ###################################
    # ANOMALY MAP COMPUTATION IN GRID #
    ###################################

    # PSPL calculation
    @staticmethod
    def _pspl_n_imagesf(t, u0, te, t0):
        """Calculates various parameters related to PSPL."""
        u_sqr = u0**2 + ((t - t0) / te)**2
        u_sqrt4 = (u_sqr + 4.)**0.5
        u = np.sqrt(u_sqr)
        
        u_plus = 0.5 * (u + u_sqrt4)
        u_minus = 0.5 * (u - u_sqrt4)
        
        # Handle potential division by zero if u_plus*2 - u_minus**2 is zero
        denominator = (u_plus * 2 - u_minus**2)
        mu_plus = u_plus**2 / denominator if denominator != 0 else np.inf
        mu_minus = u_minus**2 / denominator if denominator != 0 else np.inf
        
        spplus = 2.0 * mu_plus - 1.0
        spminus = 2.0 * mu_minus
        
        prox = (t - t0) / (te * u) if u != 0 else np.nan
        proy = u0 / u if u != 0 else np.nan
        pspl = (u_sqr + 2.0) / (u * u_sqrt4) if u != 0 and u_sqrt4 != 0 else np.nan
        
        return pspl, u_plus, u_minus, prox, proy, mu_plus, mu_minus, spplus, spminus

    # Calculates RelDev for a Map/Grid using VBB
    @staticmethod
    def _reldev_map(q, t, u0, te, t0, fs, fb, ferr, rs, X, Y, s_min=0.01, tol=0.001): 
        """
        Calculates deviation metrics based on current parameters and sky position (xp, yp).

        Adapted from Markus Hundertmark.
        """
        
        out = PlanetAnomalyDetector._pspl_n_imagesf(t, u0, te, t0)
        pspl, u_plus, u_minus, prox, proy, mu_plus, mu_minus, spplus, spminus = out
        
        # calculate binary/source positional terms
        bsx = (t - t0) / te
        bsy = u0
        invq1 = 1.0 / (q + 1.0)
        
        # Separation Calculation
        s = np.sqrt(X**2 + Y**2)

        # mask only for large s
        mask = s > s_min
        xp, yp, sm = X[mask], Y[mask], s[mask]

        # compute vector between binary and planet
        k = q / (1.0 + q)
        nsx = bsx - k * xp
        nsy = bsy - k * yp
                        
        # determine rotation angles
        phi1 = np.arctan2(xp, yp)
        phi2 = np.arctan2(nsx, nsy)
        phi3 = np.arctan2(bsx, bsy)
            
        angle = phi2 - phi1 - phi3 + np.pi * 0.5

        # apply rotation to bs directly (skip rotation matrix)
        cos, sin = np.cos(angle), np.sin(angle)
        bx = cos * bsx - sin * bsy
        by = sin * bsx + cos * bsy
        
        # Use the required dependency functions
        # result = VBB.BinaryMag(s, q, bsinmap[1], bsinmap[0], rs, 0.001)
        mag = np.fromiter((VBB.BinaryMag(si, q, yi, xi, rs, tol) 
                           for si, yi, xi in zip(sm, by, bx)),
                           dtype=float, count=sm.size)
        
        # Calculate the final metric
        Z = np.zeros_like(X, dtype=float)
        Z[mask] = (fs * mag - fs * pspl)**2 / ferr**2

        # I somehow have to transpose the map to be in the correct orientation
        Z = Z.T
        
        return Z

    # Creates the Planet Anomaly Map and Detection Zone Map
    def create_planet_anomaly_map(self, 
                                  meshgrid_corners=((-2.0,2.0),(-2.0,2.0)), resolution=100,
                                  q=0.001, rs=0.0005, delta_time=5,
                                  n_processes=None, verbose=False):
        # algorithm parameters
        t_range = (self.t0 - delta_time, self.t0 + delta_time) # distance interval around t0 to compute reldev [in JD]

        # Setup Meshgrid
        x_min = meshgrid_corners[0][0]
        x_max = meshgrid_corners[0][1]
        y_min = meshgrid_corners[1][0]
        y_max = meshgrid_corners[1][1]
        self.x_range = np.linspace(x_min, x_max, resolution)
        self.y_range = np.linspace(y_min, y_max, resolution)
        self.X_grid, self.Y_grid = np.meshgrid(self.x_range, self.y_range)

        # Setup dict for saving values while running
        self.reldev_maps = {}
        self.reldev_maps['t'] = []
        self.reldev_maps['name'] = []
        self.reldev_maps['weight'] = []
        self.reldev_maps['detection_zone'] = []

        # Multithreading
        jobs = []

        # per telescope
        for telescope in self.event.telescopes:

            name = telescope.name
            # telescope parameters from fit
            fs = self.pspl_params['fsource_' + name]
            fb = self.pspl_params['ftotal_' + name] - self.pspl_params['fsource_' + name]

            # crop lightcurve to t_range
            time = telescope.lightcurve['time'].value
            lightcurve = telescope.lightcurve[(time > t_range[0]) & (time < t_range[1])]

            # per datapoint
            for datapoint in lightcurve:

                # parameters from datapoint
                t = datapoint['time'].value # in [JD]
                try:
                    freal = datapoint['flux'].value
                    ferr = datapoint['err_flux'].value # in [W/m^2]
                except:
                    freal = datapoint['flux']
                    ferr = datapoint['err_flux']

                # compute weight by difference of pspl and real flux
                pspl = self._pspl_n_imagesf(t, self.u0, self.te, self.t0)[0]
                fmodel = fs * pspl + fb
                weight = (freal - fmodel)**2 / ferr**2 # use delta chisq
                if verbose:
                    print('real:', freal, ' model:', fmodel, ' weight:', weight)

                # save
                self.reldev_maps['t'].append(t)
                self.reldev_maps['name'].append(name)
                self.reldev_maps['weight'].append(weight)

                # append arguments to multiprocessing jobs
                jobs.append((q, t, self.u0, self.te, self.t0, 
                            fs, fb, ferr, rs, self.X_grid, self.Y_grid))

        # multiprocessing to compute reldev values on meshgrid
        with multiprocessing.Pool(n_processes) as pool:
            self.reldev_maps['detection_zone'] = pool.starmap(PlanetAnomalyDetector._reldev_map, jobs)

        # average the reldev maps
        self.planetAnomalyMap = PlanetAnomalyMap()
        self.planetAnomalyMap.create_with_reldev_maps(self.reldev_maps, self.X_grid, self.Y_grid)

        print('Planet Anomaly Map done.')
        return self.planetAnomalyMap


    ###################################################
    # ANOMALY MAP COMPUTATION WITH REJECTION SAMPLING #
    ###################################################


    def create_planet_anomaly_map_with_rejection_sampling(self, 
                                  xy_limits=((-2.0,2.0),(-2.0,2.0)),
                                  logq_limits=(-5,-1),
                                  M=10, n_samples=100, batch_size=10, 
                                  rs=0.0005, delta_time=5,
                                  n_processes=None, verbose=False):

        # algorithm parameters
        t_range = (self.t0 - delta_time, self.t0 + delta_time) # distance interval around t0 to compute reldev [in JD]
        x_limits = xy_limits[0]
        y_limits = xy_limits[1]
        low = (x_limits[0], y_limits[0], logq_limits[0])
        high = (x_limits[1], y_limits[1], logq_limits[1])

        # random number generator
        rng = np.random.default_rng(0)
        out = []

        ### Rejection Sampling
        while len(out) < n_samples:

            # random variables for rejection sampling
            xylogq = rng.uniform(low, high, size=(batch_size, 3))
            u = rng.uniform(0, M, size=batch_size)

            ### Extract part of lightcurve from each telescope
            for telescope in self.event.telescopes:
    
                name = telescope.name
                # telescope parameters from fit
                fs = self.pspl_params['fsource_' + name]
                fb = self.pspl_params['ftotal_' + name] - self.pspl_params['fsource_' + name]
    
                # crop lightcurve to t_range
                time = telescope.lightcurve['time'].value
                lightcurve = telescope.lightcurve[(time > t_range[0]) & (time < t_range[1])]

                ### Compute statistic for each datapoint
                for datapoint in lightcurve:
    
                    # parameters from datapoint
                    t = datapoint['time'].value # in [JD]
                    try:
                        freal = datapoint['flux'].value
                        ferr = datapoint['err_flux'].value # in [W/m^2]
                    except:
                        freal = datapoint['flux']
                        ferr = datapoint['err_flux']
    
                    # compute weight by difference of pspl and real flux
                    pspl = self._pspl_n_imagesf(t, self.u0, self.te, self.t0)[0]
                    fmodel = fs * pspl + fb
                    weight = (freal - fmodel)**2 / ferr**2 # delta chisq
                    if verbose:
                        print('real:', freal, ' model:', fmodel, ' weight:', weight)
                
            out.extend(xylogq)

        # crop out in case last batch
        out = out[:n_samples]
        print(out)


    #########
    # PLOTS #
    #########

    # generic heatmap plot
    @staticmethod
    def _plot_heatmap(X, Y, Z, 
                    add_cbar=True, norm=colors.LogNorm(1e-5, 1e5), cbar_label=r'$\\sum\,\\chi^2$',
                    add_contour=True, add_grid=True, add_tE=True,
                    add_trajectory=False, t0=None, u0=None, te=None, t=None):
        """
        Function to generate a generic colored contour heatmap plot for 
        planet anomaly maps or similar.
        """

        # create fig
        plt.figure(figsize=(8, 7))

        # heatmap
        contour = plt.pcolormesh(X, Y, Z, 
                                shading='auto', cmap=cm.turbo, norm=norm
                                )
        if add_contour:
            plt.contour(X, Y, Z, levels=14, colors='k', linewidths=0.7, alpha=0.3, norm=norm)

        # Einstein radius
        if add_tE:
            theta = np.linspace(0, 2 * np.pi, 100)
            radius = 1
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            plt.plot(x, y, linestyle='--', lw=1, color='darkblue', label="Einstein Radius $\\theta_E$")

        # grid
        if add_grid:
            plt.grid(True, linestyle='--', alpha=0.6)

        # colorbar
        if add_cbar:
            plt.colorbar(contour, label=cbar_label)

        # source trajectory
        if add_trajectory:
            plt.hlines(u0, xmin=np.min(X), xmax=np.max(X), 
                    linestyle='-', lw=1, color='k', label="Source Trajectory", 
                    zorder=1
                    )
            if t is not None:
                bsx = (np.asarray(t) - t0) / te
                bsy = u0
                plt.scatter(bsx, np.ones_like(bsx)*bsy, 
                            s=2, c='red', marker='x', label='Source Pos. at Obs.', 
                            zorder=2)

        # image trajectories
        if add_trajectory:
            t_arr = np.linspace(t0-te, t0+te, 100)
            bsx = (t_arr - t0) / te
            bsy = u0

            u_sqr = u0**2 + ((t_arr - t0) / te)**2
            u_sqrt4 = (u_sqr + 4.)**0.5
            u = np.sqrt(u_sqr)
            
            u_plus = 0.5 * (u + u_sqrt4)
            u_minus = 0.5 * (u - u_sqrt4)

            plt.plot(bsx*u_plus/u, bsy*u_plus/u, linestyle=':', lw=1, color='darkblue', label="Image Trajectory")
            plt.plot(bsx*u_minus/u, bsy*u_minus/u, linestyle=':', lw=1, color='darkblue')

        # labels
        plt.xlabel('X lens plane [$\\theta_{\\rm E}$]')
        plt.ylabel('Y lens plane [$\\theta_{\\rm E}$]')
        if plt.gca().get_legend_handles_labels()[1]:
            plt.legend(loc='lower right')

        # title
        plt.title('Relative deviation caused by 2nd lens')
        plt.gca().set_aspect('equal', adjustable='box')

        return plt.gcf()

    # PLOT WITH ALL MAPS
    def plot_overview(self, show=True):
        '''
        Create a figure of all maps, one per datapoint.
        '''
        # get RelDev Maps from Planet Anomaly Map
        reldev_maps = self.planetAnomalyMap.reldev_maps
        X_grid = self.planetAnomalyMap.X_grid
        Y_grid = self.planetAnomalyMap.Y_grid

        n_maps = len(reldev_maps['detection_zone']) 
        nx_maps = 4
        ny_maps = math.ceil(n_maps/nx_maps)
        # print(n_maps, nx_maps, ny_maps)

        plt.figure(figsize=(3*nx_maps, 3*ny_maps))
        # plt.suptitle('Relative deviation caused by 2nd lens $\\log_{10}(\\Delta\\mu/\\mu$))')

        for i in range(n_maps):
            # unpack
            name = reldev_maps['name'][i]
            t = reldev_maps['t'][i]
            map = reldev_maps['detection_zone'][i]
            weight = reldev_maps['weight'][i]

            # create subplot
            plt.subplot(ny_maps, nx_maps, i+1)
            plt.title(f'{name}, $t = t0 + {t - self.t0:.1f}$, $w = {weight:.1f}$')

            # heatmap
            norm = colors.LogNorm(1e-5, 1e8)
            plt.pcolormesh(X_grid, Y_grid, map*weight, shading='auto', cmap=cm.turbo, norm=norm)

            # trajectory
            plt.hlines(self.u0, xmin=np.min(X_grid), xmax=np.max(X_grid), 
                linestyle='-', lw=1, color='k', label="Source Trajectory")
            plt.scatter((t - self.t0) / self.te, self.u0, 
                        s=6, c='red', marker='x', label='Source Pos. at Obs.', zorder=2)
            
            # Einstein ring
            theta = np.linspace(0, 2 * np.pi, 100)
            x = 1 * np.cos(theta)
            y = 1 * np.sin(theta)
            plt.plot(x, y, linestyle='--', lw=1, alpha=0.5, color='darkblue', label="$\\theta_E$")

        # show
        plt.tight_layout()
        fig = plt.gcf()
        if show:
            fig.show()
        return fig

    # PLOT OF PLANET ANOMALY MAP
    def plot_planet_anomaly_map(self, show=True):
        # get Planet Anomaly Map as a grid
        (X_grid, Y_grid), Z_grid = self.planetAnomalyMap.get_planet_anomaly_map_grid()

        # Heatmap Plot
        PlanetAnomalyDetector._plot_heatmap(X_grid, Y_grid, Z_grid, 
                                norm=colors.LogNorm(1e-5,1e6),
                                add_cbar=True, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                                add_contour=True, add_grid=True, add_tE=True,
                                add_trajectory=True, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
        plt.title('Planet Anomaly Map')
        fig = plt.gcf()
        if show:
            fig.show()
        return fig

    # PLOT OF PLANET DETECTION ZONE
    def plot_planet_detection_zone_map(self, show=True):
        # get Planet Anomaly Map as a grid
        (X_grid, Y_grid), Z_grid = self.planetAnomalyMap.get_planet_detection_zone_map_grid()

        # Heatmap Plot
        PlanetAnomalyDetector._plot_heatmap(X_grid, Y_grid, Z_grid, 
                        norm=colors.LogNorm(1e-5,1e6),
                        add_cbar=True, cbar_label='Relative Deviation $\\sum \\, \\chi_i^2$',
                        add_contour=True, add_grid=True, add_tE=True,
                        add_trajectory=True, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
        plt.title('Planet Detection Zone of Checked Datapoints')
        fig = plt.gcf()
        if show:
            fig.show()
        return fig

    ##########################
    # Sample Values from Map #
    ##########################

    def sample_from_planet_anomaly_map(self, n_samples=100):
        ''' 
        Use Planet Anomaly Map as a Distribution for Initial Values
        '''
        # check mode
        mode = self.planetAnomalyMap.mode

        if mode == 'grid':
            (X_grid, Y_grid), Z_grid = self.planetAnomalyMap.get_planet_anomaly_map_grid()

            # choose what to use as a distribution/statistic
            self.anomaly_statistic = Z_grid / np.sum(Z_grid)

            # create vectorized function with Grid Interpolator
            matrix_interp = RegularGridInterpolator((self.x_range, self.y_range), self.anomaly_statistic.T)

            # create sampler and sample points
            sampler = lintsampler.LintSampler((self.x_range, self.y_range), matrix_interp, vectorizedpdf=True)
            self.samples = sampler.sample(n_samples)

            return self.samples

        if mode == 'dist':
            pass

    def convert_x_y_to_s_alpha_samples(self):
        # unpack sample coordinates
        x_samples = self.samples[:,0]
        y_samples = self.samples[:,1]

        # Convert samples from x,y to s,alpha
        s_samples = np.hypot(x_samples, y_samples)
        alpha_samples = np.arctan2(x_samples, y_samples) + np.pi/2
        # wrap alpha around
        alpha_samples = np.mod(alpha_samples, 2*np.pi)
        self.s_alpha_samples = np.stack((s_samples, alpha_samples), axis=-1)

        # Convert the grid
        self.s_grid = np.hypot(self.X_grid, self.Y_grid)
        alpha_grid = np.arctan2(self.X_grid, self.Y_grid) + np.pi/2
        self.alpha_grid = np.mod(alpha_grid, 2*np.pi)

        return self.s_alpha_samples

    def create_sample_overview_plot(self, show=True, s_alpha_plot=False):
        # unpack planet anomaly map object
        (X_grid, Y_grid), Z_grid = self.planetAnomalyMap.get_planet_anomaly_map_grid()
        # unpack sample coordinates
        x_samples = self.samples[:,0]
        y_samples = self.samples[:,1]
        try:
            s_samples = self.s_alpha_samples[:,0]
            alpha_samples = self.s_alpha_samples[:,1]
        except:
            self.convert_x_y_to_s_alpha_samples()
            s_samples = self.s_alpha_samples[:,0]
            alpha_samples = self.s_alpha_samples[:,1]

        # Create a Plot in s, alpha Space
        if s_alpha_plot:
            # plot
            PlanetAnomalyDetector._plot_heatmap(self.alpha_grid, self.s_grid, Z_grid, 
                            norm=colors.LogNorm(1e-5,1e6),
                            add_cbar=False, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                            add_contour=False, add_grid=True, add_tE=False,
                            add_trajectory=False, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
            plt.gca().clear()
            # using triangulation
            x = X_grid.ravel()
            y = Y_grid.ravel()
            a = self.alpha_grid.ravel()
            s = self.s_grid.ravel()
            z = Z_grid.ravel()
            mask = (z > 0) & (s > 0.01)
            tri = mtri.Triangulation(a, s, triangles=mtri.Triangulation(x,y).triangles)
            tri.set_mask(~mask[tri.triangles].all(axis=1) | (np.ptp(a[tri.triangles], axis=1) > np.pi))
            tc = plt.tripcolor(tri, z, 
                               cmap=cm.turbo, norm=colors.LogNorm(1e-5, 1e6), 
                               shading='gouraud')
            plt.colorbar(tc, label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$')
            plt.tricontour(tri, z, levels=14, colors='k', linewidths=0.7, alpha=0.3, norm=colors.LogNorm(1e-5, 1e6))
            # scatter samples
            plt.scatter(alpha_samples, s_samples, c='k', alpha=0.5, marker='x', label='Samples')
            plt.xlabel('Planet Angle $\\alpha$')
            plt.ylabel('Planet Separation $s$ [$\\theta_{\\rm E}$]')
            plt.grid()

        # Create a Plot in x, y Space
        else:
            # plot
            PlanetAnomalyDetector._plot_heatmap(X_grid, Y_grid, Z_grid, 
                                    norm=colors.LogNorm(1e-5,1e6),
                                    add_cbar=True, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                                    add_contour=True, add_grid=True, add_tE=False,
                                    add_trajectory=False, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
            plt.scatter(x_samples, y_samples, c='k', alpha=0.5, marker='x', label='Samples')
        # finish plot
        plt.title('Planet Anomaly Map')
        plt.legend()

        fig = plt.gcf()
        if show:
            fig.show()
        return fig



class PlanetAnomalyMap(object):
    """
    This class inhabits the functions for storing a planet anomaly map 
    as a grid or as a distribution.

    Attributes
    ----------

    """

    def __init__(self, mode=None):
        self.mode = mode

    def create_with_reldev_maps(self, reldev_maps, X_grid, Y_grid):
        self.mode = 'grid'

        self.reldev_maps = reldev_maps
        self.X_grid = X_grid
        self.Y_grid = Y_grid

        self.detection_zone_map = np.average(self.reldev_maps['detection_zone'], axis=0)
        self.planet_anomaly_map = np.average(self.reldev_maps['detection_zone'], axis=0, 
                                            weights=self.reldev_maps['weight'])

    def create_with_distribution(self, dist):
        mode = 'dist'

    ###############
    # RETURN DATA #
    ###############

    def get_planet_anomaly_map_grid(self):
        if self.mode == 'grid':
            return (self.X_grid, self.Y_grid), self.planet_anomaly_map

        if self.mode == 'dist':
            # bin distribution into grid
            pass

    def get_planet_detection_zone_map_grid(self):
        if self.mode == 'grid':
            return (self.X_grid, self.Y_grid), self.detection_zone_map

        if self.mode == 'dist':
            # bin distribution into grid
            pass

        

if __name__ == '__main__':
    from pyLIMA import event
    from pyLIMA.simulations import simulator
    from pyLIMA.fits import TRF_fit
    from pyLIMA.models import PSBL_model
    from pyLIMA.outputs import pyLIMA_plots

    ######################################################
    #### Create simulated Microlensing event
    event_3 = event.Event(ra=270, dec=0)
    event_3.name = 'My simulated event'

    CTIO_I = simulator.simulate_a_telescope(name='CTIO_I', time_start=2457190, time_end=2457220, sampling=6, 
                                            location='Earth', camera_filter='I',
                                            uniform_sampling=True, astrometry=False)
    # LCO_SAAO = simulator.simulate_a_telescope(name='LCO_SAAO', time_start=2457205, time_end=2457208, sampling=1, 
    #                                         location='Earth', camera_filter='I',
    #                                         uniform_sampling=True, astrometry=False)

    event_3.telescopes.append(CTIO_I)
    # event_3.telescopes.append(LCO_SAAO)
    event_3.check_event()

    psbl_sim = PSBL_model.PSBLmodel(event_3)
    psbl_sim_params = simulator.simulate_microlensing_model_parameters(psbl_sim)

    ### Set the parameters to some reasonable values.
    psbl_sim_params[0] = 2457205.21 # t0
    psbl_sim_params[1:3] = [0.309, 10.7821] # u0, tE
    psbl_sim_params[3:6] = [1.3, 0.001, 1.9] # s, q, alpha

    ### Set the source and blend fluxes of all telescopes to the same values for plotting purposes.
    n_tel = len(event_3.telescopes)
    psbl_sim_params[-n_tel*2::2] = 10000. * np.ones(n_tel) # source fluxes
    psbl_sim_params[-n_tel*2+1::2] = 12000. * np.ones(n_tel) # total fluxes
    print(psbl_sim_params)

    ### Transform the parameters into a pyLIMA class object.
    pyLIMA_parameters_1 = psbl_sim.compute_pyLIMA_parameters(psbl_sim_params)

    ### Inject everthing into the simulator to produce a light curve.
    simulator.simulate_lightcurve(psbl_sim, pyLIMA_parameters_1)

    ######################################################
    ### Use Detector to run a PSPL fit
    anomaly_detector = PlanetAnomalyDetector(event=event_3, 
                                window_limits=((2457200.0, 2457210.5),(16.8, 15.9)))

    anomaly_detector.run_pspl_fit()

    ##################################################
    ### Create Anomaly Map
    anomaly_map = anomaly_detector.create_planet_anomaly_map_with_rejection_sampling()