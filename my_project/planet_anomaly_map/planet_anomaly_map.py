import reldev_grid

from collections import OrderedDict
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import lintsampler
from scipy.interpolate import RegularGridInterpolator

from pyLIMA.fits import TRF_fit
from pyLIMA.models import PSPL_model, PSBL_model
from pyLIMA.outputs import pyLIMA_plots

############################
# PLANET ANOMALY MAP CLASS #
############################

class PlanetAnomalyMap(object):
    """
    This class inhabits all the functionality for creating and sampling a 
    planet anomaly map.

    Attributes
    ----------

    event : pyLIMA event object
    window_limits: (tuple x limits,tuple)
    """

    def __init__(self, event, window_limits=None):
        self.event = event

        self.ax_xlim = window_limits[0]
        self.ax_ylim = window_limits[1]

    #########
    # UTILS #
    #########

    def _set_window_limits(self, ax):
        ax.set_xlim(*self.ax_xlim)
        ax.set_ylim(*self.ax_ylim)

    #############
    # PSPL STEP #
    #############

    # PSPL FIT
    def run_pspl_fit(self, parameter_guess=None):
        # PSPL model
        self.pspl = PSPL_model.PSPLmodel(self.event)
        # Robust fit
        self.pspl_fit = TRF_fit.TRFfit(self.pspl, loss_function='soft_l1')
        if parameter_guess != None:
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
            plt.show()
        return fig

    ###########################
    # ANOMALY MAP COMPUTATION #
    ###########################

    def create_planet_anomaly_map(self, 
                                  meshgrid_corners=((-2.0,2.0),(-2.0,2.0)), 
                                  resolution=100):
        verbose = False

        # assumptions
        q = 0.001 
        rs = 0.0005

        # algorithm parameters
        t_range = (self.t0 - 5, self.t0 + 5) # distance from t0 to compute reldev [in JD]

        # Setup Meshgrid
        x_min = meshgrid_corners[0][0]
        x_max = meshgrid_corners[0][1]
        y_min = meshgrid_corners[1][0]
        y_max = meshgrid_corners[0][1]
        self.x_range = np.linspace(x_min, x_max, resolution)
        self.y_range = np.linspace(y_min, y_max, resolution)
        self.X_grid, self.Y_grid = np.meshgrid(self.x_range, self.y_range)

        # Setup dict for saving values while running
        self.reldev_maps = {}
        self.reldev_maps['t'] = []
        self.reldev_maps['name'] = []
        self.reldev_maps['weight'] = []
        self.reldev_maps['detection_zone'] = []

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

                # setup a vectorized function to compute reldev at each point
                def vectorized_reldev_wrapper(X_arr, Y_arr):
                    vectorized_func = np.vectorize(lambda x, y: reldev_grid.reldev(q, t, 
                                                                                   self.u0, self.te, self.t0, 
                                                                                   fs, fb, ferr, rs, x, y))
                    return vectorized_func(X_arr, Y_arr)
                # TODO: remove this vectorized function wrapper declaration for performance
                
                # compute reldev values on meshgrid
                if verbose:
                    print(f'Running Grid Analysis for {name} at t={t}.')
                Z = vectorized_reldev_wrapper(self.X_grid, self.Y_grid)

                # compute weight by difference of pspl and real flux
                pspl = reldev_grid.pspl_n_imagesf(t, self.u0, self.te, self.t0)[0]
                fmodel = fs * pspl + fb
                weight = (freal - fmodel)**2 / ferr**2 # use delta chisq 
                if verbose:
                    print('real:', freal, ' model:', fmodel, ' weight:', weight)

                # save
                self.reldev_maps['t'].append(t)
                self.reldev_maps['name'].append(name)
                self.reldev_maps['weight'].append(weight)
                self.reldev_maps['detection_zone'].append(Z)

        # average the reldev maps
        self.detection_zone_map = np.average(self.reldev_maps['detection_zone'], axis=0)
        self.planet_anomaly_map = np.average(self.reldev_maps['detection_zone'], axis=0, 
                                             weights=self.reldev_maps['weight'])

        print('Planet Anomaly Map done.')
        return self.planet_anomaly_map

    ###############
    # RETURN DATA #
    ###############

    def get_planet_anomaly_map(self):
        return (self.X_grid, self.Y_grid), self.planet_anomaly_map, 

    def get_planet_detection_zone_map(self):
        return (self.X_grid, self.Y_grid), self.detection_zone_map

    #########
    # PLOTS #
    #########

    # PLOT WITH ALL MAPS
    def plot_overview(self, show=True):
        '''
        Create a figure of all maps, one per datapoint.
        '''
        n_maps = len(self.reldev_maps['detection_zone']) 
        nx_maps = 4
        ny_maps = math.ceil(n_maps/nx_maps)
        # print(n_maps, nx_maps, ny_maps)

        plt.figure(figsize=(3*nx_maps, 3*ny_maps))
        # plt.suptitle('Relative deviation caused by 2nd lens $\\log_{10}(\\Delta\\mu/\\mu$))')

        for i in range(n_maps):
            # unpack
            name = self.reldev_maps['name'][i]
            t = self.reldev_maps['t'][i]
            map = self.reldev_maps['detection_zone'][i]
            weight = self.reldev_maps['weight'][i]

            # create subplot
            plt.subplot(ny_maps, nx_maps, i+1)
            plt.title(f'{name}, $t = t0 + {t - self.t0:.1f}$, $w = {weight:.1f}$')

            # heatmap
            norm = colors.LogNorm(1e-5, 1e8)
            plt.pcolormesh(self.X_grid, self.Y_grid, map*weight, shading='auto', cmap=cm.turbo, norm=norm)

            # trajectory
            plt.hlines(self.u0, xmin=np.min(self.X_grid), xmax=np.max(self.X_grid), 
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
        if show:
            plt.show()
        return plt.gcf()

    # PLOT OF PLANET ANOMALY MAP
    def plot_planet_anomaly_map(self, show=True):
        # Heatmap Plot
        plot_heatmap(self.X_grid, self.Y_grid, self.planet_anomaly_map, 
                                norm=colors.LogNorm(1e-5,1e6),
                                add_cbar=True, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                                add_contour=True, add_grid=True, add_tE=True,
                                add_trajectory=True, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
        plt.title('Planet Anomaly Map')
        if show:
            plt.show()
        return plt.gcf()

    # PLOT OF PLANET DETECTION ZONE
    def plot_planet_detection_zone_map(self, show=True):
        # Heatmap Plot
        plot_heatmap(self.X_grid, self.Y_grid, self.detection_zone_map, 
                        norm=colors.LogNorm(1e-5,1e6),
                        add_cbar=True, cbar_label='Relative Deviation $\\sum \\, \\chi_i^2$',
                        add_contour=True, add_grid=True, add_tE=True,
                        add_trajectory=True, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
        plt.title('Planet Detection Zone of Checked Datapoints')
        if show:
            plt.show()
        return plt.gcf()

    ##########################
    # Sample Values from Map #
    ##########################

    def sample_from_planet_anomaly_map(self, n_samples=100):
        ''' 
        Use Planet Anomaly Map as a Distribution for Initial Values
        '''
        # choose what to use as a distribution/statistic
        self.anomaly_statistic = self.planet_anomaly_map / np.sum(self.planet_anomaly_map)

        # create vectorized function with Grid Interpolator
        matrix_interp = RegularGridInterpolator((self.x_range, self.y_range), self.anomaly_statistic.T)

        # create sampler and sample points
        sampler = lintsampler.LintSampler((self.x_range, self.y_range), matrix_interp, vectorizedpdf=True)
        self.samples = sampler.sample(n_samples)

        return self.samples

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
            plot_heatmap(self.alpha_grid, self.s_grid, self.planet_anomaly_map, 
                            norm=colors.LogNorm(1e-5,1e6),
                            add_cbar=True, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                            add_contour=True, add_grid=True, add_tE=False,
                            add_trajectory=False, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
            plt.scatter(alpha_samples, s_samples, c='k', alpha=0.5, marker='x', label='Samples')
            plt.xlabel('Planet Angle $\\alpha$')
            plt.ylabel('Planet Separation $s$ [$\\theta_{\\rm E}$]')

        # Create a Plot in x, y Space
        else:
            # plot
            plot_heatmap(self.X_grid, self.Y_grid, self.planet_anomaly_map, 
                                    norm=colors.LogNorm(1e-5,1e6),
                                    add_cbar=True, cbar_label='Weighted Relative Deviation $\\sum \\, w_i \\cdot \\chi_i^2$',
                                    add_contour=True, add_grid=True, add_tE=False,
                                    add_trajectory=False, t0=self.t0, u0=self.u0, te=self.te, t=self.reldev_maps['t'])
            plt.scatter(x_samples, y_samples, c='k', alpha=0.5, marker='x', label='Samples')
        # finish plot
        plt.title('Planet Anomaly Map')
        plt.legend()

        if show:
            plt.show()
        return plt.gcf()


#########
# UTILS #
#########

# generic heatmap plot
def plot_heatmap(X, Y, Z, 
                add_cbar=True, norm=colors.LogNorm(1e-5, 1e5), cbar_label='$\\sum\,\\chi^2$',
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
        plt.plot(x, y, linestyle='--', lw=1, color='darkblue', label="$\\theta_E$")

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
            bsx = (t - t0) / te
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
