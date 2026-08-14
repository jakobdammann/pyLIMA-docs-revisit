
import uncertainties
import matplotlib
import numpy as np
import os
from math import log
from math import log10
from scipy import stats
from math import exp
from math import pi
from math import sqrt
from math import atan2
from math import asin
from scipy.optimize import fmin_tnc
from scipy import special
from scipy import interpolate
from operator import itemgetter, attrgetter
from scipy.integrate import quad
from math import sin
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import VBBinaryLensing
VBB=VBBinaryLensing.VBBinaryLensing()
VBB.Tol=float(0.001)
VBB.satellite=int(0)

def pspl_n_imagesf(t, u0, te, t0):
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


def reldev(q, t, u0, te, t0, fs, fb, ferr, rs, xp, yp):
    """
    Calculates deviation metrics based on current parameters and sky position (xp, yp).
    """
    
    pspl, u_plus, u_minus, prox, proy, mu_plus, mu_minus, spplus, spminus = pspl_n_imagesf(t, u0, te, t0)
    
    # 2. Calculate binary/source positional terms
    bsx = (t - t0) / te
    bsy = u0
    invq1 = 1.0 / (q + 1.0)
    
    # 3. Separation Calculation, mind the literature uses s...
    s = np.sqrt(xp**2 + yp**2)
    
    if s > 0.01:
        # Determine offset from magnification origin (VBB)
        com = np.array([xp / s, yp / s])
        com = invq1 * com
        com = s * q * com
        
        nsvec = np.array([bsx, bsy]) - com
        
        # Determine rotation angles
        phi1 = np.angle(complex(xp, yp))
        phi2 = np.angle(complex(nsvec[0], nsvec[1]))
        phi3 = np.angle(complex(bsx, bsy))
        
        angle = phi2 - phi1 - phi3 + np.pi * 0.5
        angle_alt = -phi1 + np.pi * 0.5
        
        # Rotation matrix construction
        rmatrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Applying rotation to the source position vector (bsx, bsy)
        bsinmap = np.dot(rmatrix, np.array([bsx, bsy]))
        
        # 5. Calculation
        newout = 0.0
        
        # Use the required dependency functions
        result = VBB.BinaryMag(s, q, bsinmap[1], bsinmap[0], rs, 0.001)
        
        # Calculate the final metric
        delta_chisqr = (fs * result - fs * pspl)**2 / ferr**2
        
        return delta_chisqr#, q, xp, yp, s, rs, bsinmap[1], bsinmap[0], newout
    else:
        # Case where d <= 0.005
        return 0.0#, q, xp, yp, s, rs, 1.0, 1.0, -1.0


# --- Main Execution Block for Grid Calculation ---

def run_grid_analysis(t, q, u0, te, t0, fs, fb, ferr, rs, 
                      x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=10
                      ):
    """
    Performs the analysis by iterating over a specified grid of (xp, yp) values
    and calling reldev for each point.
    """

    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    results = {}
    print(f"Iterating over a grid of {resolution} x {resolution} points...")
    
    for xp in x_range:
        for yp in y_range:
            try:
                # Call the non-vectorizable function
                result = reldev(
                    q=q,         
                    t=t, u0=u0, te=te, t0=t0, 
                    fs=fs, fb=fb, ferr=ferr, rs=rs, 
                    xp=xp, yp=yp
                )
                
                results[(xp, yp)] = result
                
            except Exception as e:
                print(f"Error at ({xp:.2f}, {yp:.2f}): {e}")
                results[(xp, yp)] = None

    print("--- Grid Analysis Complete ---")
    return results


def run_grid_analysis_and_plot(
    t, q, u0, te, t0, fs, fb, ferr, rs, 
    x_min=-2.5, x_max=2.5, y_min=-2.5, y_max=2.5, resolution=100, resolution_interpol=5
):
    """
    Performs the analysis by iterating over a specified grid of (xp, yp) values
    calling reldev, and then visualizes the resulting metric value.
    """
    print("--- Starting Grid Analysis ---")
    
    # Setup Grid Coordinates
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
            
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    def vectorized_reldev_wrapper(X_arr, Y_arr):
        """
        Wraps the single-point reldev to accept and process full X and Y matrices.
        """
        vectorized_func = np.vectorize(lambda x, y: reldev(q, t, u0, te, t0, fs, fb, ferr, rs, x, y))
        return vectorized_func(X_arr, Y_arr)

    # Computing reldev
    Z = np.log10(vectorized_reldev_wrapper(X_grid, Y_grid))

    # Interpolating for higher resolution
    x_fine = np.linspace(x_min, x_max, resolution * resolution_interpol)
    y_fine = np.linspace(y_min, y_max, resolution * resolution_interpol)   
    X_f, Y_f = np.meshgrid(x_fine, y_fine)
    interp_func = interpolate.RegularGridInterpolator((x_range, y_range), Z.T, method="linear")

    # Plotting the result
    points = np.array([X_f.ravel(), Y_f.ravel()]).T
    Z_fine = interp_func(points).reshape(X_f.shape)
    plot_heatmap(X_f, Y_f, Z_fine, 
                 add_trajectory=True, t0=t0, u0=u0, te=te
                 )
        
    print("--- Planet Grid on Lens Plane Complete ---")


def plot_heatmap(X, Y, Z, 
                 Zmin=-5, Zmax=5, 
                 add_cbar=True, add_contour=True, add_grid=True, add_tE=True,
                 add_trajectory=False, t0=None, u0=None, te=None, t=None,
                 show_plot=True
                ):
    """Helper function to generate the colored contour plot."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # create fig
    plt.figure(figsize=(8, 7))

    # heatmap
    contour = plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.turbo, vmin=Zmin, vmax=Zmax)
    if add_contour:
        plt.contour(X, Y, Z, levels=14, colors='k', linewidths=0.7, alpha=0.3)

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
        plt.colorbar(contour, label='Metric Value ($\\log_{10}(\\Delta\\mu/\\mu$))')

    # source trajectory
    if add_trajectory:
        plt.hlines(u0, xmin=np.min(X), xmax=np.max(X), 
                   linestyle='-', lw=1, color='k', label="Source Trajectory", zorder=1
                   )
        if t is not None:
            bsx = (t - t0) / te
            bsy = u0
            plt.scatter(bsx, np.ones_like(bsx)*bsy, s=2, c='red', marker='x', label='Datapoints',
                        zorder=2)

    # labels
    plt.xlabel('X lens plane ($\\theta_{\\rm E}$)')
    plt.ylabel('Y lens plane ($\\theta_{\\rm E}$)')
    plt.legend(loc='lower right')

    # title
    plt.title('Relative deviation caused by 2nd lens $\\log_{10}(\\Delta\\mu/\\mu$))')
    plt.gca().set_aspect('equal', adjustable='box')

    # show
    if show_plot:
        plt.show()


if __name__ == '__main__':
    q = 0.001 # assumption
    rho = 0.0005
    t = 7775. # from datapoint
    ferr = 0.02
    u0 = 0.3 # from fit
    te = 10.0
    t0 = 7777.0
    fs = 1.0
    fb = 0.0 # Not used in the provided snippet, for the future
    
    run_grid_analysis_and_plot(q=q,
        t=t, u0=u0, te=te, t0=t0, fs=fs, fb=fb, ferr=ferr, rs=rho,resolution=250, resolution_interpol=5
    )


