
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
from operator import itemgetter, attrgetter
from scipy.integrate import quad
from math import sin
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import VBBinaryLensing
VBB=VBBinaryLensing.VBBinaryLensing()
VBB.Tol=float(0.05)
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
    d = np.sqrt(xp**2 + yp**2)
    
    if d > 0.01:
        # Determine offset from magnification origin (VBB)
        com = np.array([xp / d, yp / d])
        com = invq1 * com
        com = d * q * com
        
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
        result = VBB.BinaryMag0(d, q, bsinmap[1], bsinmap[0])
        
        # Calculate the final metric
        delta_chisqr = (fs * result - fs * pspl)**2 / ferr**2
        
        return delta_chisqr, q, xp, yp, d, rs, bsinmap[1], bsinmap[0], newout
    else:
        # Case where d <= 0.005
        return 0.0, q, xp, yp, d, rs, 1.0, 1.0, -1.0

# --- Main Execution Block for Grid Calculation ---

def run_grid_analysis(t, u0, te, t0, fs, fb, ferr, rs, x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=10):
    """
    Performs the analysis by iterating over a specified grid of (xp, yp) values
    and calling reldev for each point.
    """
    
    # 1. Setup Grid Coordinates
    # Use linspace for defined starting/ending points and control over density
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    
    # Initialize storage for results
    # Since the output structure is complex, we will store a dictionary mapping (xp, yp) to results
    results = {}
    
    # 2. Iterative Function Calls (Not Vectorizable)
    print(f"Iterating over a grid of {resolution} x {resolution} points...")
    
    for xp in x_range:
        for yp in y_range:
            try:
                # Call the non-vectorizable function
                result = reldev(
                    q=0.001,         # Assuming 'q' needs a default value if not defined in scope
                    t=t, u0=u0, te=te, t0=t0, 
                    fs=fs, fb=fb, ferr=ferr, rs=rs, 
                    xp=xp, yp=yp
                )
                
                # Store results keyed by the input coordinates
                results[(xp, yp)] = result
                
            except Exception as e:
                print(f"Error at ({xp:.2f}, {yp:.2f}): {e}")
                # Store None or an error indicator if the function fails for a point
                results[(xp, yp)] = None

    print("--- Grid Analysis Complete ---")
    return results


def run_grid_analysis_and_plot(
    t, q, u0, te, t0, fs, fb, ferr, rs, 
    x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, resolution=100
):
    """
    Performs the analysis by iterating over a specified grid of (xp, yp) values
    calling reldev, and then visualizes the resulting metric value.
    """
    print("--- Starting Grid Analysis ---")
    
    # 1. Setup Grid Coordinates
    x_range = np.linspace(x_min, x_max, resolution)
    y_range = np.linspace(y_min, y_max, resolution)
    
    # Initialize storage for results (dictionary key: (xp, yp), value: result tuple)
    results = {}
    
    # 2. Iterative Function Calls (The inherent limitation: reldev is not vectorizable)
    print(f"Iterating over a grid of {resolution} x {resolution} points (this step is slow)...")
    
    # --- EXECUTION BLOCK ---
    for xp in x_range:
        for yp in y_range:
            try:
                # Call the expensive, non-vectorizable function
                result = reldev(
                    q, t=t, u0=u0, te=te, t0=t0, 
                    fs=fs, fb=fb, ferr=ferr, rs=rs, 
                    xp=xp, yp=yp
                )
                
                # Store results keyed by the input coordinates
                results[(xp, yp)] = result
            except Exception as e:
                print(f"Error at ({xp:.2f}, {yp:.2f}): {e}")
                results[(xp, yp)] = (np.nan, np.nan) # Store NaN if calculation fails

    print("--- Grid Analysis Complete ---")
        
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    def vectorized_reldev_wrapper(X_arr, Y_arr):
        """
        Wraps the single-point reldev to accept and process full X and Y matrices.
        """
        vectorized_func = np.vectorize(lambda x, y: reldev(q, t, u0, te, t0, fs, fb, ferr, rs, x, y)[0])
        return vectorized_func(X_arr, Y_arr)

    try:
        # Calculate the entire Z matrix using the vectorized wrapper
        Z = np.log10(vectorized_reldev_wrapper(X_grid, Y_grid))
        
        # Plotting the result
        plot_heatmap(X_grid, Y_grid, Z, x_range, y_range)
        
    except Exception as e:
        print(f"\nERROR during plotting preparation: {e}")

def plot_heatmap(X, Y, Z, x_range, y_range):
    """Helper function to generate the colored contour plot."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.figure(figsize=(10, 8))
    contour = plt.pcolormesh(X, Y, Z, shading='auto', cmap=cm.jet)
    plt.colorbar(contour, label='Metric Value ($\\Delta\\chi^2$)')
    plt.contour(X, Y, Z, levels=10, colors='k', linewidths=0.7)


    plt.xlabel('X Coordinate ($x_p$)')
    plt.ylabel('Y Coordinate ($y_p$)')
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    plt.plot(x, y, linestyle='--', color='darkblue',label="$\\theta_E$",lw=3)

    plt.title('Heatmap of Metric $\\Delta\\chi^2$ Across the Grid')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    q = 0.01
    t = 7724.
    u0 = 0.34
    te = 30.0
    t0 = 7777.0
    fs = 2.0
    fb = 0.0 # Not used in the provided snippet, for the future
    ferr = 0.02
    rs = 0.05
    
    run_grid_analysis_and_plot(q=q,
        t=t, u0=u0, te=te, t0=t0, fs=fs, fb=fb, ferr=ferr, rs=rs,resolution=800
    )


