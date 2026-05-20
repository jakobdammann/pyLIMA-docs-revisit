from graphviz import Digraph

# Graph Settings
g = Digraph('PyLIMA_Schematic_Workflow', format='pdf')
g.attr(compound='true')
g.attr(rankdir='LR')
g.attr('node', shape='box', style='filled', fontname='Helvetica', fontsize='12')
# g.attr('node', width='2.0', height='1.4', fixedsize='shape')
g.attr('graph', fontname='Helvetica')
g.attr(splines='ortho', color='darkgrey')
g.attr(nodesep='0.25')
g.attr(ranksep='0.7')

# Colors
COLORS = {
    "data": "#DDDDDD",
    "telescope": "#CFFBAF",
    "telescope0": "#E8FFC7",
    "event": "#A4FFAB",
    "model": "#B7DBFF",
    "model0": "#DCFAF4",
    "model_add": "#F3E8FF",
    "fitter": "#FDEFA8",
    "fitter_grad": "#FFDCA0",
    "fitter_mc": "#FFBBA7",
    "fitter_other": "#FFFDC9",
    "obj_func": "#FFDAD5",
    "fitter_input": "#FFF1CC",
    "results": "#98FFCA",
    "physical": "#90FF9B",
    "visualisation": "#ABFFE3"
}

####################################################################################################################################

# Telescopes
with g.subgraph(name='cluster_telescope') as c:
    # attributes
    c.attr(label='<<B>Telescope</B><BR/>telescopes.Telescope()>')
    c.attr('node', fillcolor=COLORS['telescope'])
    # nodes
    c.node(
        'sim_telescope',
        '<<B>Simulated Telescope</B><BR/>(created with<BR/>simulations.simulator.simulate_a_telescope())>',
        fillcolor=COLORS['telescope0']
        )
    c.node(
        'telescope',
        '<<B>Telescope</B><BR/>(corresponds to<BR/>one dataset and<BR/>one filter)>'
        )

# Nodes
with g.subgraph(name='cluster_data') as c:
    # attributes
    c.attr(label='<Data>')
    c.attr('node', fillcolor=COLORS['data'])
    # nodes
    with c.subgraph(name='cluster_data_telescope') as cc:
        cc.attr(label='')
        cc.node(
            'limb_darkening',
            '<<B>Limb Darkening</B><BR/>Coefficients<BR/>(per Telescope &amp; Filter)>'
            )
        cc.node(
            'location',
            '<<B>Location</B><BR/>(per Telescope)>'
            )
        cc.node(
            'filter',
            '<<B>Filter</B><BR/>(per Lightcurve)>'
            )
        cc.node(
            'lightcurve',
            '<<B>Lightcurve</B><BR/>or<BR/><B>Astrometry</B><BR/>(may have to be binned)>'
            )
    c.node(
        'ra_dec',
        '<<B>RA, DEC</B><BR/>(coordinates of object)>', 
    )

# Event
with g.subgraph(name='cluster_event') as c:
    # attributes
    c.attr(label='<<B>Event</B><BR/>event.Event()>')
    c.attr('node', fillcolor=COLORS['event'])
    # nodes
    c.node(
        'event',
        '<<B>Event</B><BR/>(contains array <BR/>of Telescopes &amp; RA, DEC,<BR/>defines alignement data)>'
        )

# Edges
g.edge('lightcurve', 'telescope', ltail='cluster_data_telescope', lhead='cluster_telescope')
g.edge('telescope', 'event', ltail='cluster_telescope', lhead='cluster_event')
g.edge('ra_dec', 'event', lhead='cluster_event')
g.edge('ra_dec', 'telescope', style='invis')

###########################################################################################################################################

# Models
with g.subgraph(name='cluster_models') as c:
    # attributes
    c.attr(label='<<B>Model</B><BR/>models.???model()>')
    c.attr('node', fillcolor=COLORS['model0'])
    # nodes
    c.node(
        'model_usbl',
        '<<B>USBL</B><BR/>Uniform Source Binary Lens<BR/>without LD, no caustic>'
    )
    c.node(
        'model_fsbl',
        '<<B>FSBL</B><BR/>Finite Source Binary Lens<BR/>with LD>'
    )
    c.node(
        'model_psbl',
        '<<B>PSBL</B><BR/>Point Source Binary Lens>'
    )
    c.attr('node', fillcolor=COLORS['model'])
    c.node(
        'model_dspl',
        '<<B>DSPL</B><BR/>Double Source Point Lens<BR/>no caustic, two peaks>'
    )
    c.node(
        'model_fsplarge',
        '<<B>FSPLarge</B>>'
    )
    c.node(
        'model_fspl',
        '<<B>FSPL</B><BR/>Finite Source Point Lens<BR/>with LD, short tE and/or small u0>',
    )
    c.node(
        'model_pspl',
        '<<B>PSPL</B><BR/>Point Source Point Lens>'
    )

# Edges
g.edge('event', 'model_pspl', ltail='cluster_event', lhead='cluster_models')

# cluster Additional Modelling
with g.subgraph(name='cluster_additional') as c:
    # attr
    c.attr(label='Additional Modelling')
    c.attr('node', fillcolor=COLORS['model_add'])
    # nodes
    c.node(
        'orbital_motion_lens',
        '<<B>Orbital Motion of Lens</B>>',
        )
    c.node(
        'xallarap',
        '<<B>Xallarap</B><BR/>orbital motion of source>',
        )
    c.node(
        'parallax',
        '<<B>Parallax</B><BR/>terrestrial, annual<BR/>and/or space>',
        )

# Edges
g.edge('xallarap', 'model_fsbl', ltail='cluster_additional', lhead='cluster_models')
# Invis Edges
g.edge('event', 'parallax', style='invis')
# step.edge('model_pspl', 'model_fspl', style='invis')

###########################################################################################################################################

# Fitter
with g.subgraph(name='cluster_fitter') as c:
    # attributes
    c.attr(label='<<B>Fitter / Optimizer</B><BR/>fits.???fit()>')
    c.attr('node', fillcolor=COLORS['fitter'])
    # nodes other and custom
    with c.subgraph(name='cluster_other_custom') as cc:
        cc.attr(label='', color='none')
        cc.node(
            'fitter_custom',
            'Custom Optimizers\n(Simplex/Nealder-Mead, etc.)',
            fillcolor=COLORS['fitter_other']
            )
        cc.node(
            'fitter_other',
            'Other Development\nOptimizers\n(Bootstrap, MINUIT, etc.)',
            fillcolor=COLORS['fitter_other']
            )
    # node mcmc
    c.node(
        'fitter_mcmc',
        '<<B>MCMC</B><BR/>Mote-Carlo-Markov-Chain,<BR/>emcee<BR/>(slow, outputs posterior/uncertainty)>',
        fillcolor=COLORS['fitter_mc']
        )
    # cluster gradient methods
    with c.subgraph(name='cluster_gradient') as cc:
        # attributes
        cc.attr(label='Gradient-Like Fits')
        cc.attr('node', fillcolor=COLORS['fitter_grad'])
        # nodes
        cc.node(
            'fitter_trf',
            '<<B>TRF</B><BR/>Trust Region Reflective>',
            )
        cc.node(
            'fitter_lm',
            '<<B>LM</B><BR/>Levenberg-Marquardt<BR/>(fast, ideal for refinement)>',
            )
        cc.node(
            'fitter_bfgs',
            '<<B>BFGS</B>>',
            )
    # node de
    c.node(
        'fitter_de',
        '<<B>DE</B><BR/>Differential Evolution<BR/>(ideal for global search)>'
        )
    
# Edges
g.edge('model_dspl', 'fitter_mcmc', ltail='cluster_models', lhead='cluster_fitter')
g.edge('model_dspl', 'fitter_mcmc', ltail='cluster_models', lhead='cluster_fitter', style='invis')
# g.edge('model_pspl', 'fitter_de', constraint="false")
# g.edge('model_usbl', 'fitter_lm')
# g.edge('model_pspl', 'fitter_mcmc', constraint="false")
# g.edge('model_usbl', 'fitter_mcmc', constraint="false")
# g.edge('model_dspl', 'fitter_lm')

# Node Initial Guess
with g.subgraph(name='cluster_initguess') as c:
    # attribute
    c.attr(color='none')
    # node
    c.node(
        'init_guess',
        '<<B>Boundary Conditions</B><BR/>or<BR/><B>Inital Guess</B><BR/>(can be from previous fit)>',
        fillcolor=COLORS['fitter_input']
        )
g.edge('init_guess', 'fitter_de', lhead='cluster_fitter')

# Cluster Objective Functions
with g.subgraph(name='cluster_objfunc') as c:
    # attr
    c.attr(label='<<B>Objective Function</B>>')
    c.attr('node', fillcolor=COLORS['obj_func'])
    # nodes
    c.node(
        'objfunc_custom',
        'Custom',
        )
    c.node(
        'objfunc_likelihood',
        'Log-Likelihood',
        )
    c.node(
        'objfunc_softl1',
        'Soft L1',
        )
    c.node(
        'objfunc_chisq',
        'Chi Squared',
        )
g.edge('objfunc_softl1', 'fitter_lm', ltail='cluster_objfunc', lhead='cluster_fitter')

# Cluster Optional
with g.subgraph(name='cluster_optional') as c:
    # attributes
    c.attr(label='<Optional Modifications>')
    c.attr('node', fillcolor=COLORS['fitter_input'])
    # Node Rescale Uncertainties
    c.node(
        'rescale_uncertainties',
        '<<B>Rescale Uncertainties</B><BR/>(when uncertainties are<BR/>under-/over-estimated)>',
        )
    # Node Coordinate Origin
    c.node(
        'coord_origin',
        '<<B>Coordinate Origin</B><BR/>(central, planetary<BR/>or manual)>',
        )
    # Node Fancy Parameters
    c.node(
        'fancyparams',
        '<<B>Fancy Parameters</B><BR/>(modifies parameter space to<BR/>e.g. log-space)>',
        )
    # Node Prior
    c.node(
        'prior',
        '<<B>Prior Distribution</B><BR/>(Bayesian inference<BR/>when using MCMC)>',
        )
# g.edge('prior', 'fitter_mcmc', lhead='cluster_fitter')
g.edge('fancyparams', 'fitter_custom', ltail='cluster_optional', lhead='cluster_fitter')
# g.edge('rescale_uncertainties', 'fitter_custom', lhead='cluster_fitter')

# Invis Edges
g.edge('model_dspl', 'rescale_uncertainties', style='invis')
# g.edge('fitter_de', 'fitter_bfgs', style='invis')

###########################################################################################################################################

# Results
with g.subgraph(name='cluster_results') as c:
    # attributes
    c.attr(label='<Results>')
    # nodes
    c.node(
        'results',
        '<<B>Best Fit<BR/>Parameters</B><BR/>(posterior/uncertainty,<BR/>info criteria)>',
        fillcolor=COLORS['results']
    )

# Edges
g.edge('fitter_lm', 'results', ltail='cluster_fitter', lhead='cluster_results')
# back at initial guess
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false', style='invis')
g.edge('results', 'init_guess', ltail='cluster_results', constraint='false')

# Physical Interpretation
with g.subgraph(name='cluster_physical') as c:
    # attributes
    c.attr(label='Physical Interpretation')
    # nodes
    c.node(
        'physical_interpretation',
        '<<B>pyLIMASS</B><BR/>infer mass>',
        shape='ellipse',
        fillcolor=COLORS['physical']
    )

# Visualisation
with g.subgraph(name='cluster_visualisation') as c:
    # attributes
    c.attr(label='Visualisation')
    # nodes
    c.node(
        'plot',
        '<<B>Plot Lightcurve</B><BR/>pyLIMA_plots.plot_lightcurves()>',
        fillcolor=COLORS['visualisation']
    )

# Edges
g.edge('results', 'physical_interpretation', ltail='cluster_results', lhead='cluster_physical')
g.edge('results', 'plot', ltail='cluster_results', lhead='cluster_visualisation')

##############################################################################################################################

# Title
g.attr(label="pyLIMA Workflow", labelloc="t", fontsize="24")

# Render PDF
g.render('./diagram/pylima_workflow', view=False)