class PlotConfig:
    
    PLOT_RES = [300, 1000]
    
    # colors -------------------------------------------------------------------
    COL_TEXT = "rgba( 20,  20,  20, 1.00)"
    COL_BAKG = "rgba(255, 255, 255, 1.00)"
    COL_GRID = "rgba(200, 200, 200, 1.00)"
    COL_ZLIN = COL_GRID
    
    COL_MOVE = "rgba(178,  34,  34, 1.00)"
    COL_ERRS = "rgba(178,  34,  34, 0.40)"
    COL_AGRM = "rgba( 46, 139,  87, 0.35)"
    COL_SWIN = "rgba(144, 213, 255, 0.55)"
    
    # sizes --------------------------------------------------------------------
    WIDTH_GRID = 2.0
    WIDTH_ZLIN = 5.0
    
    WIDTH_BARS = 0.4 # just between 1 and 0 (1 is no gap between two bars, and 0 is no bar width)
    WIDTH_MARK = 0.6 # just between 1 and 0 (1 is no gap between two markers, and 0 is no marker width)
    WIDTH_MARK_BASE = 570 # px, total plot width aparently
    HEIGHT_MARK = 6.0
    MARK_BORDER = 2.0
    
    FNTSIZE_TITLE = 20
    FNTSIZE_PLOT = 16
    
    # padding and positions ----------------------------------------------------
    PADD_X = 0.02
    PADD_Y = 0.25
    BASEMARGIN = 20
    MARGIN = dict(l=60+BASEMARGIN, r=BASEMARGIN+15, t=45+BASEMARGIN, b=BASEMARGIN, pad=5)
    
    TITLE_MAIN_XY = [ 0.015, 0.920]
    TITLE_MOVE_XY = [-0.078, 0.500]
    TITLE_AGRM_XY = [ 1.000, 0.500]
    
    # axes ---------------------------------------------------------------------
    MIN_YRANGE_AUTOMAX = 10
    N_TIX_X = 7
    N_TIX_Y = 4
    


    




    


    
    


    