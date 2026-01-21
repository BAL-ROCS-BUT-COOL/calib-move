import cv2 as cv
import einops as eo
import numpy as np
from   numpy.typing import NDArray
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

from calib_move.util.video import get_video_frame_gry

from ..util.util import sec_2_tstr
from .containers import CLIArgs
from .containers import VideoContainer
from ..config.plotconfig import PlotConfig
from ..util.plot import fig_2_numpy


def plot_video(CLIARGS: CLIArgs, PCFG: PlotConfig, video: VideoContainer) -> list[NDArray]:
    
    # prepare data to plot. Plotly has a nice feature where if a datapoint has NaN values, it will be hidden and it handles it gracefully. Since the motion and agreement values are filled with NaN where an error occured, these points will be hidden. The time coordinate does not need to have NaNs as one is sufficient to hide the datapoint.
    data_time = np.linspace(0, video.stot, CLIARGS.n_main_steps)
    data_move = np.array(video.movements)
    data_agrm = np.array(video.agreements)
    data_errs = np.array(video.errors) # False = good value, True = error
    data_time_max = data_time[-1]
    if np.all(data_errs==True): # handle NaN-only-data case
        data_move_max = PCFG.MIN_YRANGE_AUTOMAX
        data_agrm_max = 1.0
    else:
        data_move_max = max(np.nanmax(data_move), PCFG.MIN_YRANGE_AUTOMAX) 
        data_agrm_max = 1.0
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # plotting ---------------------------------------------------------------------------------------------------------

    UNITWIDTH_BARS = data_time_max/(CLIARGS.n_main_steps-1) # bars scale with domain (half is outside at edge)
    UNITWIDTH_MARK = PCFG.WIDTH_MARK_BASE/CLIARGS.n_main_steps # markers scale with plot resolution
    CONTENTPADD = PCFG.WIDTH_MARK*(data_time_max/(CLIARGS.n_main_steps-1))*0.5 # padding for the actual plot contents
    XRANGE = [-(PCFG.PADD_X*data_time_max + CONTENTPADD), (CONTENTPADD + (PCFG.PADD_X+1)*data_time_max)]
    YRANGE1 = [-(1.00*PCFG.PADD_Y)*data_move_max, (0.30*PCFG.PADD_Y + 1)*data_move_max]
    YRANGE2 = [-(1.00*PCFG.PADD_Y)*data_agrm_max, (0.30*PCFG.PADD_Y + 1)*data_agrm_max]

    # plot 1: time series movements --------------------------------------------
    fig.add_trace(go.Scatter( # main error plot
        x=data_time, y=data_move,
        mode="markers", connectgaps=True,
        marker=dict(
            symbol="line-ew", 
            size=PCFG.WIDTH_MARK*UNITWIDTH_MARK, 
            line=dict(width=PCFG.HEIGHT_MARK, color=PCFG.COL_MOVE),
        ),
        zorder=0,
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter( # white border for main error plot markers
        x=data_time, y=data_move,
        mode="markers", connectgaps=True,
        marker=dict(
            symbol="line-ew", 
            size=PCFG.WIDTH_MARK*UNITWIDTH_MARK + 2*PCFG.MARK_BORDER, 
            line=dict(width=PCFG.HEIGHT_MARK + 2*PCFG.MARK_BORDER, color="rgba(255, 255, 255, 1.0)"),
        ),
        zorder=-1,
    ), secondary_y=False)
    
    # plot 2: agreement (confidence score) -------------------------------------
    fig.add_trace(go.Bar( # agreement on secondary y-axis
        x=data_time, y=data_agrm,
        width=PCFG.WIDTH_BARS*UNITWIDTH_BARS,
        marker=dict(
            color=PCFG.COL_AGRM,
            line=dict(width=0), # barplots have thin outline by default
        ),
        zorder=-10,
    ), secondary_y=True)

    fig.add_trace(go.Bar( # errors hatched on secondary y-axis
        x=data_time, y=np.where(data_errs==True, 1.0, np.nan),
        width=1.0*UNITWIDTH_BARS,
        marker=dict(
            color="rgba(255, 255, 255, 0.0)", # makes "bgcolor" transparent
            line=dict(width=0), # barplots have thin outline by default
            pattern=dict(shape="/", size=16, solidity=0.2, fgcolor=PCFG.COL_ERRS, fgopacity=1.0,)
        ),
        zorder=-11,
    ), secondary_y=True)

    # cosmetics ----------------------------------------------------------------
    fig.add_trace(go.Scatter( # cosmetic bar right side end
        x=[data_time_max, data_time_max], y=[YRANGE1[0], YRANGE1[1]],
        mode="lines",
        line=dict(color=PCFG.COL_ZLIN, width=PCFG.WIDTH_ZLIN),
        zorder=-20, 
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter( # static window - trace cosmetic long line (has to stop drawing below window box)
        x=[XRANGE[0], video.static_window[0], np.nan, video.static_window[1], XRANGE[1]], 
        y=5*[-(0.75*PCFG.PADD_Y)*data_move_max],
        mode="lines",
        connectgaps=False,
        line=dict(color=PCFG.COL_SWIN, width=PCFG.WIDTH_GRID),
        zorder=-21, 
    ), secondary_y=False)
    
    fig.add_shape( # static window box
        type="rect",
        x0=video.static_window[0], y0=1.00*YRANGE1[0],
        x1=video.static_window[1], y1=0.50*YRANGE1[0],
        line=dict(width=0),
        fillcolor=PCFG.COL_SWIN,
        layer="above",
        yref="y1",
    )
    
    fig.add_annotation( # static window text "ref"
        text="<b>ref</b>", font_size=PCFG.FNTSIZE_PLOT, font_color=PCFG.COL_SWIN,
        x=-0.005, y=0.75*YRANGE1[0], 
        xref="paper", yref="y1", xanchor="right", yanchor="middle",
        showarrow=False,
    )
    
    # layouting --------------------------------------------------------------------------------------------------------
    
    # general layout -----------------------------------------------------------
    fig.update_layout( # layout for all plots
        width=PCFG.PLOT_RES[1], height=PCFG.PLOT_RES[0],
        title=dict(
            text=f"motion plot for: <b>{video.name}</b>", 
            font_size=PCFG.FNTSIZE_TITLE,
            x=PCFG.TITLE_MAIN_XY[0], y=PCFG.TITLE_MAIN_XY[1], 
            xref="container", yref="container", xanchor="left", yanchor="top",
        ),
        paper_bgcolor=PCFG.COL_BAKG, plot_bgcolor=PCFG.COL_BAKG,
        font=dict(family="JetBrains Mono", size=PCFG.FNTSIZE_PLOT, color=PCFG.COL_TEXT),
        margin=PCFG.MARGIN,
        showlegend=False,
        barmode="stack", # makes bars appear above and not besides each other
    )
    
    fig.add_annotation( # artificial axis title for timeseries plot
        text="<b>|x, y| movement [px]</b>", font_size=PCFG.FNTSIZE_PLOT, textangle=-90,
        x=PCFG.TITLE_MOVE_XY[0], y=PCFG.TITLE_MOVE_XY[1], xref="paper", yref="paper", xanchor="left", yanchor="middle",
        showarrow=False,
    )
    
    fig.add_annotation( # artificial axis title for agreement plot (secondary y)
        text="<b>confidence score [%]</b>", font_size=PCFG.FNTSIZE_PLOT, textangle=-90,
        x=PCFG.TITLE_AGRM_XY[0], y=PCFG.TITLE_AGRM_XY[1], xref="paper", yref="paper", xanchor="left", yanchor="middle",
        showarrow=False,
    )
    
    # plot 1 layout -----------------------------------------------------------
    
    fig.update_xaxes(
        range=XRANGE,
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickvals=np.linspace(0, data_time_max, PCFG.N_TIX_X),
        ticktext=[sec_2_tstr(sc) for sc in np.linspace(0, data_time_max, PCFG.N_TIX_X)],
    )
    
    fig.update_yaxes( # primary y-axis (movement)
        range=YRANGE1,
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickformat=".2~s",
        tickvals=np.linspace(0, data_move_max, PCFG.N_TIX_Y),
        tickfont=dict(color=PCFG.COL_MOVE, weight="bold"),
        secondary_y=False,
    )
    
    fig.update_yaxes( # secondary y-axis (agreement)
        range=YRANGE2,
        showgrid=False,
        zeroline=False,
        tickformat=".2f",
        tickvals=np.linspace(0, data_agrm_max, PCFG.N_TIX_Y),
        tickfont=dict(color=PCFG.COL_AGRM, weight="bold"),
        secondary_y=True,
    )

    return [fig_2_numpy(fig)[..., 0:3]]
