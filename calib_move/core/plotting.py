import numpy as np
from   numpy.typing import NDArray
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

from ..util.util import sec_2_tstr
from .containers import CLIArgs
from .containers import VideoContainer
from ..config.plotconfig import PlotConfig
from ..util.plot import fig_2_numpy

WIDTH_BARS = 0.4 # just between 1 and 0 (1 is no gap between two bars, and 0 is no bar width)

WIDTH_MARK = 0.6 # just between 1 and 0 (1 is no gap between two bars, and 0 is no bar width)
WIDTH_MARK_BASE = 570 # px, total plot width aparently
HEIGHT_MARK = 6.0
WIDTH_MARK_BORDER = 3.0

def plot_video(CLIARGS: CLIArgs, video: VideoContainer, PCFG: PlotConfig) -> list[NDArray]:
    
    # prepare data to plot. Plotly has a nice feature where if a datapoint has NaN values, it will be hidden and it handles it gracefully. Since the motion and agreement values are filled with NaN where an error occured, these points will be hidden. The time coordinate does not need to have NaNs as one is sufficient to hide the datapoint.
    data_time = np.linspace(0, video.stot, CLIARGS.n_main_steps)
    data_move = np.array(video.movements)
    data_agrm = np.array(video.agreements)
    data_errs = np.array(video.errors) # False = good value, True = error
    data_time_max = data_time[-1]
    data_move_max = max(np.nanmax(data_move), PCFG.MIN_YRANGE_AUTOMAX) 
    data_agrm_max = 1.0
    # TODO: handle NaN-only case (also in general)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    contentpadd = WIDTH_MARK*(data_time_max/(CLIARGS.n_main_steps-1))*0.5 # padding for the actual plot contents
    
    # plotting ---------------------------------------------------------------------------------------------------------

    # plot 1: time series ------------------------------------------------------
    fig.add_trace(go.Scatter( # main error plot
        zorder=0,
        x=data_time, y=data_move,
        mode="markers", connectgaps=True,
        marker=dict(
            size=WIDTH_MARK*(WIDTH_MARK_BASE/(CLIARGS.n_main_steps)), 
            symbol="line-ew", 
            line=dict(width=HEIGHT_MARK, color=PCFG.COL_MOVE),
        ),
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter( # white border for main error plot
        zorder=-1,
        x=data_time, y=data_move,
        mode="markers", connectgaps=True,
        marker=dict(
            size=WIDTH_MARK*(WIDTH_MARK_BASE/(CLIARGS.n_main_steps))+2*WIDTH_MARK_BORDER, 
            symbol="line-ew", 
            line=dict(width=HEIGHT_MARK+2*WIDTH_MARK_BORDER, color="rgba(255, 255, 255, 1.0)"),
        ),
    ), secondary_y=False)
    
    # plot 2: agreement (confidence score) -------------------------------------
    fig.add_trace(go.Bar( # agreement on secondary y-axis
        zorder=-10,
        x=data_time, y=data_agrm,
        width=WIDTH_BARS*(data_time_max/(CLIARGS.n_main_steps-1)),
        marker=dict(
            color=PCFG.COL_AGRM,
            line=dict(width=0),
        ),
    ), secondary_y=True)

    fig.add_trace(go.Bar( # errors hatched on secondary y-axis
        zorder=-11,
        x=data_time, y=np.where(data_errs==True, 1.0, np.nan),
        width=1.0*(data_time_max/(CLIARGS.n_main_steps-1)),
        marker=dict(
            color="rgba(255, 255, 255, 0.0)",
            line=dict(width=0),
            pattern=dict(
                shape="/", 
                size=16, 
                solidity=0.2,
                fgcolor="rgba(178,  34,  34, 0.4)",
                fgopacity=1.0,
            )
        ),
    ), secondary_y=True)

    # cosmetics ----------------------------------------------------------------
    fig.add_trace(go.Scatter( # cosmetic bar right side end
        zorder=-20, 
        x=[data_time_max, data_time_max], y=[-(1.00*PCFG.PADD_Y)*data_move_max, (1.00*PCFG.PADD_Y + 1)*data_move_max],
        mode="lines",
        line=dict(color=PCFG.COL_ZLIN, width=PCFG.WIDTH_ZLIN),
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter( # static window - trace cosmetic long line
        zorder=-21, 
        x=[0, data_time_max], y=2*[-(0.70*PCFG.PADD_Y)*data_move_max],
        mode="lines",
        line=dict(color=PCFG.COL_SWIN, width=PCFG.WIDTH_ZLIN),
    ), secondary_y=False)
    
    fig.add_shape( # static window box
        type="rect",
        x0=video.static_window[0], y0=-(1.00*PCFG.PADD_Y)*data_move_max,
        x1=video.static_window[1], y1=-(0.40*PCFG.PADD_Y)*data_move_max,
        line=dict(width=0),
        fillcolor="rgba(255, 255, 255, 1.0)",
        layer="above",
        yref="y1",
    )
    
    fig.add_shape( # static window box - white mask to have correct blending with trace behind it
        type="rect",
        x0=video.static_window[0], y0=-(1.00*PCFG.PADD_Y)*data_move_max,
        x1=video.static_window[1], y1=-(0.40*PCFG.PADD_Y)*data_move_max,
        line=dict(width=0),
        fillcolor=PCFG.COL_SWIN,
        layer="above",
        yref="y1",
    )
    
    fig.add_annotation( # static window box text
        text="<b>ref.</b>", font_size=12,
        x=(video.static_window[0]+video.static_window[1])/2, y=-(0.70*PCFG.PADD_Y)*data_move_max, 
        xref="x", yref="y1", xanchor="center", yanchor="middle",
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
        range=[-(contentpadd + PCFG.PADD_X*data_time_max), (contentpadd + (PCFG.PADD_X+1)*data_time_max)],
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickvals=np.linspace(data_time[0], data_time_max, PCFG.N_TIX_X),
        ticktext=[sec_2_tstr(sc) for sc in np.linspace(0, data_time_max, PCFG.N_TIX_X)],
    )
    
    fig.update_yaxes( # primary y-axis (movement)
        range=[-(1.00*PCFG.PADD_Y)*data_move_max, (0.40*PCFG.PADD_Y + 1)*data_move_max],
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickformat=".1~s",
        tickvals=np.linspace(0, data_move_max, PCFG.N_TIX_Y),
        tickfont=dict(color=PCFG.COL_MOVE, weight="bold"),
        secondary_y=False,
    )
    
    fig.update_yaxes( # secondary y-axis (agreement)
        range=[-(1.00*PCFG.PADD_Y)*data_agrm_max, (0.40*PCFG.PADD_Y + 1)*data_agrm_max],
        showgrid=False,
        zeroline=False,
        
        tickformat=".2f",
        tickvals=np.linspace(0, data_agrm_max, PCFG.N_TIX_Y),
        tickfont=dict(color=PCFG.COL_AGRM, weight="bold"),
        secondary_y=True,
    )


    fig.show()
    return [fig_2_numpy(fig)]

