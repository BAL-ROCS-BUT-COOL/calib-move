import numpy as np
from   numpy.typing import NDArray
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

from ..util.util import sec_2_tstr
from .containers import CLIArgs
from .containers import VideoContainer
from ..config.plotconfig import PlotConfig
from ..util.plot import fig_2_numpy


def plot_video(CLIARGS: CLIArgs, video: VideoContainer, PCFG: PlotConfig) -> list[NDArray]:
    
    data_mask = np.array(video.errors) == 1
    data_time = np.linspace(0, video.stot, CLIARGS.n_main_steps) # not necessary to also mask this
    data_move = np.where(data_mask, np.nan, np.array(video.motion))
    data_move_max = max(np.nanmax(data_move), PCFG.MIN_YRANGE_AUTOMAX)
    
    fig = go.Figure()
    
    # plotting ---------------------------------------------------------------------------------------------------------

    # plot 1: time series ------------------------------------------------------
    fig.add_trace(go.Scatter( # main timeseries-motion plot
        x=data_time, y=data_move,
        mode="lines+markers", connectgaps=True,
        line=dict(color=PCFG.COL_TRAC, width=PCFG.WIDTH_TRAC, dash="dot"),
        marker=dict(color=PCFG.COL_MARK, size=PCFG.WIDTH_MARK, symbol="square"),
    ))
    fig.add_trace(go.Scatter( # cosmetic bar right side end
        x=[data_time[-1], data_time[-1]], y=[-(PCFG.PADD_MOVE)*data_move_max, (1+PCFG.PADD_MOVE)*data_move_max],
        mode="lines",
        line=dict(color=PCFG.COL_ZLIN, width=PCFG.WIDTH_ZLIN),
        zorder=-1,
    ))
    fig.add_shape( # static window box
        type="rect",
        x0=video.static_window[0], y0=-(PCFG.PADD_MOVE/2)*data_move_max,
        x1=video.static_window[1], y1=(1+PCFG.PADD_MOVE/2)*data_move_max,
        line=dict(width=0),
        fillcolor=PCFG.COL_SWIN,
        layer="below",
    )
    
    # layouting --------------------------------------------------------------------------------------------------------
    
    # general layout -----------------------------------------------------------
    fig.update_layout( # layout for all plots
        width=PCFG.PLOT_RES[1], height=PCFG.PLOT_RES[0],
        title=dict(
            text=f"plot for: <b>{video.name}</b>", 
            font_size=PCFG.FNTSIZE_TITLE,
            x=PCFG.TITLE_MAIN_XY[0], y=PCFG.TITLE_MAIN_XY[1], 
            xref="container", yref="container", xanchor="left", yanchor="top",
        ),
        paper_bgcolor=PCFG.COL_BAKG, plot_bgcolor=PCFG.COL_BAKG,
        font=dict(family="JetBrains Mono", size=PCFG.FNTSIZE_PLOT, color=PCFG.COL_TEXT),
        margin=PCFG.MARGIN,
        showlegend=False,
    )
    fig.add_annotation( # artificial axis title for timeseries plot
        text="<b>|x, y| movement [px]</b>", font_size=PCFG.FNTSIZE_PLOT, textangle=-90,
        x=PCFG.TITLE_MOVE_XY[0], y=PCFG.TITLE_MOVE_XY[1], xref="paper", yref="paper", xanchor="left", yanchor="middle",
        showarrow=False,
    )
    
    # plot 1 layout -----------------------------------------------------------
    fig.update_xaxes(
        range=[-(PCFG.PADD_TIME)*data_time[-1], (PCFG.PADD_TIME+1)*data_time[-1]],
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickvals=np.linspace(data_time[0], data_time[-1], PCFG.NTIX_TS),
        ticktext=[sec_2_tstr(sc) for sc in np.linspace(0, video.stot, PCFG.NTIX_TS)],
    )
    fig.update_yaxes(
        range=[-(PCFG.PADD_MOVE)*data_move_max, (PCFG.PADD_MOVE+1)*data_move_max],
        showgrid=True,
        zeroline=True,
        gridcolor=PCFG.COL_GRID,
        zerolinecolor=PCFG.COL_ZLIN,
        gridwidth=PCFG.WIDTH_GRID,
        zerolinewidth=PCFG.WIDTH_ZLIN,
        tickformat=".1~s",
        tickvals=np.linspace(0, data_move_max, PCFG.NTIX_DAT), 
    )

    return [fig_2_numpy(fig)]

 