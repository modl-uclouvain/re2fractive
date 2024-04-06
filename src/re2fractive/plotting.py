import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parity_plot(test_moddata, predictions, uncertainties):
    colours = px.colors.qualitative.Plotly

    point_colour = colours[0]
    error_colour = colours[1]
    point_colour_2 = colours[2]

    target = test_moddata.df_targets.columns[0]
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"colspan": 2}, {}], [{}, {}]],
    )

    fig.update_layout(
        template="none",
        font_family="Arial",
        width=1000,
        height=800,
    )

    # Add y=x parity line
    fig.add_trace(
        go.Scatter(
            x=[
                0,
                test_moddata.df_targets[target].max(),
            ],
            y=[
                0,
                test_moddata.df_targets[target].max(),
            ],
            mode="lines",
            line=dict(color="DarkSlateGrey", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=test_moddata.df_targets[target],
            y=predictions[target],
            error_y=dict(
                type="data",
                array=uncertainties[target],
                visible=True,
                color=error_colour,
            ),
            mode="markers",
            marker=dict(
                size=5,
                symbol="circle",
                opacity=0.5,
                line=dict(width=1, color="DarkSlateGrey"),
                color=point_colour,
            ),
            showlegend=False,
            hovertext=test_moddata.df_targets.index,
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    # Label axes
    fig.update_xaxes(
        title_text=f'True {target.replace("_", " ").title()}', row=1, col=1
    )
    fig.update_yaxes(
        title_text=f'Predicted {target.replace("_", " ").title()}', row=1, col=1
    )

    # Start both axes at 0
    fig.update_xaxes(range=[0, None], zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[0, None], zeroline=False, row=1, col=1)

    errors = test_moddata.df_targets[target] - predictions[target]

    # Add y=x parity line
    fig.add_trace(
        go.Scatter(
            x=[
                0,
                test_moddata.df_targets[target].max(),
            ],
            y=[
                0,
                0,
            ],
            mode="lines",
            line=dict(color="DarkSlateGrey", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=test_moddata.df_targets[target],
            y=errors,
            error_y=dict(
                type="data",
                array=uncertainties[target],
                visible=True,
                color=error_colour,
            ),
            mode="markers",
            marker=dict(
                size=5,
                symbol="circle",
                opacity=0.5,
                line=dict(width=1, color="DarkSlateGrey"),
                color=point_colour_2,
            ),
            showlegend=False,
            hovertext=test_moddata.df_targets.index,
            hoverinfo="text",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text=f'True {target.replace("_", " ").title()}', row=2, col=1
    )
    fig.update_yaxes(
        title_text=f'{target.replace("_", " ").title()} prediction error', row=2, col=1
    )

    max_uncertainty = uncertainties[target].max()

    fig.update_yaxes(
        range=[errors.min() - max_uncertainty, errors.max() + max_uncertainty],
        zeroline=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[errors.min() - max_uncertainty, errors.max() + max_uncertainty],
        zeroline=True,
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(y=errors, showlegend=False, marker_color=error_colour),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Frequency", row=2, col=2)
    fig.update_yaxes(
        title_text=f'{target.replace("_", " ").title()} prediction error', row=2, col=2
    )

    fig.show()
