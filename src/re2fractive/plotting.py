import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modnet.preprocessing import MODData
from plotly.subplots import make_subplots


def plot_design_space(
    campaign,
    aux: str = "band_gap",
    show: bool = True,
):
    colours = px.colors.qualitative.Plotly

    point_colour = colours[0]
    point_colour_2 = colours[2]
    point_colour_3 = colours[3]
    point_colour_4 = colours[1]

    fig = go.Figure()

    fig.update_layout(
        template="none",
        font_family="Arial",
        width=1000,
        height=800,
        title=f"Design space after {len(campaign.epochs)} iteration(s)",
    )

    # plot initial oracle set
    initial_dataset = campaign.datasets[0]
    if isinstance(initial_dataset, type):
        initial_dataset = initial_dataset.load()

    target = next(iter(initial_dataset.targets))

    target_name = initial_dataset.properties[target]
    aux_name = initial_dataset.properties[aux]

    target_y = [d.as_dict["attributes"][target_name] for d in initial_dataset.data]
    target_x = [d.as_dict["attributes"][aux_name] for d in initial_dataset.data]

    fig.add_trace(
        go.Scatter(
            x=target_x,
            y=target_y,
            mode="markers",
            marker=dict(
                size=5,
                symbol="circle",
                opacity=0.1,
                line=dict(width=1, color="DarkSlateGrey"),
                color=point_colour,
            ),
            name=initial_dataset.__class__.__name__,
        ),
    )

    bins = np.arange(0, 15, step=0.5)
    top = 20

    epoch = campaign.epochs[-1]

    if len(campaign.datasets) > 1:
        for ind, d in enumerate(campaign.datasets[1:]):
            if isinstance(d, type):
                d = d.load()
            aux_name = d.properties[aux]
            design_x = np.array(
                [entry.as_dict["attributes"][aux_name] for entry in d.data]
            )
            pred_y = np.array(epoch["design_space"]["predictions"][ind])
            std_y = np.array(epoch["design_space"]["std_devs"][ind])

            colour = [point_colour_2, point_colour_3][ind]

            name = f"Predicted {d.__class__.__name__}"

            for j, bin in enumerate(bins[:-1]):
                binds = np.where((design_x > bin) & (design_x < bin + bins[j + 1]))
                binned_y = pred_y[binds]
                binned_x = design_x[binds]
                binned_std = std_y[binds]

                top_bins = np.argsort(binned_y)[len(binds) - top :]

                fig.add_trace(
                    go.Scatter(
                        x=binned_x[top_bins],
                        y=binned_y[top_bins],
                        error_y=dict(
                            type="data",
                            array=binned_std[top_bins],
                            visible=True,
                            color=colour,
                            opacity=0.5,
                            line=dict(width=0.5),
                        ),
                        mode="markers",
                        marker=dict(
                            size=5,
                            symbol="circle",
                            opacity=0.5,
                            line=dict(width=1, color="DarkSlateGrey"),
                            color=colour,
                        ),
                        name=name,
                        legendgroup=name,
                        showlegend=(j == 0),
                    ),
                )

    if epoch.get("selected"):
        aux_name = aux
        target_x = np.array([entry[aux_name] for entry in epoch["selected"]])
        target_y = np.array([entry[target] for entry in epoch["selected"]])

        fig.add_trace(
            go.Scatter(
                x=target_x,
                y=target_y,
                mode="markers",
                marker=dict(
                    size=10,
                    symbol="star",
                    opacity=1,
                    line=dict(width=1, color="DarkSlateGrey"),
                    color=point_colour_4,
                ),
                name="Selected by AL",
                showlegend=True,
            ),
        )

    # Label axes
    fig.update_xaxes(title_text=f"{aux}")
    fig.update_yaxes(title_text=f"{target}")

    # Start both axes at 0
    fig.update_xaxes(range=[0, None], zeroline=False)
    fig.update_yaxes(range=[0, None], zeroline=False)

    if show:
        fig.show()

    return fig


def parity_plot(
    test_moddata: MODData,
    predictions: pd.DataFrame,
    uncertainties: pd.DataFrame,
    x: pd.DataFrame | None = None,
    selected_df: pd.DataFrame | None = None,
    title: str = "Parity plot",
    show: bool = True,
):
    """Make parity plots of model performance vs true values.

    Parameters:
        test_moddata: MODData object containing test data.
        predictions: DataFrame of model predictions.
        uncertainties: DataFrame of model uncertainties.
        x: Optional alternative x-axis to use for plotting.
        selected_df: Optional DataFrame of selected data points to highlight.
        title: Title of the plot.

    """
    colours = px.colors.qualitative.Plotly

    point_colour = colours[0]
    error_colour = colours[1]
    point_colour_2 = colours[2]
    point_colour_3 = colours[3]

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
        title=title,
    )

    if x is None:
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

        if selected_df:
            fig.add_trace(
                go.Scatter(
                    x=selected_df[target],
                    y=predictions.loc[selected_df.index, target],
                    mode="markers",
                    marker=dict(
                        size=5,
                        symbol="circle",
                        opacity=1,
                        line=dict(width=1, color="DarkSlateGrey"),
                        color=point_colour_3,
                    ),
                    showlegend=False,
                    hovertext=selected_df.index,
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

    else:
        fig.add_trace(
            go.Scatter(
                x=x[x.columns[0]],
                y=predictions[target],
                mode="markers",
                marker=dict(
                    size=5,
                    symbol="circle",
                    opacity=0.2,
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
            title_text=f'{list(x.columns)[0].replace("_", " ").title()}', row=1, col=1
        )
        fig.update_yaxes(
            title_text=f'Predicted {target.replace("_", " ").title()}', row=1, col=1
        )

        # Start both axes at 0
        fig.update_yaxes(range=[0, None], zeroline=False, row=1, col=1)

    errors = test_moddata.df_targets[target] - predictions[target]
    mae = errors.abs().mean()
    error_std = errors.abs().std()

    if x is None:
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
            title_text=f'{target.replace("_", " ").title()} prediction error',
            row=2,
            col=1,
        )

        max_uncertainty = uncertainties[target].max()

        fig.update_yaxes(
            range=[errors.min() - max_uncertainty, errors.max() + max_uncertainty],
            zeroline=False,
            row=2,
            col=1,
        )

    else:
        fom = x[x.columns[0]] * test_moddata.df_targets[target]
        fig.add_trace(
            go.Scatter(
                x=fom,
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

        fig.update_xaxes(title_text="True FOM", row=2, col=1)
        fig.update_yaxes(
            title_text=f'{target.replace("_", " ").title()} prediction error',
            row=2,
            col=1,
        )

        max_uncertainty = uncertainties[target].max()

        fig.update_yaxes(
            range=[errors.min() - max_uncertainty, errors.max() + max_uncertainty],
            zeroline=False,
            row=2,
            col=1,
        )

    fig.update_yaxes(
        range=[errors.min() - max_uncertainty, errors.max() + max_uncertainty],
        zeroline=False,
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            y=errors, showlegend=False, marker_color=error_colour, hoverinfo="none"
        ),
        row=2,
        col=2,
    )

    fig.add_annotation(
        x=10,
        y=errors.max(),
        xanchor="left",
        text=f"MAE: {mae:.4f}±{error_std:.4f}",
        showarrow=False,
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Frequency", row=2, col=2)
    fig.update_yaxes(
        title_text=f'{target.replace("_", " ").title()} prediction error', row=2, col=2
    )

    if show:
        fig.show()

    return fig


def compare_error_distributions(test_moddata, predictions_a, predictions_b):
    fig = go.Figure()
    fig.update_layout(
        width=1000,
        height=800,
    )

    mae_a = (
        (
            test_moddata.df_targets[test_moddata.df_targets.columns[0]]
            - predictions_a[test_moddata.df_targets.columns[0]]
        )
        .abs()
        .mean()
    )
    fig.add_trace(
        go.Histogram(
            x=test_moddata.df_targets[test_moddata.df_targets.columns[0]]
            - predictions_a[test_moddata.df_targets.columns[0]],
            name=f"Default MAE: {mae_a:.4f}",
            hovertext=mae_a,
        )
    )
    mae_b = (
        (
            test_moddata.df_targets[test_moddata.df_targets.columns[0]]
            - predictions_b[test_moddata.df_targets.columns[0]]
        )
        .abs()
        .mean()
    )
    fig.add_trace(
        go.Histogram(
            x=test_moddata.df_targets[test_moddata.df_targets.columns[0]]
            - predictions_b[test_moddata.df_targets.columns[0]],
            name=f"Hyperopt MAE: {mae_b:.4f}",
            hovertext=mae_b,
        )
    )

    fig.show()
