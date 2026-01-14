from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_sums(
    scan_dir: Path,
    out_file: Path = Path("summary"),
    show: bool = False,
    html: bool = False,
    image: bool = False,
):
    scan_dir = Path(scan_dir).absolute()
    data = {}
    for parent in scan_dir.iterdir():
        for path in parent.glob("*_summary.json"):
            name = parent.name
            with path.open() as f:
                loaded = json.load(f)
            scalars = {}
            for k, v in loaded.items():
                if isinstance(v, (float, int)):
                    scalars[k] = v
            if (
                "rouge" in loaded
                and "rougeL" in loaded["rouge"]
                and len(loaded["rouge"]["rougeL"]) == 3
            ):
                scalars["rougeL F1"] = loaded["rouge"]["rougeL"][-1]
            data[name] = scalars

    maxes = {}
    for d in data.values():
        for k, v in d.items():
            if k not in maxes or v > maxes[k]:
                maxes[k] = v

    df = pd.DataFrame(data).T.reset_index().rename(columns={"index": "model"})
    df_long = df.melt(id_vars="model", var_name="metric", value_name="score")

    fig = px.bar(
        df_long,
        x="model",
        y="score",
        color="model",
        facet_col="metric",
        facet_col_wrap=3,
        title="Model Performance",
        facet_col_spacing=0.08
    )
    fig.update_yaxes(matches=None, showgrid=True, showticklabels=True)

    if show:
        fig.show()
    if html:
        fig.write_html(out_file.with_suffix(".html"))
    if image:
        fig.write_image(out_file.with_suffix(".png"))


# plot_sums("./eval", show=True, image=True)
