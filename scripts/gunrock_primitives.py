#!/usr/bin/env python3

import altair as alt
import pandas  # http://pandas.pydata.org
import numpy
import datetime

from fileops import save, getChartHTML
from filters import *
from logic import *

alt.data_transformers.disable_max_rows()

name = "gunrock_primitives"

# begin user settings for this script
roots = [
    "../gunrock-output/v1-0-0/sssp",
    "../gunrock-output/v1-0-0/bc",
    "../gunrock-output/v1-0-0/tc",
    "../gunrock-output/v1-0-0/pr",
    "../gunrock-output/v1-0-0/bfs",
    "../gunrock-output/launch_bounds_comparison",
    "../gunrock-output/cuda_arch_comparison",
]
fnFilterInputFiles = [
    fileEndsWithJSON,
]
fnPreprocessDF = [
    # convertCtimeStringToDatetime,
    # normalizePRMTEPS,
    mergeAlgorithmIntoPrimitive,
    lowercasePrimitives,
    renameAdvanceModeWithAHyphen,
    addJSONDetailsLink,
    gunrockVersionGPU,
]
fnFilterDFRows = [
    filterOut(True, "64bit-SizeT"),
    filterOut(True, "64bit-VertexT"),
]
fnPostprocessDF = [
    equateNVIDIAGPUs,
    BFStoDOBFS10,
    copyQueuedToVisitedForPR,
    computeMTEPSFromEdgesAndElapsed10,
    mergeMaxIterationIntoMaxIter,
    normalizePRByIterations,
    renameColumnsWithMinus,
    undirectedAndMarkPred,
    # get rid of all tc + directed
    lambda df: df[(df["primitive"] != "tc") | (df["undirected"] == True)],
]
# end user settings for this script

# actual program logic
# do not modify

# choose input files
df = filesToDF(roots=roots, fnFilterInputFiles=fnFilterInputFiles)
for fn in fnPreprocessDF:  # alter entries / compute new entries
    df = fn(df)
for fn in fnFilterDFRows:  # remove rows
    df = fn(df)
for fn in fnPostprocessDF:  # alter entries / compute new entries
    df = fn(df)

# end actual program logic

columnsOfInterest = [
    "primitive",
    "dataset",
    "avg_mteps",
    "avg_process_time",
    "engine",
    # 'tag',
    "num_vertices",
    "num_edges",
    "nodes_visited",
    "edges_visited",
    "search_depth",
    "gunrock_version",
    "gpuinfo_name",
    "gpuinfo_name_full",
    "advance_mode",
    "undirected",
    "mark_pred",
    "undirected_markpred",
    "idempotence",
    "pull",
    "64bit_SizeT",
    "64bit_VertexT",
    "time",
    "details",
]
# would prefer a cleanup call https://github.com/altair_viz/altair/issues/183
# without this, output is gigantic
df = (keepTheseColumnsOnly(columnsOfInterest))(df)

# now make the graph

prim_fullname = {
    "bfs": "Forward-Only BFS",
    "dobfs": "Direction-Optimized BFS",
    "sssp": "SSSP",
    "tc": "Triangle Counting",
    "bc": "Betweenness Centrality",
    "pr": "PageRank",
}

datatypes = {
    "advance_mode": "nominal",
    "avg_mteps": "quantitative",
    "avg_process_time": "quantitative",
    "dataset": "nominal",
    "edges_visited": "quantitative",
    "gpuinfo_name": "nominal",
    "gpuinfo_name_full": "nominal",
    "gunrock_version": "nominal",
    "idempotence": "nominal",
    "mark_pred": "nominal",
    "max(avg_mteps)": "quantitative",
    "min(avg_process_time)": "quantitative",
    "nodes_visited": "quantitative",
    "num_edges": "quantitative",
    "num_vertices": "quantitative",
    "primitive": "nominal",
    "pull": "nominal",
    "search_depth": "quantitative",
    "undirected": "nominal",
    "undirected_markpred": "nominal",
}

chart = {}

my = {}

# set up all the plots; each entry in "my" becomes a plot

for prim in ["bfs", "dobfs", "sssp", "tc", "bc", "pr"]:
    my[(prim, "mteps_best")] = {
        "mark": "point",
        "x": ("dataset", "Dataset", "linear"),
        "y": ("avg_mteps", "MTEPS", "log"),
        "y_aggregate": max,
        "color": ("gpuinfo_name", "GPU"),
        "shape": ("gpuinfo_name", "GPU"),
        # "prim=prim" forces "prim" to bind to the primitive in the above loop
        # otherwise it binds when it's called, that's bad
        "filter": lambda df, prim=prim: df[df["primitive"] == prim],
        "title": f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs, combining all options (measured in MTEPS)",
    }
    my[(prim, "mteps")] = my[(prim, "mteps_best")].copy()
    my[(prim, "mteps")]["row"] = ("undirected", "Undirected")
    my[(prim, "mteps")][
        "title"
    ] = f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs, separating out options (measured in MTEPS)"

    if prim == "sssp" or prim == "bfs" or prim == "dobfs":
        # for SSSP/BFS/DOBFS, mark_pred is significant, but not for the others
        my[(prim, "mteps")]["row"] = (
            "undirected_markpred",
            "Undirected / Mark Predecessors",
        )

    # avg_process_time is identical except pick the min
    my[(prim, "avg_process_time")] = my[(prim, "mteps")].copy()
    my[(prim, "avg_process_time_best")] = my[(prim, "mteps_best")].copy()
    for plot in ["avg_process_time", "avg_process_time_best"]:
        my[(prim, plot)]["y"] = (
            # pr has a normalized runtime per iteration (already computed),
            # but fix the caption
            "avg_process_time",
            "Per_iteration runtime (ms)" if prim == "pr" else "Runtime (ms)",
            "log",
        )
        my[(prim, plot)]["y_aggregate"] = min
    my[(prim, "avg_process_time")][
        "title"
    ] = f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs, separating out options (measured in ms)"
    my[(prim, "avg_process_time_best")][
        "title"
    ] = f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs, combining all options (measured in ms)"

    if prim == "tc":
        del my[(prim, "mteps")]  # we don't have MTEPS for tc
        del my[(prim, "mteps_best")]  # we don't have MTEPS for tc
        del my[(prim, "avg_process_time")]["row"]  # only undirected

    my[(prim, "advance_mode")] = {
        "mark": "point",
        "x": ("dataset", "Dataset", "linear"),
        "y": ("avg_mteps", "MTEPS", "log"),
        "y_aggregate": max,
        "row": ("undirected", "Undirected"),
        "color": ("advance_mode", "Advance Mode"),
        "shape": ("advance_mode", "Advance Mode"),
        "filter": lambda df, prim=prim: df[
            (df["primitive"] == prim) & (df["gpuinfo_name"] == "Tesla V100")
        ],
        "title": f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs, per advance mode, measured on V100",
    }
    if prim == "sssp" or prim == "bfs" or prim == "dobfs":
        my[(prim, "advance_mode")]["row"] = (
            "undirected_markpred",
            "Undirected / Mark Predecessors",
        )
    if prim == "tc":
        del my[(prim, "advance_mode")]  # we don't have MTEPS for tc

    my[(prim, "edges")] = my[(prim, "avg_process_time_best")].copy()
    my[(prim, "edges")]["x"] = ("num_edges", "Number of Edges", "log")
    my[(prim, "edges")]["title"] = f"{prim_fullname[prim]}: Runtime vs. Number of Edges"

# now do some pr-specific plots
prim = "pr"
my[("pr", "push_pull_mteps")] = my[("pr", "mteps")].copy()
my[("pr", "push_pull_mteps")]["shape"] = ("pull", "Pull")
my[(prim, "push_pull_mteps")][
    "title"
] = f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs for push vs. pull, separating out options (measured in MTEPS)"
my[("pr", "push_pull_avg_process_time")] = my[("pr", "avg_process_time")].copy()
my[("pr", "push_pull_avg_process_time")]["shape"] = ("pull", "Pull")
my[(prim, "push_pull_avg_process_time")][
    "title"
] = f"{prim_fullname[prim]}: Fastest Gunrock 1.0+ runs for push vs. pull, separating out options (measured in ms)"
my[("pr", "push_pull_search_depth")] = {
    "mark": "point",
    "x": ("dataset", "Dataset", "linear"),
    "y": ("search_depth", "Search Depth", "log"),
    "row": ("undirected", "Undirected"),
    "color": ("pull", "Pull"),
    "shape": ("pull", "Pull"),
    "filter": lambda df, prim=prim: df[
        (df["primitive"] == prim) & (df["gpuinfo_name"] == "Tesla V100")
    ],
    "title": f"{prim_fullname[prim]}: Search depth for push vs. pull, separating out options, measured on V100",
}
my[("pr", "push_pull_edges_visited")] = my[("pr", "push_pull_search_depth")].copy()
my[("pr", "push_pull_edges_visited")]["y"] = (
    "edges_visited",
    "Number of Edges Visited/Queued",
    "log",
)
my[("pr", "push_pull_edges_visited")][
    "title"
] = f"{prim_fullname[prim]}: Edges Visited/Queued for push vs. pull, separating out options, measured on V100"
my[("pr", "push_pull_vertices_visited")] = my[("pr", "push_pull_search_depth")].copy()
my[("pr", "push_pull_vertices_visited")]["y"] = (
    "nodes_visited",
    "Number of Vertices Visited/Queued",
    "log",
)
my[("pr", "push_pull_vertices_visited")][
    "title"
] = f"{prim_fullname[prim]}: Vertices Visited/Queued for push vs. pull, separating out options, measured on V100"
#  done with pr-specific plots


my[("all_V100", "edges_visited_vs_num_edges")] = {
    "mark": "point",
    "x": ("num_edges", "Number of Edges", "log"),
    "y": ("edges_visited", "Number of Edges Visited/Queued", "log"),
    "row": ("undirected", "Undirected"),
    "color": ("primitive", "Primitive"),
    "shape": ("primitive", "Primitive"),
    "filter": lambda df: df[
        (df["edges_visited"] > 0) & (df["gpuinfo_name"] == "Tesla V100")
    ],
    "title": "Edges Visited vs. Number of Edges, measured on V100",
}

my[("all_V100", "vertices_visited_vs_num_vertices")] = {
    "mark": "point",
    "x": ("num_vertices", "Number of Vertices", "log"),
    "y": ("nodes_visited", "Number of Vertices Visited/Queued", "log"),
    "row": ("undirected", "Undirected"),
    "color": ("primitive", "Primitive"),
    "shape": ("primitive", "Primitive"),
    "filter": lambda df: df[
        (df["nodes_visited"] > 0) & (df["gpuinfo_name"] == "Tesla V100")
    ],
    "title": "Vertices Visited vs. Number of Vertices, measured on V100",
}

my[("all_V100", "search_depth")] = {
    "mark": "point",
    "x": ("search_depth", "Search Depth", "log"),
    "y": ("avg_process_time", "Runtime (ms)", "log"),
    "row": ("undirected", "Undirected"),
    "color": ("primitive", "Primitive"),
    "shape": ("primitive", "Primitive"),
    "filter": lambda df: df[
        (df["search_depth"] > 0) & (df["gpuinfo_name"] == "Tesla V100")
    ],
    "title": "Runtime vs. Search Depth, measured on V100",
}

for plot in my.keys():
    print(f"*** Processing {plot} ***")

    primitive = plot[0]
    if "filter" in my[plot]:
        dfx = my[plot]["filter"](df)
    else:
        dfx = df

    # filter the dataframe to only contain the aggregate (do the aggregation in
    # Pandas not Altair, otherwise the resulting dataframe table is huge)
    # y_aggregate is simply the aggregation function (e.g., min, max)
    if "y_aggregate" in my[plot]:
        # what are our groupbys? They're our encodings in the plot.
        columns = [
            my[plot][field][0]
            for field in ["x", "row", "col", "color", "shape"]
            if field in my[plot]
        ]
        # keep only those rows that match the aggregate
        idx = (
            dfx.groupby(columns)[my[plot]["y"][0]].transform(my[plot]["y_aggregate"])
            == dfx[my[plot]["y"][0]]
        )
        dfx = dfx[idx]

    selection = {}

    # we assume that the only aggregate is max or min in the y channel
    def generateTooltip2(field, y):
        mmin = re.match(r"min\((.*)\)$", y)
        mmax = re.match(r"max\((.*)\)$", y)
        if mmin:
            return alt.Tooltip(
                field=field,
                type=datatypes[field],
                aggregate=alt.Aggregate(alt.ArgminDef(argmin=mmin.group(1))),
            )
        elif mmax:
            return alt.Tooltip(
                field=field,
                type=datatypes[field],
                aggregate=alt.Aggregate(alt.ArgmaxDef(argmax=mmax.group(1))),
            )
        else:
            return alt.Tooltip(field=field, type=datatypes[field])

    # this is currying. will it bind correctly?
    def generateTooltip(field):
        return generateTooltip2(field=field, y=my[plot]["y"][0])

    tooltip = [
        "primitive",
        "dataset",
        generateTooltip("avg_mteps"),
        generateTooltip("avg_process_time"),
        generateTooltip("advance_mode"),
        generateTooltip("gpuinfo_name"),
        generateTooltip("gpuinfo_name_full"),
        generateTooltip("gunrock_version"),
        generateTooltip("num_vertices"),
        generateTooltip("nodes_visited"),
        generateTooltip("num_edges"),
        generateTooltip("edges_visited"),
        generateTooltip("search_depth"),
        generateTooltip("undirected"),
        generateTooltip("mark_pred"),
        generateTooltip("idempotence"),
        "64bit_SizeT",
        "64bit_VertexT",
    ]

    # Altair
    # tooltip=[alt.Tooltip(c, type='quantitative') for c in columns]
    # Vega_Lite
    # {
    #   "type": "nominal",
    #   "field": "Fighting Style",
    #   "aggregate": {"argmin": "Place"}
    # }
    #
    # so:
    #
    # alt.Tooltip(field, alt.aggregate(alt.ArgmaxDef(argmax=argmaxfield?)))

    chart[plot] = (
        alt.Chart(dfx, mark=my[plot]["mark"])
        .encode(
            x=alt.X(
                my[plot]["x"][0],
                type=datatypes[my[plot]["x"][0]],
                axis=alt.Axis(title=my[plot]["x"][1],),
                scale=alt.Scale(type=my[plot]["x"][2]),
            ),
            y=alt.Y(
                my[plot]["y"][0],
                type=datatypes[my[plot]["y"][0]],
                # aggregate=my[plot].get("y_aggregate", alt.Undefined),
                axis=alt.Axis(title=my[plot]["y"][1],),
                scale=alt.Scale(type=my[plot]["y"][2]),
            ),
        )
        .interactive()
    )

    if "col" in my[plot]:
        chart[plot] = chart[plot].encode(
            column=alt.Column(
                my[plot]["col"][0],
                type=datatypes[my[plot]["col"][0]],
                header=alt.Header(title=my[plot]["col"][1]),
            )
        )

    if "row" in my[plot]:
        chart[plot] = chart[plot].encode(
            row=alt.Row(
                my[plot]["row"][0],
                type=datatypes[my[plot]["row"][0]],
                header=alt.Header(title=my[plot]["row"][1]),
            )
        )

    if "color" in my[plot]:
        color = stripShorthand(my[plot]["color"][0])
        selection["color"] = alt.selection_multi(fields=[color], bind="legend")
        chart[plot] = (
            chart[plot]
            .encode(
                color=alt.Color(
                    color,
                    type=datatypes[color],
                    legend=alt.Legend(title=my[plot]["color"][1]),
                    # scale=alt.Scale(scheme="dark2"),
                ),
            )
            .add_selection(selection["color"])
        )

    if "shape" in my[plot]:
        shape = stripShorthand(my[plot]["shape"][0])
        selection["shape"] = alt.selection_multi(fields=[shape], bind="legend")
        chart[plot] = (
            chart[plot]
            .encode(
                shape=alt.Shape(
                    shape,
                    type=datatypes[shape],
                    legend=alt.Legend(title=my[plot]["shape"][1]),
                ),
            )
            .add_selection(selection["shape"])
        )

    # https://github.com/altair-viz/altair/issues/1890
    # how to handle both color and shape selection
    # won't work properly until https://github.com/vega/vega-lite/issues/5553
    #   is fixed
    if ("color" in my[plot]) and ("shape" in my[plot]):
        chart[plot] = chart[plot].encode(
            opacity=alt.condition(
                selection["color"] & selection["shape"], alt.value(1), alt.value(0.2)
            )
        )
    elif "color" in my[plot]:
        chart[plot] = chart[plot].encode(
            opacity=alt.condition(selection["color"], alt.value(1), alt.value(0.2))
        )
    elif "shape" in my[plot]:
        chart[plot] = chart[plot].encode(
            opacity=alt.condition(selection["shape"], alt.value(1), alt.value(0.2))
        )

    if "title" in my[plot]:
        chart[plot] = chart[plot].properties(title=my[plot]["title"])

    chart[plot] = chart[plot].encode(tooltip=list(tooltip))

    plotname = "_".join(filter(lambda x: bool(x), [name, plot[0], plot[1]]))
    save(
        chart=chart[plot],
        df=dfx,
        plotname=plotname,
        outputdir="../plots",
        formats=["json", "tablehtml", "tablemd", "md", "html", "png", "pdf"],
        sortby=[
            "primitive",
            "dataset",
            "engine",
            "gunrock_version",
            "advance_mode",
            "undirected",
            "mark_pred",
            "idempotence",
        ],
        columns=columnsOfInterest,
        mdtext=(
            """
         # Data for %s

         """
            % plot[0]
            + getChartHTML(chart[plot], anchor=plotname)
            + """
                 [Source data](tables/%s_table.html), with links to the output JSON for each run
                 """
            % plotname
        ),
    )
