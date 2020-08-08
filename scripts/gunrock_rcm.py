#!/usr/bin/env python3

import altair as alt
import pandas  # http://pandas.pydata.org
import numpy
import datetime

from fileops import save, getChartHTML
from filters import *
from logic import *

name = "gunrock_rcm"
prims = ["sssp"]

# begin user settings for this script
roots = [
    "../gunrock-output/RCM_SSSP_8_3_2020",
    "../gunrock-output/NORCM_SSSP_8_3_2020",
]
fnFilterInputFiles = [
    fileEndsWithJSON,
    fileNotInArchiveDir,
]
fnPreprocessDF = [
    # convertCtimeStringToDatetime,
    # normalizePRMTEPS,
    renameColumnsWithMinus,
    selectAnyOfThese("engine", ["Gunrock"]),  # only Gunrock
    selectAnyOfThese("mark_pred", [True, False]),
    selectAnyOfThese("undirected", [True]),
    mergeAllUpperCasePrimitives,
    selectAnyOfThese("primitive", prims),
    deleteZero("avg_process_time"),
    normalizeByTag(
        dest="speedup_vs_norcm",
        tag="no-rcm",
        quantityToNormalize="avg_process_time",
        columnsToGroup=["primitive", "dataset", "advance_mode", "mark_pred"],
    ),
    addJSONDetailsLink,
    gunrockVersionGPU,
]
fnFilterDFRows = [
    # filterOut(True, '64bit-SizeT'),
    # filterOut(True, '64bit-VertexT'),
]
fnPostprocessDF = []
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
    "speedup_vs_norcm",
    "engine",
    "gunrock_version",
    "gpuinfo_name",
    "tag",
    "num_vertices",
    "num_edges",
    "advance_mode",
    "undirected",
    "mark_pred",
    "64bit_SizeT",
    "64bit_ValueT",
    "64bit_VertexT",
    "time",
    "details",
]
# would prefer a cleanup call https://github.com/altair-viz/altair/issues/183
# without this, output is gigantic
df = (keepTheseColumnsOnly(columnsOfInterest))(df)

# now make the graph

chart = {}

my = {}
for prim in prims:
    pt = (prim, "best")
    my[pt] = {}
    my[pt]["x"] = ("dataset:N", "Dataset", "linear")
    my[pt]["y"] = ("avg_process_time:Q", "Runtime (ms)", "log")
    my[pt]["color"] = ("tag:N", "Tag")
    my[pt]["shape"] = ("tag:N", "Tag")
    my[pt]["row"] = ("mark_pred:N", "Mark Predecessors")
    my[pt]["col"] = ("advance_mode", "Advance Mode")

    if prim == "pr":
        del my[pt]["col"]

    pt_norm = (prim, "norm")
    my[pt_norm] = my[pt].copy()
    my[pt_norm]["y"] = ("speedup_vs_norcm:Q", "Speedup vs. no-rcm", "log")


for primtuple in my:
    primitive = primtuple[0]
    if primitive in prims:
        dfx = df[df["primitive"] == primitive]
    else:
        dfx = df

    selection = {}

    # https://github.com/altair-viz/altair/issues/291
    # how to alter Charts after they're created

    chart[primtuple] = (
        alt.Chart(dfx)
        .mark_point()
        .encode(
            x=alt.X(
                my[primtuple]["x"][0],
                axis=alt.Axis(title=my[primtuple]["x"][1],),
                scale=alt.Scale(type=my[primtuple]["x"][2]),
            ),
            y=alt.Y(
                my[primtuple]["y"][0],
                axis=alt.Axis(title=my[primtuple]["y"][1],),
                scale=alt.Scale(type=my[primtuple]["y"][2]),
            ),
            tooltip=[
                "primitive",
                "dataset:N",
                "gpuinfo_name:N",
                "gunrock_version",
                "tag",
                "num_vertices",
                "num_edges",
                "advance_mode:N",
                "mark_pred",
                "undirected",
                "64bit_SizeT",
                "64bit_VertexT",
                "64bit_ValueT",
                "avg_mteps:Q",
                "avg_process_time:Q",
                "speedup_vs_norcm:Q",
                "details",
            ],
        )
        .interactive()
    )
    if "col" in my[primtuple]:
        chart[primtuple] = chart[primtuple].encode(
            column=alt.Column(
                my[primtuple]["col"][0],
                header=alt.Header(title=my[primtuple]["col"][1]),
            )
        )
    if "row" in my[primtuple]:
        chart[primtuple] = chart[primtuple].encode(
            row=alt.Row(
                my[primtuple]["row"][0],
                header=alt.Header(title=my[primtuple]["row"][1]),
            )
        )
    if "color" in my[primtuple]:
        chart[primtuple] = chart[primtuple].encode(
            color=alt.Color(
                my[primtuple]["color"][0],
                legend=alt.Legend(title=my[primtuple]["color"][1]),
                scale=alt.Scale(scheme="dark2"),
            )
        )
        selection["color"] = alt.selection_multi(
            fields=[my[primtuple]["color"][0]], bind="legend"
        )
        chart[primtuple] = chart[primtuple].add_selection(selection["color"])

    if "shape" in my[primtuple]:
        chart[primtuple] = chart[primtuple].encode(
            shape=alt.Shape(
                my[primtuple]["shape"][0],
                legend=alt.Legend(title=my[primtuple]["shape"][1]),
            )
        )
        selection["shape"] = alt.selection_multi(
            fields=[my[primtuple]["shape"][0]], bind="legend"
        )
        chart[primtuple] = chart[primtuple].add_selection(selection["shape"])

    if primtuple[1] == "sel":
        # input_checkbox = alt.binding_checkbox()
        # checkbox_selection = alt.selection_single(bind=input_checkbox, name="Big Budget Films")
        # size_checkbox_condition = alt.condition(checkbox_selection,
        #                                         alt.SizeValue(25),
        #                                         alt.Size('Hundred_Million_Production:Q')
        # )
        checkbox = {}
        checkbox_selection = {}
        for box in ["undirected"]:
            for l in [box + "_on", box + "off"]:
                checkbox_selection[l] = alt.selection_single(
                    bind=alt.binding_checkbox(), name=l
                )
                chart[primtuple] = chart[primtuple].add_selection(checkbox_selection[l])

    plotname = "_".join(filter(lambda x: bool(x), [name, primtuple[0], primtuple[1]]))
    save(
        chart=chart[primtuple],
        df=dfx,
        plotname=plotname,
        formats=["tablehtml", "tablemd", "md", "html", "png", "pdf"],
        sortby=[
            "primitive",
            "dataset",
            "gpuinfo_name",
            "gunrock_version",
            "advance_mode",
            "undirected",
            "mark_pred",
        ],
        columns=columnsOfInterest,
        mdtext=(
            """
         # Data for %s

         """
            % primtuple[0]
            + getChartHTML(chart[primtuple], anchor=plotname)
            + """
                 [Source data](tables/%s_table.html), with links to the output JSON for each run
                 """
            % plotname
        ),
    )
