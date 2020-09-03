#!/usr/bin/env python3

import altair as alt
import pandas  # http://pandas.pydata.org
import numpy
import datetime

from fileops import save, getChartHTML
from filters import *
from logic import *

name = "graphblast-pull-pr-200901"
# begin user settings for this script
roots = [
    "../gunrock-output/v1-0-0",
]
fnFilterInputFiles = [
    fileEndsWithJSON,
    fileNotInArchiveDir,
]


def renameAlgToPrim(df):
    return df.rename(columns={"algorithm": "primitive"})


fnPreprocessDF = [
    # this is 1.0+ only so we don't have to do a bunch of the normal filters
    lowercasePrimitives,
    normalizePRByIterations,
    renameColumnsWithMinus,
    equateNVIDIAGPUs,
]
fnFilterDFRows = [
    undirectedOnly,
    selectAnyOfThese("primitive", ["pr", "bfs", "sssp", "tc"]),
    selectAnyOfThese("gpuinfo_name", ["TITAN V", "Tesla K40/80"]),
    # get rid of PR push
    lambda df: df[(df["primitive"] != "pr") | (df["pull"] == True)],
    # 1.0 only
    lambda df: df[df["gunrock_version"].str.startswith("1.")],
    # soc-ork soc-lj h09 i04 rmat-22 rmat-23 rmat-24 rgg road_usa
    selectAnyOfThese(
        "dataset",
        [
            "rmat_n22_e64.000000",
            "soc-orkut",
            "indochina-2004",
            "rmat_n24_e16.000000",
            "rmat_n23_e32.000000",
            "rgg_n_2_24_s0",
            "road_usa",
            "hollywood-2009",
            "soc-LiveJournal1",
        ],
    ),
    keepFastestAvgProcessTime(
        ["primitive", "dataset", "undirected", "gpuinfo_name",],
        sortBy="avg_process_time",
    ),
]
fnPostprocessDF = [
    addJSONDetailsLink,
    gunrockVersionGPU,
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
    "postprocess_time",
    "engine",
    "gunrock_version",
    "gpuinfo_name",
    "pull",
    "num_vertices",
    "num_edges",
    "undirected",
    "time",
    "details",
]
# would prefer a cleanup call https://github.com/altair-viz/altair/issues/183
# without this, output is gigantic
df = (keepTheseColumnsOnly(columnsOfInterest))(df)

c = alt.Chart(df)
plotname = name

save(
    chart=c,
    df=df,
    plotname=plotname,
    formats=["tablehtml", "tablemd",],
    sortby=["primitive", "dataset", "gpuinfo_name", "undirected", "gunrock_version",],
    columns=columnsOfInterest,
)

columnsOfInterest = [
    "primitive",
    "dataset",
    "avg_process_time",
    "gpuinfo_name",
    "undirected",
    "details",
]

df = (keepTheseColumnsOnly(columnsOfInterest))(df)

df = df[df["undirected"] == True]

save(
    chart=c,
    df=df,
    plotname=plotname + "_abbrev",
    formats=["tablehtml", "tablemd",],
    sortby=["primitive", "dataset", "gpuinfo_name",],
    columns=columnsOfInterest,
)
