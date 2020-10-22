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
roots = ["../gunrock-output/v1-0-0", "../gunrock-output/v0-5-1/cc"]
fnFilterInputFiles = [
    fileEndsWithJSON,
    fileNotInArchiveDir,
]

fnPreprocessDF = [
    mergeAlgorithmIntoPrimitive,
    lowercasePrimitives,
    mergeMTEPSToAvgMTEPS,
    lambda df: merge(df, dst="num-vertices", src="num_vertices", delete=True),
    lambda df: merge(df, dst="num-edges", src="num_edges", delete=True),
    mergeElapsedIntoAvgProcessTime,
    mergeGunrockVersionWithUnderscoreIntoHyphen,
    normalizePRByIterations,
    renameColumnsWithMinus,
    equateNVIDIAGPUs,
]
fnFilterDFRows = [
    undirectedOnly,
    selectAnyOfThese("primitive", ["pr", "bfs", "sssp", "tc", "cc"]),
    selectAnyOfThese("gpuinfo_name", ["TITAN V", "Tesla K40/80"]),
    # get rid of PR push
    lambda df: df[(df["primitive"] != "pr") | (df["pull"] == True)],
    # soc-ork soc-lj h09 i04 rmat-22 rmat-23 rmat-24 rgg road_usa
    selectAnyOfThese(
        "dataset",
        [
            "cit-Patents",
            "coAuthorsCiteseer",
            "coPapersDBLP",
            "com-orkut",
            "hollywood-2009",
            "indochina-2004",
            "rgg_n_2_24_s0",
            "rmat_n22_e64.000000",
            "rmat_n23_e32.000000",
            "rmat_n24_e16.000000",
            "road_central",
            "road_usa",
            "soc-LiveJournal1",
            "soc-orkut",
        ],
    ),
    keepFastestAvgProcessTime(
        [
            "primitive",
            "dataset",
            "undirected",
            "gpuinfo_name",
        ],
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
    formats=[
        "tablehtml",
        "tablemd",
    ],
    sortby=[
        "primitive",
        "dataset",
        "gpuinfo_name",
        "undirected",
        "gunrock_version",
    ],
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
    formats=[
        "tablehtml",
        "tablemd",
    ],
    sortby=[
        "primitive",
        "dataset",
        "gpuinfo_name",
    ],
    columns=columnsOfInterest,
)

# pandoc -i graphblast-pull-pr-200901_table.html -t gfm -o graphblast-pull-pr-200901_table.md
