import pandas  # http://pandas.pydata.org
import os  # built-in
import re  # built-in
import math  # built-in
import numpy  # built-in

# possible filtering functions


def fileEndsWithJSON(f):
    return (
        os.path.isfile(f)
        and (os.path.splitext(f)[1] == ".json")
        and not os.path.basename(f).startswith("_")
    )


def fileNotInArchiveDir(f):
    return os.path.isfile(f) and ("/archive/" not in f)


def convertCtimeStringToDate(df):
    # 'time' column is in (text) ctime format
    # datetime.strptime(jsonobj['time'], "%a %b %d %H:%M:%S %Y\n")
    # or
    # http://stackoverflow.com/questions/26763344/convert-pandas-column-to-datetime
    # normalize() resets the time to midnight (so it can be == vs. dates)
    df["time"] = df["time"].apply(
        lambda x: pandas.to_datetime(x, infer_datetime_format=True).normalize()
    )
    return df


def convertCtimeStringToDatetime(df):
    df["time"] = df["time"].apply(
        lambda x: pandas.to_datetime(x, infer_datetime_format=True)
    )
    return df


def tupleify(tag):
    def fn(df):
        if tag in df.columns:
            pandas.set_option("display.max_rows", None)
            df[tag] = df[tag].apply(tuple)
        return df

    return fn


def DOBFStoBFS(df):
    df.loc[df.algorithm == "DOBFS", "algorithm"] = "BFS"
    return df


def BFStoDOBFS(df):
    m = (df.algorithm == "BFS") & (df.direction_optimized == True)
    df.loc[m, "algorithm"] = "DOBFS"
    return df


def BFStoDOBFS10(df):
    m = (df["primitive"] == "bfs") & (df["direction-optimized"] == True)
    df.loc[m, "primitive"] = "dobfs"
    return df


def BFSCCtoCC(df):
    df.loc[df.algorithm == "BFSCC", "algorithm"] = "CC"
    return df


def equateRGG(df):
    df.loc[df.dataset == "rgg_n_2_24_s0", "dataset"] = "rgg_n24_0.000548"
    return df


def SSSPtosssp(df):
    df.loc[df.primitive == "SSSP", "primitive"] = "sssp"
    return df


def lowercasePrimitives(df):
    df.loc[df.primitive == "SSSP", "primitive"] = "sssp"
    df.loc[df.primitive == "PR", "primitive"] = "pr"
    return df


def mergeAllUpperCasePrimitives(df):
    df.loc[df.primitive == "SSSP", "primitive"] = "sssp"
    df.loc[df.primitive == "BC", "primitive"] = "bc"
    df.loc[df.primitive == "CC", "primitive"] = "cc"
    df.loc[df.primitive == "BFS", "primitive"] = "bfs"
    df.loc[df.primitive == "DOBFS", "primitive"] = "bfs"
    df.loc[df.primitive == "PR", "primitive"] = "pr"
    df.loc[df.primitive == "PageRank", "primitive"] = "pr"
    df.loc[df.primitive == "TC", "primitive"] = "tc"
    df.loc[df.primitive == "RW", "primitive"] = "rw"
    return df


def replaceWith(src, dest, column):
    def fn(df):
        df.loc[df[column] == src, column] = dest
        return df

    return fn


def equateM40(df):
    df.loc[df["gpuinfo_name"] == "Tesla M40 24GB", "gpuinfo_name"] = "Tesla M40"
    return df


def equateNVIDIAGPUs(df):
    df["gpuinfo_name_full"] = df["gpuinfo_name"]
    df = equateM40(df)
    df.loc[df["gpuinfo_name"] == "Tesla K40c", "gpuinfo_name"] = "Tesla K40/80"
    df.loc[df["gpuinfo_name"] == "Tesla K40m", "gpuinfo_name"] = "Tesla K40/80"
    df.loc[df["gpuinfo_name"] == "Tesla K80", "gpuinfo_name"] = "Tesla K40/80"
    df.loc[df["gpuinfo_name"] == "m60", "gpuinfo_name"] = "Tesla M60"
    df.loc[df["gpuinfo_name"] == "p100", "gpuinfo_name"] = "Tesla P100-PCIE-16GB"
    df.loc[df["gpuinfo_name"] == "Quadro GV100", "gpuinfo_name"] = "Tesla V100"
    df.loc[df["gpuinfo_name"] == "Tesla V100-PCIE-16GB", "gpuinfo_name"] = "Tesla V100"
    df.loc[df["gpuinfo_name"] == "Tesla V100-PCIE-32GB", "gpuinfo_name"] = "Tesla V100"
    df.loc[df["gpuinfo_name"] == "Tesla V100-DGXS-16GB", "gpuinfo_name"] = "Tesla V100"
    df.loc[df["gpuinfo_name"] == "Tesla V100-DGXS-32GB", "gpuinfo_name"] = "Tesla V100"
    return df


def replaceFromDict(d, out_column, in_column):
    def fn(df):
        for key, value in d.iteritems():
            df.loc[df[in_column] == key, out_column] = value
        return df

    return fn


def normalizePRMTEPS(df):
    if "search_depth" in df.columns:
        df.loc[df.algorithm == "PageRank", "m_teps"] = df["m_teps"] * df["search_depth"]
    return df


def normalizePRByIterations(df):
    # run mergeMaxIterationIntoMaxIter first
    df.loc[df.primitive == "pr", "avg-process-time"] = (
        df["avg-process-time"] / df["max-iter"]
    )
    return df


def renameGpuinfoname(df):
    return df.rename(columns={"gpuinfo_name": "gpuinfo_name"})


def mergeIdempotentToIdempotence(df):
    return merge(df, dst="idempotence", src="idempotent", delete=True)


def mergePostprocessTimeUnderscoreIntoHyphen(df):
    return merge(df, dst="postprocess-time", src="postprocess_time", delete=True)


def merge(df, dst, src, delete=True):
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


def mergeAlgorithmIntoPrimitive(df):
    return merge(df, dst="primitive", src="algorithm", delete=True)


def mergeAlgorithmIntoEngine(df):
    return merge(df, dst="engine", src="algorithm", delete=True)


def mergeSNNElapsedIntoElapsed(df):
    return merge(df, dst="elapsed", src="snn-elapsed", delete=True)


def mergeMHyphenTEPSIntoAvgMTEPS(df):
    return merge(df, dst="avg-mteps", src="m-teps", delete=True)


def mergeElapsedIntoAvgProcessTime(df):
    return merge(df, dst="avg-process-time", src="elapsed", delete=True)


def mergeGunrockVersionWithUnderscoreIntoHyphen(df):
    return merge(df, dst="gunrock-version", src="gunrock_version", delete=True)


def mergeAdvanceModeWithUnderscoreIntoHyphen(df):
    return merge(df, dst="advance-mode", src="advance_mode", delete=True)


def mergeTraversalModeWithUnderscoreIntoAdvanceModeWithHyphen(df):
    return merge(df, dst="advance-mode", src="traversal_mode", delete=True)


def mergeTraversalModeWithUnderscoreIntoAdvanceModeWithUnderscore(df):
    return merge(df, dst="advance_mode", src="traversal_mode", delete=True)


def mergeMaxIterationIntoMaxIter(df):
    if "max_iteration" in df.columns:
        return merge(df, dst="max-iter", src="max_iteration", delete=True)
    else:
        return df


def mergeMinusSignsIntoUnderscores(df):
    for (dst, src) in [
        ("num_vertices", "num-vertices"),
        ("num_edges", "num-edges"),
        ("nodes_visited", "nodes-visited"),
        ("edges_visited", "edges-visited"),
        ("gunrock_version", "gunrock-version"),
        ("64bit_SizeT", "64bit-SizeT"),
        ("advance_mode", "advance-mode"),
        ("mark_pred", "mark-pred"),
        ("search_depth", "search-depth"),
    ]:
        if (src in df.columns) and (dst in df.columns):
            df = merge(df, dst=dst, src=src, delete=True)
    return df


def renameMTEPSToAvgMTEPS(df):
    return df.rename(columns={"m_teps": "avg-mteps"})


def renameGunrockVersionWithAHyphen(df):
    return df.rename(columns={"gunrock-version": "gunrock_version"})


def renameAdvanceModeWithAHyphen(df):
    return df.rename(columns={"advance-mode": "advance_mode"})


def mergeMarkPredecessors(df):
    # first, if there's no mark_pred, create it
    if "mark_pred" not in df.columns:
        if "mark-pred" in df.columns:
            df = df.rename(columns={"mark-pred": "mark_pred"})
        elif "mark_predecessors" in df.columns:
            df = df.rename(columns={"mark_predecessors": "mark_pred"})
        else:
            return df

    # now we definitely have a mark_pred
    for col in ["mark_predecessors", "mark-pred"]:
        if col in df.columns:
            df = merge(df, dst="mark_pred", src=col, delete=True)
    return df


def gunrockVersionGPU(df):
    if {"gunrock_version", "gpuinfo_name"}.issubset(df.columns):
        df["gunrock_version_gpu"] = df["gunrock_version"] + " / " + df["gpuinfo_name"]
    return df


def tagPlus64(df):
    if {"tag", "64bit-SizeT", "64bit-VertexT", "64bit-ValueT"}.issubset(df.columns):
        df["tag_64"] = ""
        for c in ["64bit-SizeT", "64bit-VertexT", "64bit-ValueT"]:
            df["tag_64"] = df["tag_64"] + df[c].map(lambda s: str(s)[0])
        df["tag_64"] = df["tag_64"] + " / " + df["tag"].map(str)
    return df


def summarize64(df):
    if {"64bit-SizeT", "64bit-VertexT", "64bit-ValueT"}.issubset(df.columns):
        df["summarize64"] = ""
        for c in ["64bit-SizeT", "64bit-VertexT", "64bit-ValueT"]:
            df["summarize64"] = df["summarize64"] + df[c].map(lambda s: str(s)[0])
    return df


def algorithmDataset(df):
    if {"algorithm", "dataset"}.issubset(df.columns):
        df["algorithm_dataset"] = df["algorithm"] + " / " + df["dataset"]
    return df


def insertMissing(col, val):
    def fn(df):
        df[col] = df[col].fillna(val)
        return df

    return fn


def addJSONDetailsLink(df):
    df["details"] = df["details"].apply(
        lambda s: re.sub(
            r".*/([-\w]*)-output",
            r'<a href="https://github.com/gunrock/io/tree/master/\1-output',
            s,
        )
        + '">JSON output</a>'
    )
    return df


# @TODO: The below bunch of functions are all really the same function


def selectAnyOfTheseDates(dates):
    return lambda df: df[df["time"].isin(dates)]


def selectAnyOfThese(column, these):
    return lambda df: df[df[column].isin(these)]


def selectTag(tag):
    def fn(df):
        if "tag" in df.columns:
            return df[df["tag"] == tag]
        else:
            return df

    return fn


def selectTags(tagList):
    def fn(df):
        if "tag" in df.columns:
            return df[df["tag"].isin(tagList)]
        else:
            return df

    return fn


def deselectTag(tag):
    def fn(df):
        if "tag" in df.columns:
            return df[df["tag"] != tag]
        else:
            return df

    return fn


def filterOut(value, column):
    return lambda df: df[df[column] != value]


def selectOneDataset(dataset):
    return lambda df: df[df["dataset"] == dataset]


def undirectedOnly(df):
    return df[df["undirected"] == True]


def idempotentOnly(df):
    return df[df["idempotent"] == True]


def thirtyTwoBitOnly(df):
    return df[
        (df["64bit-SizeT"] == False)
        & (df["64bit-VertexT"] == False)
        & (df["64bit-ValueT"] == False)
    ]


def directionOptimizedOnly(df):
    return df[df["direction-optimized"] == True]


def undirectedAndIdempotenceAndMarkPred(df):
    markpred = "uninitialized"
    for col in ["mark-pred", "mark_pred"]:
        if col in df.columns:
            if markpred != "uninitialized":
                print("Duplicate markpred in undirectedAndIdempotenceAndMarkPred")
            markpred = col
    df["undirected_idempotence_markpred"] = df[
        ["undirected", "idempotence", markpred]
    ].apply(lambda x: " / ".join(x.astype(str)), axis=1)
    return df


def undirectedAndMarkPred(df):
    markpred = "uninitialized"
    for col in ["mark-pred", "mark_pred"]:
        if col in df.columns:
            if markpred != "uninitialized":
                print("Duplicate markpred in undirectedAndMarkPred")
            markpred = col
    df["undirected_markpred"] = df[["undirected", markpred]].apply(
        lambda x: " / ".join(x.astype(str)), axis=1
    )
    return df


def undirectedAndPull(df):
    df["undirected_pull"] = df[["undirected", "pull"]].apply(
        lambda x: " / ".join(x.astype(str)), axis=1
    )
    return df


def concatFields(name, fieldlist, abbrev=False):
    def fn(df):
        if abbrev == True:
            df[name] = df[fieldlist].apply(
                lambda x: " / ".join(x.astype(str).str[0]), axis=1
            )
        else:
            df[name] = df[fieldlist].apply(lambda x: " / ".join(x.astype(str)), axis=1)
        return df

    return fn


def collapseAdvanceMode(df):
    df["advance_mode"].apply(", ".join)
    return df


def computeOtherMTEPSFromGunrock(df):
    # if df['m_teps'] is NaN, but df['elapsed'] is there, use
    # Gunrock's edges_visited to compute m_teps
    #
    # formula: edges_visited / (elapsed * 1000.0f)
    df["algorithm_dataset"] = df["algorithm"] + "_" + df["dataset"]

    # series mapping {algorithm+dataset} to edges_visited
    # not quite clear why there's duplicates, so average edges_visited
    dfg = (
        df.loc[df["engine"] == "Gunrock"]
        .groupby(["algorithm_dataset"])
        .mean()["edges_visited"]
    )

    # fill in missing values for edges_visited, per algorithm_dataset
    df = df.set_index("algorithm_dataset")
    df["edges_visited"] = df["edges_visited"].fillna(value=dfg)
    df = df.reset_index()

    # now calculate m_teps if it's empty but edges_visited and elapsed are
    # valid
    m = df.edges_visited.notnull() & df.elapsed.notnull() & df.m_teps.isnull()
    df.loc[m, "m_teps"] = df["edges_visited"] / (df["elapsed"] * 1000.0)
    return df


def computeMTEPSFromEdgesAndElapsed(df):
    if not {"m_teps"}.issubset(df.columns):
        df["m_teps"] = numpy.nan
    m = df.edges_visited.notnull() & df.elapsed.notnull() & df.m_teps.isnull()
    df.loc[m, "m_teps"] = df["edges_visited"] / (df["elapsed"] * 1000.0)
    return df


def computeMTEPSFromEdgesAndElapsed10(df):
    m = (
        df["edges-visited"].notnull()
        & df["avg-process-time"].notnull()
        & (df["avg-mteps"] == 0)
    )
    df.loc[m, "avg-mteps"] = df["edges-visited"] / (df["avg-process-time"] * 1000.0)
    return df


def computeNewMTEPSFromProcessTimes(df):
    def averagePT(row):
        pt = row["process_times"]
        avg = sum(pt) / len(pt)
        pt0 = list(filter(lambda f: f > (0.2 * avg), pt))
        return sum(pt0) / len(pt0)

    df["process_times_avg"] = df.apply(averagePT, axis=1)
    # now recompute m_teps
    df["m_teps"] = df["edges_visited"] / (df["process_times_avg"] * 1000.0)
    return df


# @TODO: next two functions are actually the same function
def deleteZeroMTEPS(df):
    return df[df["avg-mteps"] != 0]


def deleteZeroElapsed(df):
    return df[df["elapsed"] != 0]


def deleteZero(column):
    def fn(df):
        df = df[df[column] != 0]
        return df

    return fn


def setLigraAlgorithmFromSubalgorithm(df):
    ligranoalg = df["engine"] == "Ligra" & df.algorithm.isnull()
    m = ligranoalg & df["subalgorithm"] == "bfs-bitvector"
    df.loc[m, "algorithm"] = "BFS"
    return df


def copyQueuedToVisitedForPR(df):
    df.loc[
        (df["primitive"] == "pr") & (df["edges-visited"] == 0), "edges-visited"
    ] = df["edges-queued"]
    df.loc[
        (df["primitive"] == "pr") & (df["nodes-visited"] == 0), "nodes-visited"
    ] = df["nodes-queued"]
    return df


# @TODO: next two functions are actually the same function


def keepLatest(columns, sortBy="time"):
    def fn(df):
        newest = df.groupby(columns)[sortBy].transform(max)
        df = df[df[sortBy] == newest]
        return df

    return fn


def keepFastest(columns, sortBy="m_teps"):
    def fn(df):
        fastest = df.groupby(columns)[sortBy].transform(max)
        df = df[df[sortBy] == fastest]
        return df

    return fn


def keepFastestAvgProcessTime(columns, sortBy="avg-process-time"):
    def fn(df):
        idx = df.groupby(columns)[sortBy].transform(min) == df[sortBy]
        return df[idx]

    return fn


def combineGunrock1Plus(df):
    # All Gunrock 1.0.*  -> 1.0+
    df.loc[df["gunrock_version"].str.startswith("1."), "gunrock_version"] = "1.0+"
    return df


def normalizeByGunrock(dest, quantityToNormalize, columnsToGroup):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    def fn(df):
        dfgunrock = df.loc[
            df["engine"] == "Gunrock", columnsToGroup + [quantityToNormalize]
        ]
        suffix = "_gunrock"
        dfmerge = pandas.merge(df, dfgunrock, on=columnsToGroup, suffixes=["", suffix])
        dfmerge[dest] = (
            dfmerge[quantityToNormalize] / dfmerge[quantityToNormalize + suffix]
        )
        return dfmerge

    return fn


def normalizeByTag(dest, tag, quantityToNormalize, columnsToGroup):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    def fn(df):
        # https://stackoverflow.com/questions/63311153/pandas-how-to-see-if-an-argument-string-is-in-a-list-of-strings-that-is-in-a
        df1 = df.loc[
            df["tag"].explode().eq(tag).loc[lambda x: x].index,
            columnsToGroup + [quantityToNormalize],
        ]
        suffix = "_ref"
        dfmerge = pandas.merge(df, df1, on=columnsToGroup, suffixes=["", suffix])
        dfmerge[dest] = (
            dfmerge[quantityToNormalize + suffix] / dfmerge[quantityToNormalize]
        )
        return dfmerge

    return fn


def normalizeBy1GPU(dest, quantityToNormalize, columnsToGroup):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    def fn(df):
        df1 = df.loc[df["num_gpus"] == 1, columnsToGroup + [quantityToNormalize]]
        suffix = "_1"
        dfmerge = pandas.merge(df, df1, on=columnsToGroup, suffixes=["", suffix])
        dfmerge[dest] = (
            dfmerge[quantityToNormalize + suffix] / dfmerge[quantityToNormalize]
        )
        return dfmerge

    return fn


def normalizeToGPU(dest, quantityToNormalize, columnsToGroup, gpu):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    def fn(df):
        dfgunrock = df.loc[
            df["gpuinfo_name"] == gpu, columnsToGroup + [quantityToNormalize]
        ]
        suffix = "_refgpu"
        dfmerge = pandas.merge(df, dfgunrock, on=columnsToGroup, suffixes=["", suffix])
        dfmerge[dest] = (
            dfmerge[quantityToNormalize + suffix] / dfmerge[quantityToNormalize]
        )
        return dfmerge

    return fn


def normalizeToGunrock1Plus(dest, quantityToNormalize, columnsToGroup):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    # somehow this filters out all non-1.0+ gunrock_version
    def fn(df):
        dfgunrock = df.loc[
            df["gunrock_version"] == "1.0+", columnsToGroup + [quantityToNormalize]
        ]
        suffix = "_10+"
        dfmerge = df.merge(dfgunrock, on=columnsToGroup, suffixes=["", suffix])
        # expected this is normalized time (reference is in numerator)
        dfmerge[dest] = (
            dfmerge[quantityToNormalize + suffix] / dfmerge[quantityToNormalize]
        )
        return dfmerge

    return fn


def formatColumn(out_column, in_column, string_format):
    # oddly, I was not able to figure out how to do this with a lambda
    def fn(df):
        # df['var3'] = pd.Series(["{0:.2f}%".format(val * 100) for val in
        # df['var3']], index = df.index)
        df[out_column] = df[in_column].map(string_format.format)
        # df[out_column] = pandas.Series(
        #     [string_format.format(f) for f in df[in_column]])
        return df

    return fn


def loclist_expand(df, loclist, sample, sampleMinimum):
    # http://stackoverflow.com/questions/38577737/pandas-unflatten-data-frame-with-columns-containing-array
    rows = []
    for idx, row in df.iterrows():
        sampleRow = sample
        vss = []
        for loc in loclist:
            vss.append(row.at[loc])  # pick out the array, put into vss
        # print "At index ", idx, " I see ", len(vss[0]), " elements in the
        # array"
        n = len(vss[0])
        if n < sampleMinimum:
            sampleRow = False
        nsqrt = math.floor(math.sqrt(n))
        for i, v in enumerate(vss[0]):  # how many elements in the list?
            if sampleRow and (i % nsqrt != 0):  # we're sampling
                continue
            new = row.copy()
            newIsValid = True
            for j, loc in enumerate(loclist):
                if vss[j][i] == -1:
                    newIsValid = False
                else:
                    new.at[loc] = vss[j][i]
            if newIsValid:
                rows.append(new)

    return pandas.DataFrame(rows)


def flattenArrays(loclist, sample=False, sampleMinimum=100):
    return lambda df: loclist_expand(
        df, loclist=loclist, sample=sample, sampleMinimum=sampleMinimum
    )


def recomputeMTEPSFromMax(df):
    # bet there's a cleaner way to do this
    m = max(df["edges_visited"])
    df["edges_visited"] = df["edges_visited"].apply(lambda x: m)
    df["m_teps"] = df["edges_visited"] / (df["elapsed"] * 1000.0)
    return df


def addInto(dest, src1, src2):
    def fn(df):
        df[dest] = df[src1] + df[src2]
        return df

    return fn


def roundSig(column, significant_figures=1):
    # http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    def roundSigFn(x, sig):
        return round(x, sig - int(math.floor(math.log10(x))) - 1)

    def fn(df):
        df[column] = df[column].apply(lambda x: roundSigFn(x, significant_figures))
        return df

    return fn


def keepTheseColumnsOnly(columns):
    return lambda df: df[columns]


def renameColumnsWithMinus(df):
    # df = df.rename(str.replace(pat="-", repl="_"), axis="columns")
    # this didn't work
    df = df.rename(lambda s: s.replace("-", "_"), axis="columns")
    return df


def extractCTAThreadsFromTag(df):
    # "cta_6_threads_1024"
    df["tag0"] = df["tag"].apply(pandas.Series)
    df["tag_cta"] = (
        df["tag0"].str.extract(pat="cta_([0-9]+)_threads_[0-9]+").astype(int)
    )
    df["tag_threads"] = (
        df["tag0"].str.extract(pat="cta_[0-9]+_threads_([0-9]+)").astype(int)
    )
    return df


def stripShorthand(str):
    if str[-2] == ":" and str[-1].isupper():
        str = str[:-2]
    if str[0] == "[" and str[-1] == "]":
        str = str[1:-1]
    return str
