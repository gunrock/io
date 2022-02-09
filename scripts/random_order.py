#!/usr/bin/env python3

from math import ceil, floor
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.io import mminfo, mmread, mmwrite

import sys, getopt

def random_order(gr):
    rperm = np.random.permutation(gr.shape[0])
    return rperm

def permute_graph(permutation, grph):
 return grph[permutation,:][:,permutation]


def main(argv):
    inputFile = ""
    outputFile = ""
    permFile = ""
    visFile = ""
    useTest = False
    debug = False
    symmetry = None

    try:
        opts, args = getopt.getopt(argv, "dhi:o:p:s:tv:")
    except getopt.GetoptError:
        print(
            "{argv[0]} [-d] [-t] [-i <inputfile>] [-s <symmetry>] [-o <outputfile>] [-v <visfile>] [-p <permfile>]"
        )
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "{argv[0]} [-d] [-t] [-i <inputfile>] [-s <symmetry>] [-o <outputfile>] [-v <visfile>] [-p <permfile>]"
            )
            sys.exit()
        elif opt == "-d":
            debug = True
        elif opt == "-t":
            useTest = True
        elif opt == "-s":
            symmetry = arg
        elif opt in ("-i", "--ifile"):
            inputFile = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg
        elif opt in ("-v", "--visfile"):
            visFile = arg
        elif opt in ("-p", "--permfile"):
            permFile = arg

    if useTest:
        csrMatrix = csr_matrix(
            [
                [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            ]
        )
      #I believe the answer is 
      #[[0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
      #[0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
      #[0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
      #[0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
      #[0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
      #[1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
      #[0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
      #[1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
      #[0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
      #[0, 0, 0, 1, 0, 0, 1, 1, 0, 0]]

        #  it's the same graph after min-degree perfect elim
        # Note it's symmetric, and mmwrite will output only the lower triangle
        cooMatrix = csrMatrix.tocoo()
    else:
        if not inputFile:
            inputFile = sys.stdin

        cooMatrix = mmread(inputFile)  # returns coo_matrix
        csrMatrix = csr_matrix(cooMatrix, dtype=np.int32)

    if visFile:
        (dim, dimtemp) = cooMatrix.shape
        assert dim == dimtemp
        downscale = dim // 1024
        if useTest:
            downscale = 2
        visdim = ceil(float(dim) / float(downscale))
        # create a new array
        visarray = np.ndarray(shape=(visdim, visdim), dtype="int")  # nnz
        imarray = np.ndarray(shape=(visdim, visdim), dtype="uint8")  # int [0,255]
        for i, j, v in zip(cooMatrix.row, cooMatrix.col, cooMatrix.data):
            visarray[i // downscale, j // downscale] += 1
        vismax = float(np.max(visarray))
        for x in range(visdim):
            for y in range(visdim):
                # max value gets black (0.0), min value gets white (255.0)
                imarray[x, y] = floor(255.0 * (1.0 - (float(visarray[x, y]) / vismax)))
        im = Image.fromarray(imarray)
        print(im.format, im.size, im.mode)
        im.save(visFile)

    perm = random_order(csrMatrix)
    if permFile:
        np.savetxt(permFile, perm, fmt="%u")

    if debug:
        print(f"Here's my input matrix, called csrMatrix:\n{csrMatrix.toarray()}")
        print(f"Then I call perm = random_order(csrMatrix)\nperm = {perm}")
        print(f"csrMatrix[perm, :][:, perm]:\n{csrMatrix[perm, :][:, perm].toarray()}")
        print(f"csrMatrix[perm][perm]:\n{csrMatrix[perm][perm].toarray()}")
        print(f"csrMatrix[perm, perm]:\n{csrMatrix[perm, perm]}")
    randomPermMatrix = csrMatrix[perm, :][:, perm]

    # now save it

    if outputFile:
        mmwrite(outputFile, randomPermMatrix, symmetry=symmetry)

if __name__ == "__main__":
    main(sys.argv[1:])
