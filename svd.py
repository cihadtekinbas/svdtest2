from pyspark import SparkContext
# $example on$
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
# $example off$


from scipy import sparse
from scipy import sparse
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import hstack
from scipy.sparse import vstack

if __name__ == "__main__":
    sc = SparkContext(appName="PythonSVDExample")

    # $example on$
    rows = sc.parallelize([
        Vectors.sparse(5, {1: 1.0, 3: 7.0}),
        Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
        Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ])

    mat = RowMatrix(rows)

    # Compute the top 5 singular values and corresponding singular vectors.
    svd = mat.computeSVD(5, computeU=True)
    U = svd.U       # The U factor is a RowMatrix.
    s = svd.s       # The singular values are stored in a local dense vector.
    V = svd.V       # The V factor is a local dense matrix.
    # $example off$
    collected = U.rows.collect()
    print("U factor is:")
    for vector in collected:
        print(vector)
    print("Singular values are: %s" % s)
    print("V factor is:\n%s" % V)
    

    for i in range(0,5):
        print(V[i,i])

    f=open("recommend","w")
    for x in recommend:
        ind = np.argpartition(recommend[x], -10)[-10:]
        f.write(str(ind))
        f.write("\t")

        f.write(str(recommend[x][ind]))
        f.write("\n")
    f.close()
    
    
    result = V.rows.collect()
    result.saveAsTextFile("test")
    V.rdd.saveAsTextFile("hdfs://...")
    print(type(V))
    sc.stop()