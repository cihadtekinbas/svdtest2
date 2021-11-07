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
    # rows = sc.parallelize([
    #     Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    #     Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    #     Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    # ])
    trainA = sparse.load_npz("./testcode/data/Aratings-transform_user.npz")
    testB = sparse.load_npz("./testcode/data/Bratings-transform_user.npz")
    trainC=sparse.load_npz("./testcode/data/Cratings-transform_user.npz")
    testD=sparse.load_npz("./testcode/data/Dratings-transform_user.npz")


    print(trainA.shape)
    print(testB.shape)
    testB.resize(251457,4710)
    print(trainC.shape)
    print(testD.shape)

    testDcoo=testD.tocoo()

    dictt={}
    for i in range(0,len(testDcoo.row)):
        x=testDcoo.row[i]
        if x not in dictt:
            dictt[x]=[]
        dictt[x].append(testDcoo.data[i])

    data=[]
    for i in dictt:
        s=0
        for j in dictt[i] :
            s=s+j
        t=s/len(dictt[i])
        for z in range(len(dictt[i])):
            data.append(t)
    data=np.asarray(data)
    import numpy as np
    from scipy.sparse import coo_matrix
    coo = coo_matrix((data, (testDcoo.row, testDcoo.col)))
    print(coo.shape)
    coocsr=coo.tocsr()

    user_number=10000

    A_B=hstack((trainA, testB))
    print("buradayım")
    print(trainC.shape,coocsr.shape)
    numbers=[]
    for i in range(user_number):
        numbers.append(i)


    C_D=hstack((trainC[numbers], coocsr[numbers]))
    print(A_B.shape)
    print(C_D.shape)
    C_D.resize(user_number,23513)
    A_B_C_D=vstack((A_B, C_D))
    print(A_B_C_D.shape)












    print("i am here###################")
    

    A_B_C_D=A_B_C_D.tocoo()
    vectorlist={} 
    dictt={}
    for i in range(0,len(A_B_C_D.row)):
        x=A_B_C_D.row[i]
        if x not in vectorlist:
            vectorlist[x]={}
        vectorlist[x][A_B_C_D.col[i]]=A_B_C_D.data[i]
    listofvectors=[]

    for i in range(0,len(A_B_C_D.row)):
        if i in vectorlist:
            listofvectors.append(Vectors.sparse(23513, vectorlist[i]))
        # print(vectorlist[i])
        
    print(len(listofvectors))
    rows = sc.parallelize([
        listofvectors
    ])
    print(rows)
    mat = RowMatrix(rows)
    print(mat)
    
    
    

    # Compute the top 5 singular values and corresponding singular vectors.
    svd = mat.computeSVD(23513, computeU=True)
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
    
    
    recommend={}
    for i in C_D.row:
        if i not in recommend:
            recommend[i]=[]
    for i in recommend:
        recommend[i]=V[251457+i,18803:]

    f=open("recommend","w")
    for x in recommend:
        ind = np.argpartition(recommend[x], -10)[-10:]
        f.write(str(ind))
        f.write("\t")

        f.write(str(recommend[x][ind]))
        f.write("\n")
    f.close()
    
    
    sc.stop()