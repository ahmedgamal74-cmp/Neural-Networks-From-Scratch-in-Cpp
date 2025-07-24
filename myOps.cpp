#include "myOps.h"

void myMatTranspose(const float32 *X, const uint32 m_x, const uint32 n_x)
{

}

void myRandomIntegarArrayGenerator(int32* array, uint32 length, int32 min, int32 max)
{
    for(uint32 i=0; i<length; i++){
        array[i] = min + rand() % (max-min+1);
    }
}

void myMatMul2D(const float32 *A, const float32 *B, float32 *C,
                                   const uint32 m_a, const uint32 n_a, const uint32 m_b, const uint32 n_b)
{
    if(n_a != m_b){
        printf("ERROR: matrices sizes are NOT matched. \n");
        assert(0);
        return;
    }

    const uint32 m_c = m_a;
    const uint32 n_c = n_b;

    for(uint32 i = 0; i < m_c; i++){
        for(uint32 j = 0; j < n_c; j++){
            C[i * n_c + j] = 0;
            for(uint32 k = 0; k < n_a; k++){
                C[i * n_c + j] += A[i * n_a + k] * B[k * n_b + j];
            }
        }
    }
}

void myMatAdd(const float32 *A, const float32 *B, float32 *C,
                                   const uint32 m_a, const uint32 n_a, const uint32 m_b, const uint32 n_b)
{
    if((m_a != m_b) || n_a != n_b){
        printf("ERROR: matrices sizes are NOT matched. \n");
        assert(0);
        return;
    } 
    
    uint32 elementsCount = m_a*n_a;

    for(uint32 i = 0; i < elementsCount; i++){
        C[i] = A[i] + B[i];
    }

}

void myMatVecMulAdd(const float32 *W, const float32 *X, const float32 *B, float32 *Y,
                                   const uint32 m_w, const uint32 n_w, const uint32 m_x, const uint32 m_b)
{
    // if((m_w != m_b) || m_x != n_w){
    //     printf("ERROR: matrices sizes are NOT matched. \n");
    //     assert(0);
    //     return;
    // } 
    

    // for(uint32 i = 0; i < m_w; i++){
    //     Y[i] = 0;
    //     for(uint32 j = 0; j < n_w; j++){
    //         Y[i] += W[i*n_w + j] * X[j];
    //     }
    //     Y[i] += B[i];
    // }

    myMatMul2D(W, X, Y, m_w, n_w, n_w, 1);
    myMatAdd(Y, B, Y, m_w, 1, m_b, 1);     

}


