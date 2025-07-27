#include "myOps.h"

void myPrintMatrix(const float32* arr, const uint32 arr_m, const uint32 arr_n)
{
    for(uint32 i = 0; i < arr_m; i++){
        for(uint32 j = 0; j < arr_n; j++){
            printf(" %f     ", arr[i * arr_n + j]);
        }
        printf("\n");
    }
}

void myRandomInitializerWithNormalDistribution(float32* array, const uint32 length, const float32 mean, const float32 std_dev)
{
    float32 n1=0.0f, n2=0.0f, norm_var=0.0f;
    for(uint32 i=0; i<length; i++){

        n1 = float32(rand()) / float32(RAND_MAX);
        n2 = float32(rand()) / float32(RAND_MAX);

        norm_var = sqrt(-2.0f * logf(n1 + 1e-7f)) * cosf(2.0f * PI * n2);

        array[i] = norm_var*std_dev + mean;
    }
}

void myMatrixTranspose(const float32 *X, float32* X_T, const uint32 m_x, const uint32 n_x)
{
    for (uint32 i = 0; i < m_x; i++){
        for (uint32 j = 0; j < n_x; j++){
            X_T[j * m_x + i] = X[i * n_x + j];
        }
    }
}

void myMatMul2D(const float32 *A, const float32 *B, float32 *C,
                                   const uint32 m_a, const uint32 n_a, const uint32 m_b, const uint32 n_b)
{
    if(n_a != m_b){
        printf("ERROR myMatMul2D: matrices sizes are NOT matched. \n");
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
        printf("ERROR myMatAdd: matrices sizes are NOT matched. \n");
        assert(0);
        return;
    } 
    
    uint32 elementsCount = m_a*n_a;

    for(uint32 i = 0; i < elementsCount; i++){
        C[i] = A[i] + B[i];
    }

}

void myMatVecMulAdd(const float32 *W, const float32 *X, const float32 *B, float32 *Y,
                                   const uint32 m_w, const uint32 n_w, const uint32 m_x, const uint32 n_x)
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

    myMatMul2D(W, X, Y, m_w, n_w, m_x, n_x);
    // myMatAdd(Y, B, Y, m_w, n_x, m_w, 1);  
    for(uint32 i = 0; i < m_w; i++){
        for(uint32 j = 0; j < n_x; j++){
            Y[j + i*n_x] += B[i];
        }
    }   

}

float32 myMeanSquaredError(const float32* Y_true, const float32* Y_pred, const uint32 length)
{
    float sum = 0.0f;
    for(uint32 i = 0; i < length; i++){
        sum += (Y_true[i] - Y_pred[i]) * (Y_true[i] - Y_pred[i]) ;
    }
    sum = sum / length;
    return sum;
}

void myMeanSquaredErrorGrad(const float32* Y_true, const float32* Y_pred, float32* grad, const uint32 length)
{
    for(uint32 i = 0; i < length; i++){
        grad[i] = 2.0f * (Y_pred[i] - Y_true[i]) / length;
    }
}

float32 myBinaryCrossEntropy(const float32* Y_true, const float32* Y_pred, const uint32 length)
{
    float sum = 0.0f;


    for(uint32 i = 0; i < length; i++){

        float32 y_pred_temp = Y_pred[i];

        if(y_pred_temp < 1e-7f)
            y_pred_temp = 1e-7f;
        else if(y_pred_temp > (1.0f - 1e-7f))
            y_pred_temp = 1.0f - 1e-7f;


        sum += -Y_true[i] * logf(y_pred_temp) - (1.0f - Y_true[i]) * logf(1.0f - y_pred_temp);
    }
    sum = sum / length;
    return sum;
}

void myBinaryCrossEntropyGrad(const float32* Y_true, const float32* Y_pred, float32* grad, const uint32 length)
{
    for(uint32 i = 0; i < length; i++){

        float32 y_pred_temp = Y_pred[i];

        if(y_pred_temp < 1e-7f)
            y_pred_temp = 1e-7f;
        else if(y_pred_temp > (1.0f - 1e-7f))
            y_pred_temp = 1.0f - 1e-7f;

        grad[i] = (y_pred_temp - Y_true[i]) / (y_pred_temp * (1.0f - y_pred_temp) * length);
    }
}

void myGradientDescent(const float32* grads, float32* W, float32 LR, const uint32 w_a, const uint32 w_b)
{
    uint32 length = w_a * w_b;
    for(uint32 i = 0; i < length; i++){
        W[i] -= LR * grads[i];
    }
}











































































































































