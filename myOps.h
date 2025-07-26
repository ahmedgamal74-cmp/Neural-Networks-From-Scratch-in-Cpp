#ifndef MY_OPS_H
#define MY_OPS_H
/*************************************************************/
#include <iostream>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <bitset>
#include <cctype>
#include <climits>
#include <chrono>
#include <assert.h>
#include <cstdlib>

#include "typedefs.h"

/** Done yet:
            myMatAdd
            myMatMul2D
            myMatVecMulAdd
            myReLU
            myReLUDeriv
            mySigmoid
            mySigmoidDeriv
 */

/* ***************************************************************************************************************** */

void myMatTranspose(const float32 *X, const uint32 m_x, const uint32 n_x);

void myRandomIntegarArrayGenerator(int32* array, uint32 length, int32 min, int32 max);

/**
 * @brief Multiplies two 2D matrices of arbitrary size (row-major, flat arrays).
 *
 * Performs the operation: C = A * B
 * where:
 *  - A is an m_a x n_a matrix
 *  - B is an m_b x n_b matrix
 *  - C is the resulting m_a x n_b matrix
 *
 * All matrices are expected to be stored in contiguous row-major order,
 * i.e., for element [i][j] use arr[i * ncols + j].
 *
 * @param[in]  A     Pointer to the first element of the left matrix (m_a x n_a)
 * @param[in]  B     Pointer to the first element of the right matrix (m_b x n_b)
 * @param[out] C     Pointer to the first element of the output matrix (m_a x n_b)
 * @param[in]  m_a   Number of rows in matrix A (and C)
 * @param[in]  n_a   Number of columns in matrix A (and rows in matrix B)
 * @param[in]  m_b   Number of rows in matrix B (must equal n_a)
 * @param[in]  n_b   Number of columns in matrix B (and C)
 *
 * @note All input and output matrices must be allocated by the caller.
 *       The function asserts if n_a != m_b.
 *       This function does not perform any dynamic memory allocation.
 *
 * Example usage:
 *     float32 A[2][3], B[3][2], C[2][2];
 *     my2DMatrixMultiplication(&A[0][0], &B[0][0], &C[0][0], 2, 3, 3, 2);
 */
void myMatMul2D(const float32 *A, const float32 *B, float32 *C,
                                   const uint32 m_a, const uint32 n_a, const uint32 m_b, const uint32 n_b);

/**
 * @brief Performs elementwise addition of two 2D matrices (row-major, flat arrays).
 *
 * Each element of the result is the sum of corresponding elements in A and B.
 * 
 * Performs: C[i][j] = A[i][j] + B[i][j]
 *
 * @param[in]  A     Pointer to the first element of matrix A (m_a x n_a)
 * @param[in]  B     Pointer to the first element of matrix B (m_b x n_b)
 * @param[out] C     Pointer to the first element of result matrix C (m_a x n_a)
 * @param[in]  m_a   Number of rows in A (and C)
 * @param[in]  n_a   Number of columns in A (and C)
 * @param[in]  m_b   Number of rows in B (must equal m_a)
 * @param[in]  n_b   Number of columns in B (must equal n_a)
 *
 * @note All matrices must be allocated by the caller and have the same shape.
 *       The function asserts if dimensions do not match.
 *       Use with flat arrays or `&A[0][0]` for 2D arrays.
 *
 * Example usage:
 *     float32 A[2][2] = {{1,2},{3,4}}, B[2][2] = {{10,20},{30,40}}, C[2][2];
 *     myMatAdd(&A[0][0], &B[0][0], &C[0][0], 2, 2, 2, 2);
 */
void myMatAdd(const float32 *A, const float32 *B, float32 *C,
                                   const uint32 m_a, const uint32 n_a, const uint32 m_b, const uint32 n_b);

/**
 * @brief Matrix-vector multiplication with bias: Y = W X + B
 *
 * W: (m_w x n_w), X: (n_w), B: (m_w), Y: (m_w)
 *
 * @param[in]  W    Pointer to (m_w x n_w) matrix W (row-major)
 * @param[in]  X    Pointer to input vector of length n_w
 * @param[in]  B    Pointer to bias vector of length m_w
 * @param[out] Y    Pointer to output vector of length m_w
 * @param[in]  m_w  Number of rows in W (and Y, and B)
 * @param[in]  n_w  Number of columns in W (and length of X)
 * @param[in]  m_x  Length of vector X (must equal n_w)
 * @param[in]  m_b  Length of bias vector B (must equal m_w)
 */
void myMatVecMulAdd(const float32 *W, const float32 *X, const float32 *B, float32 *Y,
                                    const uint32 m_w, const uint32 n_w, const uint32 m_x, const uint32 m_b);

/**
 * Computes the sigmoid activation function for a single float32 input.
 * Formula: sigmoid(x) = 1 / (1 + exp(-x))
 * Input:  X (float32) - the input value
 * Returns: (float32) sigmoid activation in the range (0, 1)
 */                                    
inline float32 mySigmoid(float32 X)
{
    return 1.0f / (1.0f + expf(-X));
}

/**
 * Computes the derivative of the sigmoid activation function for a single float32 input.
 * Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 * Input:  X (float32) - the input value
 * Returns: (float32) derivative of sigmoid at X
 */
inline float32 mySigmoidDeriv(float32 X)
{
    float32 sig = mySigmoid(X);
    return sig * (1.0f - sig);
}

/**
 * Computes the ReLU activation function for a single float32 input.
 * Formula: ReLU(x) = max(X,0)
 * Input:  X (float32) - the input value
 * Returns: (float32) ReLU activation in the range (0, X)
 */                                    
inline float32 myReLU(float32 X)
{
    return X > 0.0f ? X : 0.0f;
}

/**
 * Computes the derivative of the ReLU activation function for a single float32 input.
 * Formula: ReLU'(x) = 1 if X > 0, else = 0
 * Input:  X (float32) - the input value
 * Returns: (float32) derivative of ReLU at X
 */
inline float32 myReLUDeriv(float32 X)
{
    return X > 0.0f ? 1.0f : 0.0f;
}

/**
 * @brief Computes the Mean Squared Error (MSE) loss between two arrays.
 *
 * Calculates the average of the squared differences between true values and predicted values.
 * Commonly used as a loss function for regression tasks in neural networks.
 *
 * @param Y_true   Pointer to the array of true (target) values.
 * @param Y_pred   Pointer to the array of predicted values.
 * @param length   Number of elements in the arrays.
 * @return         The MSE loss as a float32.
 */
float32 myMeanSquaredError(const float32* Y_true, const float32* Y_pred, const uint32 length);

/**
 * @brief Computes the gradient (derivative) of Mean Squared Error (MSE) loss w.r.t. predictions.
 *
 * For each element, computes the partial derivative of the MSE loss with respect to Y_pred,
 * and writes it to the output gradient array. Used during neural network backpropagation.
 *
 * @param Y_true   Pointer to the array of true (target) values.
 * @param Y_pred   Pointer to the array of predicted values.
 * @param grad     Pointer to the array where the gradient will be stored (output).
 * @param length   Number of elements in the arrays.
 */
void myMeanSquaredErrorGrad(const float32* Y_true, const float32* Y_pred, float32* grad, const uint32 length);

/**
 * @brief Computes the binary cross entropy loss between two arrays.
 *
 * Calculates the cross entropy between true values and predicted values.
 * Commonly used as a loss function for classififcation tasks in neural networks.
 *
 * @param Y_true   Pointer to the array of true (target) values.
 * @param Y_pred   Pointer to the array of predicted values.
 * @param length   Number of elements in the arrays.
 * @return         The binary cross entropy loss as a float32.
 */
float32 myBinaryCrossEntropy(const float32* Y_true, const float32* Y_pred, const uint32 length);

/**
 * @brief Computes the gradient (derivative) of binary cross entropy loss loss w.r.t. predictions.
 *
 * For each element, computes the partial derivative of the binary cross entropy loss with respect to Y_pred,
 * and writes it to the output gradient array. Used during neural network backpropagation.
 *
 * @param Y_true   Pointer to the array of true (target) values.
 * @param Y_pred   Pointer to the array of predicted values.
 * @param grad     Pointer to the array where the gradient will be stored (output).
 * @param length   Number of elements in the arrays.
 */
void myBinaryCrossEntropyGrad(const float32* Y_true, const float32* Y_pred, float32* grad, const uint32 length);

/**
 * @brief Performs a gradient descent (SGD) update on a 2D parameter array (flattened).
 *
 * Updates the parameter array W in-place by subtracting the gradient (scaled by learning rate) from each element.
 * 
 * W[i] = W[i] - LR * grads[i]
 *
 * @param W      Pointer to parameter array (weights, shape w_a * w_b, flattened).
 * @param grads  Pointer to gradients array (same size as W).
 * @param LR     Learning rate (float32).
 * @param w_a    First dimension of weight matrix.
 * @param w_b    Second dimension of weight matrix.
 */
void myGradientDescent(float32* W, const float32* grads, float32 LR, const uint32 w_a, const uint32 w_b);



/*************************************************************/
#endif // MY_OPS_H
