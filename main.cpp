#include <iostream>
#include <cmath>
#include <iomanip>
#include <bitset>
#include <cctype>
#include <climits>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <assert.h>
#include <cstdlib>  
#include <cmath>    
#include <ctime> 
#include <random> 

#include "typedefs.h"
#include "myOps.h"
#include "config.h"

// using namespace std;
// using Matrix = std::vector<std::vector<int>>;

int main() {
	srand(static_cast<unsigned int>(time(0)));
	std::cout<<"---------------------------------------------------------------------------------------------------"<<std::endl;
	// srand(time(nullptr));
	
	// // Set the precision for floating-point output
	// std::cout << std::fixed << std::setprecision(6);

	// int32 arr[2][2] = {{1, 2}, {3, 4}};
	// printf("%i \n", arr[0][0]); 

	// float32 A[2][3] = {
    //     {1, 2, 3},
    //     {4, 5, 6}
    // };
    // float32 B[3][2] = {
    //     {7, 8},
    //     {9, 10},
    //     {11, 12}
    // };
    // float32 C[2][2] = {{1,2}, {3,4}};

    // myMatMul2D(&A[0][0], &B[0][0], &C[0][0], 2, 3, 3, 2);

	// for (int i = 0; i < 2; ++i) {
	// 	for (int j = 0; j < 2; ++j)
	// 		printf(" %f ", C[i][j]);
	// 		// std::cout << "C[" << i << "][" << j << "] = " << C[i][j] << "  ";
	// 	// std::cout << std::endl;
	// }




	// float32 A[2][2] = {{1,2},{3,4}};
	// float32 B[2][2] = {{10,20},{30,40}};
	// float32 C[2][2];

	// myMatAdd(&A[0][0], &B[0][0], &C[0][0], 2 , 2, 2, 2);
	// //  C = {{11,22},{33,44}}

	// for (int i = 0; i < 2; ++i) {
	// 	for (int j = 0; j < 2; ++j)
	// 		printf(" %f ", C[i][j]);
	// 		// std::cout << "C[" << i << "][" << j << "] = " << C[i][j] << "  ";
	// 	// std::cout << std::endl;
	// 	printf("\n");
	// }



	// float32 W[2][3] = { {1,2,3}, {4,5,6} }; // 2 * 3
	// float32 X[3] = 	{{10},{20},{30}};		// {10, 20, 30};
	// float32 B[2] = {{100},{200}}; // {100, 200};
	// float32 Y[2];

	// myMatVecMulAdd(&W[0][0], X, B, Y, 2, 3, 3, 2);

	// // Y[0] = 1*10 + 2*20 + 3*30 + 100 = 10+40+90+100 = 240
	// // Y[1] = 4*10 + 5*20 + 6*30 + 200 = 40+100+180+200 = 520
	// printf("%f	", Y[0]);
	// printf("%f \n", Y[1]);



	// // Test values
    // double test_vals[] = {-3, -1, 0, 1, 3};
    // int n = sizeof(test_vals) / sizeof(double);

    // std::cout << "x\tSigmoid(x)\tSigmoid'(x)\n";
    // std::cout << "---------------------------------\n";
    // for(int i = 0; i < n; ++i) {
    //     double x = test_vals[i];
    //     double sx = mySigmoid(x);
    //     double dsx = mySigmoidDeriv(x);
    //     std::cout << x << "\t" << sx << "\t" << dsx << std::endl;
    // }

	// float32 weight_matrix[4] = {0.0, 0.5, 1.5, 2.0};
	// float32 grads[4] = {0.0, 0.5, 1.5, 2.0};

	// myGradientDescent(weight_matrix, grads, 0.5, 1, 4);

	// for(int i = 0; i < 4; i++){
	// 	printf(" %f 	", weight_matrix[i]);
	// }

// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	const uint32 DataSamplesCount = 4;
	const uint32 DataFeatures = 2;

	const float32 X[DataSamplesCount][DataFeatures] = {
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0 ,1.0},
	};

	const float32 y[DataSamplesCount][1] = {
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	};

	// printf("Data: \n");
	// myPrintMatrix(&X[0][0], 4, 2);	
	// printf("Labels: \n");
	// myPrintMatrix(&y[0], 4, 1);

	// float32 X_T[2][4];
	// myMatrixTranspose(&X[0][0], &X_T[0][0], 4, 2);
	// myPrintMatrix(&X[0][0], 4, 2);
	// myPrintMatrix(&X_T[0][0], 2, 4);


	// float32 X2[2][3] = {
    // {1, 2, 3},
    // {4, 5, 6}
	// };
	// float32 X_T2[3][2];
	// myMatrixTranspose(&X2[0][0], &X_T2[0][0], 2, 3);
	// myPrintMatrix(&X2[0][0], 2, 3);
	// myPrintMatrix(&X_T2[0][0], 3, 2);


	printf("Data: \n");
	printf(" X1	       X2           y \n");
    for(uint32 i = 0; i < DataSamplesCount; i++){
        for(uint32 j = 0; j < DataFeatures; j++){
            printf(" %f     ", X[i][j]);
        }
		printf("%f", y[i][0]);
        printf("\n");
    }
	printf("------------------------------------\n");

	const uint32 input_size = DataFeatures;
	const uint32 hidden_layer_size = 2;
	const uint32 output_layer_size = 1;

	float32 W1[hidden_layer_size][input_size];
	float32 b1[hidden_layer_size][DataSamplesCount];
	float32 W2[output_layer_size][hidden_layer_size];
	float32 b2[output_layer_size][DataSamplesCount];	
	
	float32 d_W1[hidden_layer_size][input_size];
	float32 d_b1[hidden_layer_size][DataSamplesCount];
	float32 d_W2[output_layer_size][hidden_layer_size];
	float32 d_b2[output_layer_size][DataSamplesCount];


	myRandomInitializerWithNormalDistribution(&W1[0][0], hidden_layer_size*input_size, 0.0f, 1.0f);
	myRandomInitializerWithNormalDistribution(&b1[0][0], hidden_layer_size*DataSamplesCount, 0.0f, 1.0f);
	myRandomInitializerWithNormalDistribution(&W2[0][0], output_layer_size*hidden_layer_size, 0.0f, 1.0f);
	myRandomInitializerWithNormalDistribution(&b2[0][0], output_layer_size*DataSamplesCount, 0.0f, 1.0f);

	// W1 -> HxI , X -> 4xI , b1 -> Hx4
	// z1 = W1.X + b1 = HxI.I*4 + H*4
	float32 z1[hidden_layer_size][DataSamplesCount];
	float32 a1[hidden_layer_size][DataSamplesCount];
	float32 z2[output_layer_size][DataSamplesCount];
	float32 a2[output_layer_size][DataSamplesCount];	
	
	float32 d_z1[hidden_layer_size][DataSamplesCount];
	float32 d_a1[hidden_layer_size][DataSamplesCount];
	float32 d_z2[output_layer_size][DataSamplesCount];
	float32 d_a2[output_layer_size][DataSamplesCount];

	float32 X_T[DataFeatures][DataSamplesCount];
	myMatrixTranspose(&X[0][0], &X_T[0][0], DataSamplesCount, DataFeatures);
	float32 y_t[1][DataSamplesCount];
	myMatrixTranspose(&y[0][0], &y_t[0][0], DataSamplesCount, 1);

	
	for(uint32 epoch = 0; epoch < 100; epoch++)
	{
		float32 loss = 0.0f;

		myMatVecMulAdd(&W1[0][0], &X_T[0][0], &b1[0][0], &z1[0][0],
							hidden_layer_size, input_size, DataSamplesCount, hidden_layer_size);
		
		for(uint32 i = 0; i < hidden_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				a1[i][j] = mySigmoid(z1[i][j]);

		myMatVecMulAdd(&W2[0][0], &a1[0][0], &b2[0][0], &z2[0][0], 
							output_layer_size, hidden_layer_size, hidden_layer_size, output_layer_size);	
				
		for(uint32 i = 0; i < output_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				a2[i][j] = mySigmoid(z2[i][j]);


		loss = myMeanSquaredError(&y_t[0][0], &a2[0][0], DataSamplesCount);
		

		// Ahmed
		




		if(epoch%10 == 0){
			printf("Epcoh: %i, Loss: %f \n", epoch, loss);
		}
	}

	std::cout<<std::endl;
	std::cout<<"---------------------------------------------------------------------------------------------------"<<std::endl;
	return 0;
}
