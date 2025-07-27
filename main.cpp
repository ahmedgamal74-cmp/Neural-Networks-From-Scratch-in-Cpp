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

#include <sys/stat.h>
#include <sys/types.h>

void writeMatrixToFile(FILE* f, const char* label, float32* mat, uint32 rows, uint32 cols)
{
    fprintf(f, "# %s (%ux%u)\n", label, rows, cols);

    for (uint32 i = 0; i < rows; i++) {
        for (uint32 j = 0; j < cols; j++) {
            fprintf(f, "%f ", mat[i * cols + j]);
        }
        fprintf(f, "\n");
    }
}


int main() {
	srand(static_cast<unsigned int>(time(0)));
	// srand(time(nullptr));
	// std::cout << std::fixed << std::setprecision(6);
	std::cout<<"---------------------------------------------------------------------------------------------------"<<std::endl;
	
	FILE* loss_file = fopen("../log_loss.txt", "w");
	FILE* f = fopen("../log_weights.txt", "w");

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
	float32 b1[hidden_layer_size][1];
	float32 W2[output_layer_size][hidden_layer_size];
	float32 b2[output_layer_size][1];	
	
	float32 d_W1[hidden_layer_size][input_size];
	float32 d_b1[hidden_layer_size][1];
	float32 d_W2[output_layer_size][hidden_layer_size];
	float32 d_b2[output_layer_size][1];


	myRandomInitializerWithNormalDistribution(&W1[0][0], hidden_layer_size*input_size, 0.0f, sqrtf(1.0f / input_size));
	myRandomInitializerWithNormalDistribution(&b1[0][0], hidden_layer_size*1, 0.0f, sqrtf(1.0f / input_size));
	myRandomInitializerWithNormalDistribution(&W2[0][0], output_layer_size*hidden_layer_size, 0.0f, sqrtf(1.0f / input_size));
	myRandomInitializerWithNormalDistribution(&b2[0][0], output_layer_size*1, 0.0f, sqrtf(1.0f / input_size));

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
	float32 a1_t[DataSamplesCount][hidden_layer_size];
	float32 W2_t[hidden_layer_size][output_layer_size];

	uint32 epochs = 100000;
	for(uint32 epoch = 0; epoch < epochs; epoch++)
	{
		float32 loss = 0.0f;

		myMatVecMulAdd(&W1[0][0], &X_T[0][0], &b1[0][0], &z1[0][0],
							hidden_layer_size, input_size, DataFeatures, DataSamplesCount);
		
		for(uint32 i = 0; i < hidden_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				a1[i][j] = mySigmoid(z1[i][j]);

		myMatrixTranspose(&a1[0][0], &a1_t[0][0], hidden_layer_size, DataSamplesCount);	

		myMatVecMulAdd(&W2[0][0], &a1[0][0], &b2[0][0], &z2[0][0], 
							output_layer_size, hidden_layer_size, hidden_layer_size, DataSamplesCount);	
				
		myMatrixTranspose(&W2[0][0], &W2_t[0][0], output_layer_size, hidden_layer_size);


		for(uint32 i = 0; i < output_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				a2[i][j] = mySigmoid(z2[i][j]);


		loss = myBinaryCrossEntropy(&y_t[0][0], &a2[0][0], DataSamplesCount);
		

		myBinaryCrossEntropyGrad(&y_t[0][0], &a2[0][0], &d_a2[0][0], 4);
		
		for(uint32 i = 0; i < output_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				d_z2[i][j] = d_a2[i][j] * mySigmoidDeriv(z2[i][j]);

		myMatMul2D(&d_z2[0][0], &a1_t[0][0], &d_W2[0][0], output_layer_size, DataSamplesCount, DataSamplesCount, hidden_layer_size);		

		for(uint32 i = 0; i < output_layer_size; i++){
			d_b2[i][0] = 0.0f;
			for(uint32 j = 0; j < DataSamplesCount; j++)
				d_b2[i][0] += d_z2[i][j];
			d_b2[i][0] /= DataSamplesCount;
		}

		myMatMul2D(&W2_t[0][0], &d_z2[0][0], &d_a1[0][0], hidden_layer_size, output_layer_size, output_layer_size, DataSamplesCount);		

		for(uint32 i = 0; i < hidden_layer_size; i++)
			for(uint32 j = 0; j < DataSamplesCount; j++)
				d_z1[i][j] = d_a1[i][j] * mySigmoidDeriv(z1[i][j]);

		myMatMul2D(&d_z1[0][0], &X[0][0], &d_W1[0][0], hidden_layer_size, DataSamplesCount, DataSamplesCount, input_size);		

		for(uint32 i = 0; i < hidden_layer_size; i++){
			d_b1[i][0] = 0.0f;
			for(uint32 j = 0; j < DataSamplesCount; j++)
				d_b1[i][0] += d_z1[i][j];
			d_b1[i][0] /= DataSamplesCount;	
		}
			

		myGradientDescent(&d_W2[0][0], &W2[0][0], 0.1, output_layer_size, hidden_layer_size);
		myGradientDescent(&d_b2[0][0], &b2[0][0], 0.1, output_layer_size, 1);
		myGradientDescent(&d_W1[0][0], &W1[0][0], 0.1, hidden_layer_size, input_size);
		myGradientDescent(&d_b1[0][0], &b1[0][0], 0.1, hidden_layer_size, 1);


		
		if(epoch%(epochs/10) == 0){
			printf("Epcoh: %i, Loss: %f \n", epoch, loss);
			fprintf(loss_file, "%d,%f\n", epoch, loss);

		}
	}
	fclose(loss_file);
	writeMatrixToFile(f, "W1", &W1[0][0], hidden_layer_size, input_size);
	writeMatrixToFile(f, "b1", &b1[0][0], hidden_layer_size, 1);
	writeMatrixToFile(f, "W2", &W2[0][0], output_layer_size, hidden_layer_size);
	writeMatrixToFile(f, "b2", &b2[0][0], output_layer_size, 1);

	fclose(f);



	printf("\nFinal Predictions:\n");
	for(uint32 i = 0; i < DataSamplesCount; i++){
		printf("Input: [%f, %f], Prediction: %f, Target: %f\n",
			X[i][0], X[i][1], a2[0][i], y[i][0]);
	}
	// printf("------------------------------------\n");
	// printf("W1: \n");
	// myPrintMatrix(&W1[0][0], 2, 2);
	// printf("------------------------------------\n");
	// printf("b1: \n");
	// myPrintMatrix(&b1[0][0], 2, 1);
	// printf("------------------------------------\n");
	// printf("W2: \n");
	// myPrintMatrix(&W2[0][0], 1, 2);
	// printf("------------------------------------\n");
	// printf("b2: \n");
	// myPrintMatrix(&b2[0][0], 1, 1);
	// printf("------------------------------------\n");



	std::cout<<std::endl;
	std::cout<<"---------------------------------------------------------------------------------------------------"<<std::endl;
	return 0;
}
