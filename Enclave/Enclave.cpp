#include <stdio.h>
#include "Enclave_t.h"

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <sgx_trts.h>

//#include <time.h>

//#include <cstdlib>

//#include <cstring>
//#include <string>
//#include <cstdio>
//#include <stdlib.h>
//#include <cmath>
//#include <vector>
//#include <set>
//#include <iterator>
//#include <algorithm>

// Number of training samples

// Image size in MNIST database
const int width = 28;
const int height = 28;

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;
const int maxOfRand = 65535;


/*
// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
*/
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];


//void secure_init_array();


double w1[n1 + 1][n2 + 1];
double delta1[n1 + 1][n2 + 1];
double out1[n1 + 1];
double w2[n2 + 1][n3 + 1];
double delta2[n2 + 1][n3 + 1];
double in2[n2 + 1];
double out2[n2 + 1];
double theta2[n2 + 1];
double in3[n3 + 1];
double out3[n3 + 1];
double theta3[n3 + 1];

int nCorrect = 0;
int nTesting = 0;
int nTraining = 0;
/*
int randomlist[10000];
int r_cur = 0;
int r_all = 10000;
*/




/*
int* k_means(float** data, int n, int m, int k, float t, float** centroids) {

	// output cluster label for each data point
	int * labels = (int * ) calloc(n, sizeof(int));

	int h, i, j;
	int * counts = (int * ) calloc(k, sizeof(int)); // size of each cluster
	float old_error, error = DBL_MAX; // sum of squared euclidean distance
	float ** c = centroids ? centroids : (float ** ) calloc(k*m, sizeof(float * ));
	float ** c1 = (float * * ) calloc(k*m, sizeof(float * )); // temp centroids

        //initialization

	for (h = i = 0; i < k; h += n / k, i++) {
		c1[i] = (float * ) calloc(m, sizeof(float));
		if (!centroids) {
			c[i] = (float * ) calloc(m, sizeof(float));
		}
		// pick k points as initial centroids
		for (j = m; j-- > 0; c[i][j] = data[h][j]);
	}

	do {
		// save error from last step
		old_error = error, error = 0;

		// clear old counts and temp centroids
		for (i = 0; i < k; counts[i++] = 0) {
			for (j = 0; j < m; c1[i][j++] = 0);
		}

		for (h = 0; h < n; h++) {
			// identify the closest cluster
			float min_distance = DBL_MAX;
			for (i = 0; i < k; i++) {
				float distance = 0;
				for (j = m; j-- > 0; distance += pow(data[h][j] - c[i][j], 2));
				if (distance < min_distance) {
					labels[h] = i;
					min_distance = distance;
				}
			}

			// update size and temp centroid of the destination cluster
			for (j = m; j-- > 0; c1[labels[h]][j] += data[h][j]);
			counts[labels[h]]++;
			// update standard error
			error += min_distance;

		}

		for (i = 0; i < k; i++) { // update all centroids
			for (j = 0; j < m; j++) {
				c[i][j] = counts[i] ? c1[i][j] / counts[i] : c1[i][j];
			}
		}

	} while (fabs(error - old_error) > t);


	for (i = 0; i < k; i++) {
		if (!centroids) {
			free(c[i]);
		}
		free(c1[i]);
	}

	if (!centroids) {
		free(c);
	}
	free(c1);

	free(counts);

	return labels;
}
*/

int rand() {
    /*
    int temp = 8;
    if (r_all <= r_cur) {
        r_cur = 0;
        }
    temp = randomlist[r_cur++];

    return temp;
    */
    
    uint32_t val;
    sgx_read_rand((unsigned char *) &val, 4);
    return val % (maxOfRand);
}

// +-----------------------------------+
// | Secure Memory allocation for the network |
// +-----------------------------------+

void secure_init_array() {
    /*
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double [n2 + 1];
        delta1[i] = new double [n2 + 1];
    }
    
    out1 = new double [n1 + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double [n3 + 1];
        delta2[i] = new double [n3 + 1];
    }
    
    in2 = new double [n2 + 1];
    out2 = new double [n2 + 1];
    theta2 = new double [n2 + 1];

    // Layer 3 - Output layer
    in3 = new double [n3 + 1];
    out3 = new double [n3 + 1];
    theta3 = new double [n3 + 1];
    */
    
    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);
            
            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
		w1[i][j] = - w1[i][j];
		}
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;
			
            // Another strategy to randomize the weights - quite good 
            // w2[i][j] = (double)(rand() % 6) / 10.0;

            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
		w2[i][j] = - w2[i][j];
		}
        }
	}

}

// +------------------+
// | Secure Sigmoid function |
// +------------------+

double secure_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------+
// | Secure Forward process - Perceptron |
// +------------------------------+

void secure_perceptron() {
    for (int i = 1; i <= n2; ++i) {
            in2[i] = 0.0;
	}

    for (int i = 1; i <= n3; ++i) {
            in3[i] = 0.0;
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * w1[i][j];
            }
	}

    for (int i = 1; i <= n2; ++i) {
            out2[i] = secure_sigmoid(in2[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
            }
	}

    for (int i = 1; i <= n3; ++i) {
            out3[i] = secure_sigmoid(in3[i]);
	}
}

// +---------------+
// | Secure Norm L2 error |
// +---------------+

double secure_square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +----------------------------+
// | Secure Back Propagation Algorithm |
// +----------------------------+

void secure_back_propagation() {
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
            }
	}
    
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
            }
	}
}

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

int secure_learning_process() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
		delta1[i][j] = 0.0;
            }
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
		delta2[i][j] = 0.0;
            }
	}

    for (int i = 1; i <= epochs; ++i) {
        secure_perceptron();
        secure_back_propagation();
        if (secure_square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

/*
void secure_kmeans(float** data, int npoints, int k, size_t ndata) {
	char message[200];
	snprintf(message, 200, "data size: %d, # cluster: %d", npoints, k);
	print_message(message);

	print_message("DEBUG_E0");
	int * labels;
	print_message("DEBUG_E1");
	int dim = 2;
	labels = k_means(data, npoints, dim, k, 1e-4, NULL);
	print_message("DEBUG_E2");
	for (int i = 0; i < 20; i++) {
	    snprintf(message, 200, "data point %d is in cluster %d", i, labels[i]);
	    print_message(message);
	}

	free(labels);
    print_message("DEBUG_E3");
}
*/

void secure_train(double** data, double** gt, size_t ndata1) {
    
    for (int i = 0; i < ndata1; ++i) {
        
        ++nTraining;
        
        for (int j = 1; j <= n1; ++j) {
            out1[j] = data[i][j];
            }
        
        for (int j = 1; j <= n3; ++j) {
            expected[j] = gt[i][j];
            }
        
        int nIterations = secure_learning_process();
    }
    
}


void secure_test(double** datat, double** gtt, size_t ndata2) {
    
    int outcome[ndata2];
    
    for (int i = 0; i < ndata2; ++i) {
        ++nTesting;
        for (int j = 1; j <= n1; ++j) {
            out1[j] = datat[i][j];
            }
        
        for (int j = 1; j <= n3; ++j) {
            expected[j] = gtt[i][j];
        }
        
        secure_perceptron();
        
        // Prediction
        int predict = 1;
        for (int j = 2; j <= n3; ++j) {
            if (out3[j] > out3[predict]) {
                predict = j;
                }
            }
        
        int predi = (int)expected[predict];
        if (predi == 1) {
            ++nCorrect;
            }
        
        outcome[i] = predict - 1;
    
    }
    
    outputresult(outcome, ndata2);

}

void secure_summurize() {
    
    char messaget[200];
    snprintf(messaget, 200, "%d, ", nTraining);
    print_message(messaget);
    
    char message0[200];
    snprintf(message0, 200, "%d, ", nCorrect);
    print_message(message0);
    
    double accuracy = 0.0;
    accuracy = (double) (nCorrect) / nTesting * 100.0;
    char message[200];
    snprintf(message, 200, "%lf, ", accuracy);
    print_message(message);

}


void secure_nn(int randseed) {
    
    //r_all = ndata3;
    
    int temp_r = randseed;
    for (int uu = 0; uu < temp_r; ++uu) {
        int uuu = rand();
    }
    
    /*
    for (int i = 0; i < ndata3; ++i) {
        randomlist[i] = rlist[i];
    }
    */
    
    secure_init_array();
    
    // test
    
    /*
    char message1[200];
    snprintf(message1, 200, " Random: %d, %d, ", randomlist[--r_cur], r_cur);
    print_message(message1);
    */
}
