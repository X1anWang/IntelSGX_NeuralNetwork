#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <iterator>
#include <algorithm>

#include <unistd.h>
#include "Enclave_u.h"
#include "sgx_urts.h"
#include "sgx_utils/sgx_utils.h"
#include <time.h>
using namespace std;

/* Global EID */
sgx_enclave_id_t global_eid = 0;






// Testing image file name
const string testing_image_fn = "mnist/t10k-images.idx3-ubyte";

// Testing label file name
const string testing_label_fn = "mnist/t10k-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "testing-report.dat";

// Number of testing samples
const int nTesting = 10000;
const int nRan = 10000;




// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
// const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn2 = "training-report.dat";

// Number of training samples
const int nTraining = 60000;






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

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

ifstream image2;
ifstream label2;
ofstream report2;





// +--------------------+
// | About the software |
// +--------------------+

/*
void about() {
	
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
	
}
*/

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+

void init_array() {
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
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------+
// | Forward process - Perceptron |
// +------------------------------+

void perceptron() {
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
		out2[i] = sigmoid(in2[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= n3; ++i) {
		out3[i] = sigmoid(in3[i]);
	}
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation() {
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

int learning_process() {
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
        perceptron();
        back_propagation();
        if (square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

int input() {
    // Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}
    
    /*
    cout << "Image:" << endl;
    for (int j = 1; j <= height; ++j) {
	for (int i = 1; i <= width; ++i) {
            cout << d[i][j];
            }
            cout << endl;
	}
    */
    
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
            }
	}

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    
    return (int)number;
    /*
    cout << "Label: " << (int)(number) << endl;
    */
}

int input2() {
    // Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image2.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}
    
    /*
    cout << "Image:" << endl;
    for (int j = 1; j <= height; ++j) {
	for (int i = 1; i <= width; ++i) {
            cout << d[i][j];
            }
            cout << endl;
	}
    */
    
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
            }
	}

    // Reading label
    label2.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    
    return (int)number;
    /*
    cout << "Label: " << (int)(number) << endl;
    */
}

// +------------------------+
// | Saving weights to file |
// +------------------------+
/*
void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
    // Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	
    // Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	
    file.close();
}
*/




// Neural Network using SGX enclave
int SGXnn()  {
    //about();
    int randomseed = rand() % 100;
    /*
    //prepare random
    int *rlist;
    rlist = new int[nRan];
    
    for (int u = 0; u < nRan; ++u) {
        rlist[u] = rand();
        }
    */
    
    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file
    //report2.open(report_fn2.c_str(), ios::out);
    image2.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label2.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file
    
    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
    
    cout << "Start enclave init..." << endl;
    //auto enclaveinit = chrono::high_resolution_clock::now();

    if (initialize_enclave(&global_eid, "enclave.token", "enclave.signed.so") < 0) {
        std::cout << "Fail to initialize enclave." << std::endl;
        return 0;
    }
    
    double **data;
    data = new double*[nTraining];
    
    double **ground_truth;
    ground_truth = new double*[nTraining];
    
    for (int o = 0; o < nTraining; ++o){
        
        char number;
        
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                image.read(&number, sizeof(char));
                if (number == 0) {
                    d[i][j] = 0;
                    }
                else {
                    d[i][j] = 1;
                    }
                }
            }
        
        data[o] = new double[(height * width + 1)];
        
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                int pos = i + (j - 1) * width;
                data[o][pos] = d[i][j];
                }
            }
        
        // Reading label        
        ground_truth[o] = new double[n3 + 1];
        label.read(&number, sizeof(char));
        for (int i = 1; i <= n3; ++i) {
                    ground_truth[o][i] = 0.0;
            }
        ground_truth[o][number + 1] = 1.0;
        
        }
    
    double **datat;
    datat = new double*[nTesting];
    
    double **ground_trutht;
    ground_trutht = new double*[nTesting];
    
    // Reading file headers
    //char number;
    for (int i = 1; i <= 16; ++i) {
        image2.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label2.read(&number, sizeof(char));
	}
    
    for (int o = 0; o < nTesting; ++o){
        
        char number;
        
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                image2.read(&number, sizeof(char));
                if (number == 0) {
                    d[i][j] = 0; 
                    }
                else {
                    d[i][j] = 1;
                    }
                }
            }
        
        datat[o] = new double[(height * width + 1)];
        
        for (int j = 1; j <= height; ++j) {
            for (int i = 1; i <= width; ++i) {
                int pos = i + (j - 1) * width;
                datat[o][pos] = d[i][j];
                }
            }
        
        // Reading label        
        ground_trutht[o] = new double[n3 + 1];
        label2.read(&number, sizeof(char));
        for (int i = 1; i <= n3; ++i) {
                ground_trutht[o][i] = 0.0;
            }
        ground_trutht[o][number + 1] = 1.0;
        
        }
    // Neural Network Initialization
    // init_array();
    
    report.close();
    image.close();
    label.close();
    //report2.close();
    image2.close();
    label2.close();
    
    secure_nn(global_eid, randomseed);
    
    int batchh = 5;
    int timees = nTraining / batchh;
    for (int iiu = 0; iiu < timees; ++iiu) {
        int figr = iiu * batchh;
        double** ppoint;
        double** ppoint2;
        ppoint = new double*[batchh];
        ppoint2 = new double*[batchh];
        
        for (int oo = 0; oo < batchh; ++oo) {
            ppoint[oo] = new double[(height * width + 1)];
            for (int jj = 1; jj <= width * height; ++jj) {
                    ppoint[oo][jj] = data[(oo+figr)][jj];
                }
            }
        
        for (int oo = 0; oo < batchh; ++oo) {
            ppoint2[oo] = new double[n3 + 1];
            for (int jj = 1; jj <= n3; ++jj) {
                    ppoint2[oo][jj] = ground_truth[(oo+figr)][jj];
                }
            }
        
        secure_train(global_eid, ppoint, ppoint2, batchh);
    }
    
    timees = nTesting / batchh;
    for (int iiu = 0; iiu < timees; ++iiu) {
        int figr = iiu * batchh;
        double** ppoint;
        double** ppoint2;
        ppoint = new double*[batchh];
        ppoint2 = new double*[batchh];
        
        for (int oo = 0; oo < batchh; ++oo) {
            ppoint[oo] = new double[(height * width + 1)];
            for (int jj = 1; jj <= width * height; ++jj) {
                    ppoint[oo][jj] = datat[(oo+figr)][jj];
                }
            }
        
        for (int oo = 0; oo < batchh; ++oo) {
            ppoint2[oo] = new double[n3 + 1];
            for (int jj = 1; jj <= n3; ++jj) {
                    ppoint2[oo][jj] = ground_trutht[(oo+figr)][jj];
                }
            }
        
        secure_test(global_eid, ppoint, ppoint2, batchh);
    }
    
    secure_summurize(global_eid);
    //out1 = new double [n1 + 1];
    
    /*
    for (int sample = 1; sample <= nTraining; ++sample) {
        
        //cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        input();
		
	// Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();

	// Write down the squared error
	cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
		
	// Save the current network (weights)
	if (sample % 100 == 0) {
            cout << "Saving the network to " << model_fn << " file." << endl;
            write_matrix(model_fn);
            }
        
        
        input();
        secure_train(global_eid, out1);
    }
    */
    
    // Save the final network
    // write_matrix(model_fn);


    
    return 1;
}

// Neural Network without using SGX enclave
int noSGXnn() {
    //about();
    
    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
    
    // Neural Network Initialization
    init_array();
    int temp = 0;
    
    for (int sample = 1; sample <= nTraining; ++sample) {
        //cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        temp = input();
		
        // Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();

        /*
        // Write down the squared error
        cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
    
    
        // Save the current network (weights)
        if (sample % 100 == 0) {
            cout << "Saving the network to " << model_fn << " file." << endl;
            write_matrix(model_fn);
            }
        */
    }
    // Save the final network
    //write_matrix(model_fn);

    report.close();
    image.close();
    label.close();
    
    
    
    
    
    //testing
    report2.open(report_fn2.c_str(), ios::out);
    image2.open(testing_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label2.open(testing_label_fn.c_str(), ios::in | ios::binary ); // Binary label file
    
    // Reading file headers
    //char number;
    for (int i = 1; i <= 16; ++i) {
        image2.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label2.read(&number, sizeof(char));
	}
    
    int nCorrect = 0;
    for (int sample = 1; sample <= nTesting; ++sample) {
        
        // Getting (image, label)
        int label = input2();
		
	// Classification - Perceptron procedure
        perceptron();
        
        // Prediction
        int predict = 1;
        for (int i = 2; i <= n3; ++i) {
            if (out3[i] > out3[predict]) {
                predict = i;
                }
        }
	--predict;
	
	if (label == predict) {
            ++nCorrect;
            }
    }
    
    //printf("%d, ", nCorrect);
    double accuracy = (double)(nCorrect) / nTesting * 100.0;
    printf("%lf, ", accuracy);
    
    report2.close();
    image2.close();
    label2.close();

    return 1;
}

// OCALL implementations
void print_message(const char* str) {
    std::cout << str;
}

// OCALL2
void outputresult(int* predictions, size_t ndata3) {
    for (int pp = 0; pp < ndata3; ++pp) {
        int temp1 = predictions[pp];
    }
}

int main(int argc, char const *argv[]) {
    
    srand(time(NULL));
    
    int suc = 0;
    int q = atoi(argv[1]);
    
    auto start = chrono::high_resolution_clock::now();
    if (q == 0){
        suc = noSGXnn();
        }
    else if (q == 1){
        suc = SGXnn();    
        }
    else {
        cout << "Wrong input...";
        }
    auto stop = chrono::high_resolution_clock::now();
    
    chrono::duration<double> dura = stop - start;
    std::cout << dura.count();
    
    if (suc){
        return 0;
    }
    else {
        cout << "failed...";
        return 1;
    }

}
