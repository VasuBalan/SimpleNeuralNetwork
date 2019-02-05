/****************************************************************
*    			Learning Neural Network 						*
*																*
*																*
* Date of Written : 05.02.19 		Last Modified : 05.02.19	*
* Mailid: vasubdevan@yahoo.com									*
*****************************************************************/																* 

/* Simple Program to understand the concepts of parameters, variables 
   and optimizer in neural network. 
   Ref :
   https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%201b%20-%20Deep%20Neural%20Networks%20Are%20Our%20Friends.pdf
   */

#include <iostream>
#define MAXLOOP 10

using namespace std;

class Network {
	public:
	    double BestWeight; // Final Weight value
		double BestBias;	// Final bias value
		double OptimalCost;	// Final Optimal cost
		int* ComputeActualoutput(int*, int*, int, int, int);
		int* InitializeIntPtr(int*, int);
		double* InitializeDoublePtr(double*, int); 
		double ComputeCost(double*, int*, int*, int);
};
int* Network::ComputeActualoutput(int* A, int* B, int n, int W, int b) {
  int i;
  int *op = NULL;
  op = new int[n];
  
  for(i = 0; i < n; i++) {
	op[i] = W*A[i] + b;  
  }
  	
  return op;
}
int* Network::InitializeIntPtr(int* A, int n) {
  int i;
  for(i = 0; i < n; i++)
	A[i] = 0;

  return A;	
}
double* Network::InitializeDoublePtr(double* A, int n) {
  int i;
  for(i = 0; i < n; i++)
	A[i] = 0.0;

  return A;	
}
double Network::ComputeCost(double* Cost, int* A, int* T, int n) {
  // Input arguments are as follows Cost, Actual output, Target output, and noip.
  int i;
  double cos;
  cos = 0.0;
  
  for(i = 0; i < n; i++) {
    cos += ((A[i]-T[i])*(A[i]-T[i]));	  
  }
  return cos;	
}

// ******** Main program ********
 
int main(int argc, char* argv[]) {
	
  int noip, i, loop; // no of input variable
  noip = 3;
  
  int* input = NULL;
  int* Target = NULL; // target output
  int* Actual = NULL;
  double* weight = NULL;
  double *bias = NULL;
  double* Cost = NULL; // Cost function 
  
  input = new int[noip];
  Target = new int[noip]; 
  Actual = new int[noip]; 
  Cost = new double[MAXLOOP];
  weight = new double[MAXLOOP];
  bias = new double[MAXLOOP];
      
  cout << "\nWelcome to Neural network\n\n";
  
  input[0] = 1, input[1] = 5, input[2]=6;
  Target[0] = 0, Target[1]=16, Target[2]=20;
  
  Network Ntk;
  
  int index;
  double OptBias, OptWgt, OptCost;
    
  unsigned wt = 1;
  unsigned int bs = 0;
  double cos;
  weight = Ntk.InitializeDoublePtr(weight, MAXLOOP);
  bias = Ntk.InitializeDoublePtr(bias, MAXLOOP);
  
loop = 0;
    do {
	Actual = Ntk.InitializeIntPtr(Actual, noip);  
	Actual = Ntk.ComputeActualoutput(input, Target, noip, wt, bs);
  		
	cos = 0.0;
	cos = Ntk.ComputeCost(Cost, Actual, Target, noip);	
	Cost[loop] = cos;
	//cout << "Weight is " << wt << "\t" << "Bias is " << bs << "\t"; 
	cout << "Cost("<< wt<< ", "  << bs <<") is \t" << Cost[loop] << endl;
	weight[loop] = wt;
	bias[loop] = bs;
	wt += 1;
	bs += 0;
	loop += 1;
  } while(loop < MAXLOOP);
   
  // Determine the optimal among the existing cost values
  OptCost = Cost[0];
  for(i = 1; i < MAXLOOP; i++) {
	//cout << "Cost[" << i << "] is \t" << Cost[i] << "\n";
	if(Cost[i] < OptCost) {
	  OptCost = Cost[i];
	  index = i;
	}
  }
  OptWgt = weight[index];
  OptBias = bias[index];
  
  cout << "\n Optimal Cost is " << OptCost << "\t Index is " << index;
  // cout << "\nOptimal Weight is " << OptWgt << "\t Optimal Bias is" << OptBias << "\n\n";
  
  // ******************** Simple problem finished ******************** //
  
  double Uwt;
  double Ubs;
  
  Uwt = OptWgt;
  Ubs = OptBias;
  
  Cost = Ntk.InitializeDoublePtr(Cost, MAXLOOP);
  loop = 0;
    do {
	Actual = Ntk.InitializeIntPtr(Actual, noip);  
	Actual = Ntk.ComputeActualoutput(input, Target, noip, Uwt, Ubs);
  		
	cos = 0.0;
	cos = Ntk.ComputeCost(Cost, Actual, Target, noip);	
	Cost[loop] = cos;
	//cout << "Weight is " << wt << "\t" << "Bias is " << bs << "\t"; 
	cout << "Cost("<< Uwt<< ", "  << Ubs <<") is \t" << Cost[loop] << endl;
	weight[loop] = Uwt;
	bias[loop] = Ubs;
	Uwt += 0.2;
	Ubs -= 1;
	loop += 1;
  } while(loop < MAXLOOP);
   
  // Determine the optimal among the existing cost values
  OptCost = Cost[0];
  for(i = 1; i < MAXLOOP; i++) {
	//cout << "Cost[" << i << "] is \t" << Cost[i] << "\n";
	if(Cost[i] < OptCost) {
	  OptCost = Cost[i];
	  index = i;
	}
  }
  OptWgt = weight[index];
  OptBias = bias[index];
  
  cout << "\n Optimal Cost is " << OptCost << "\t Index is " << index;
  cout << "\nOptimal Weight is " << OptWgt << "\t Optimal Bias is" << OptBias << "\n\n";
  
  return 0;
}