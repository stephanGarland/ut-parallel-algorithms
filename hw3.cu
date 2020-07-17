#include <iostream>
#include <fstream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

using std::cout;
using std::endl;
using std::vector;

int main (void) {

    std::ifstream F;
    F.open("./inp.txt");
    int num;
    thrust::host_vector<int> host_nums;
    // This assumes the input is terminated with newline
    while ((F >> num) && (F.ignore())) {
        host_nums.push_back(num);
    }
    // Copy vector to GPU
    thrust::device_vector<int> A = host_nums;
	// Find minimum element using thrust, specifying it to be done on the GPU
    thrust::device_vector<int>::iterator iter = thrust::min_element(thrust::device, A.begin(), A.end());
    
    int pos = iter - A.begin();
    int minA = *iter;

	// Since we know the range to be 0-999, check that the returned min is 0
	assert(minA == 0);
	cout << "Min value is " << minA << " at position " << pos << endl;

	thrust::device_vector<int> B(A.size());
	thrust::fill(B.begin(), B.end(), A.back());
	
	cout << "Last value of A is: " << A.back() << endl;
	// Check that all elements of vector B equal each other
	assert(std::equal(B.begin() + 1, B.end(), B.begin()));

	// And that the two arrays are equal, given that the previous assertion is true
	assert(A.back() == B.back());
	cout << "Every value of B is: " << B.back() << endl;

}
