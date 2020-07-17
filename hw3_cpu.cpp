#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main (void) {

    std::ifstream F;
    F.open("./inp.txt");
    int num;
    vector<int> A;
    // This assumes the input is terminated with newline
    while ((F >> num) && (F.ignore())) {
        A.push_back(num);
    }
    auto minA = *std::min_element(A.begin(), A.end());
    std::vector<int>::iterator iter = std::find(A.begin(), A.end(), minA);
    int pos = std::distance(A.begin(), iter);

    // Since we know the range to be 0-999, check that the returned min is 0
	assert(minA == 0);
	cout << "Min value is " << minA << " at position " << pos << endl;

    vector<int> B(A.size());
    std::fill(B.begin(), B.end(), A.back());
	
	cout << "Last value of A is: " << A.back() << endl;
	// Check that all elements of vector B equal each other
	assert(std::equal(B.begin() + 1, B.end(), B.begin()));

	// And that the two arrays are equal, given that the previous assertion is true
	assert(A.back() == B.back());
	cout << "Every value of B is: " << B.back() << endl;

}
