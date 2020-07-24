#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using std::cout;
using std::endl;

int main (void) {

    std::ifstream inp;
    inp.open("./inp.txt");
    std::vector<int> A;
    int num;
    while ((inp >> num) && inp.ignore()) {
        A.push_back(num);
    }
    // inp.ignore() defaults to EOF, and since the example file doesn't include a \n, add the last number
    // But just in case the test does include a newline, don't add anything
    inp.seekg(-1, std::ios_base::end);
    char c;
    if (c != '\n') {
        A.push_back(num);
    }

    auto minA = *std::min_element(A.begin(), A.end());
    std::vector<int>::iterator iter = std::find(A.begin(), A.end(), minA);
    int pos = std::distance(A.begin(), iter);

    // Since we know the range to be 0-999, check that the returned min is 0
	assert(minA == 0);
	cout << "Min value is " << minA << " at position " << pos << endl;

    std::vector<int> B(A.size());
    std::fill(B.begin(), B.end(), A.back());
	
	cout << "Last value of A is: " << A.back() << endl;
	// Check that all elements of vector B equal each other
	assert(std::equal(B.begin() + 1, B.end(), B.begin()));

	// And that the two arrays are equal, given that the previous assertion is true
	assert(A.back() == B.back());
	cout << "Every value of B is: " << B.back() << endl;

}
