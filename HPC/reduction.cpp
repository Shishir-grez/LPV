#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <iomanip>

using namespace std;

// Function to generate a random array
vector<int> generateRandomArray(int n) {
    vector<int> arr(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 100000); 
    for (int i = 0; i < n; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

// Sequential Min
int sequentialMin(const vector<int>& arr) {
    int min_val = arr[0];
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Sequential Max
int sequentialMax(const vector<int>& arr) {
    int max_val = arr[0];
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Sequential Sum
long long sequentialSum(const vector<int>& arr) {
    long long sum = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// Sequential Average
double sequentialAverage(const vector<int>& arr) {
    long long sum = sequentialSum(arr);
    return static_cast<double>(sum) / arr.size();
}

// Parallel Min
int parallelMin(const vector<int>& arr) {
    int min_val = arr[0];
    #pragma omp parallel for reduction(min:min_val)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    return min_val;
}

// Parallel Max
int parallelMax(const vector<int>& arr) {
    int max_val = arr[0];
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}

// Parallel Sum
long long parallelSum(const vector<int>& arr) {
    long long sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < arr.size(); ++i) {
        sum += arr[i];
    }
    return sum;
}

// Parallel Average
double parallelAverage(const vector<int>& arr) {
    long long sum = parallelSum(arr);
    return static_cast<double>(sum) / arr.size();
}

int main() {
    int n;
    char choice;

    // Input array length
    cout << "Enter the length of the array: ";
    cin >> n;
    if (n <= 0) {
        cerr << "Invalid array length!" << endl;
        return 1;
    }

    // Input choice for random or manual input
    cout << "Generate random array? (y/n): ";
    cin >> choice;

    vector<int> arr;
    if (choice == 'y' || choice == 'Y') {
        arr = generateRandomArray(n);
    } else {
        arr.resize(n);
        cout << "Enter " << n << " elements: ";
        for (int i = 0; i < n; ++i) {
            cin >> arr[i];
        }
    }

    // Set number of threads
    omp_set_num_threads(4);

    // Variables to store results and times
    int seq_min, par_min, seq_max, par_max;
    long long seq_sum, par_sum;
    double seq_avg, par_avg;
    double seq_min_time, par_min_time, seq_max_time, par_max_time;
    double seq_sum_time, par_sum_time, seq_avg_time, par_avg_time;

    // Sequential Min
    double start_time = omp_get_wtime();
    seq_min = sequentialMin(arr);
    seq_min_time = omp_get_wtime() - start_time;

    // Parallel Min
    start_time = omp_get_wtime();
    par_min = parallelMin(arr);
    par_min_time = omp_get_wtime() - start_time;

    // Sequential Max
    start_time = omp_get_wtime();
    seq_max = sequentialMax(arr);
    seq_max_time = omp_get_wtime() - start_time;

    // Parallel Max
    start_time = omp_get_wtime();
    par_max = parallelMax(arr);
    par_max_time = omp_get_wtime() - start_time;

    // Sequential Sum
    start_time = omp_get_wtime();
    seq_sum = sequentialSum(arr);
    seq_sum_time = omp_get_wtime() - start_time;

    // Parallel Sum
    start_time = omp_get_wtime();
    par_sum = parallelSum(arr);
    par_sum_time = omp_get_wtime() - start_time;

    // Sequential Average
    start_time = omp_get_wtime();
    seq_avg = sequentialAverage(arr);
    seq_avg_time = omp_get_wtime() - start_time;

    // Parallel Average
    start_time = omp_get_wtime();
    par_avg = parallelAverage(arr);
    par_avg_time = omp_get_wtime() - start_time;

    // Output results
    cout << "\nResults:\n";
    cout << "Min: Sequential = " << seq_min << ", Parallel = " << par_min << endl;
    cout << "Max: Sequential = " << seq_max << ", Parallel = " << par_max << endl;
    cout << "Sum: Sequential = " << seq_sum << ", Parallel = " << par_sum << endl;
    cout << "Average: Sequential = " << fixed << setprecision(2) << seq_avg << ", Parallel = " << par_avg << endl;

    // Output time comparison table
    cout << "\nExecution Time Comparison (in seconds):\n";
    cout << "+---------+------------------+------------------+----------+\n";
    cout << "| Operation | Sequential Time  | Parallel Time    | Speedup  |\n";
    cout << "+---------+------------------+------------------+----------+\n";
    cout << "| Min     | " << fixed << setprecision(6) << seq_min_time << " | " << par_min_time << " | " << (seq_min_time / par_min_time) << " |\n";
    cout << "| Max     | " << seq_max_time << " | " << par_max_time << " | " << (seq_max_time / par_max_time) << " |\n";
    cout << "| Sum     | " << seq_sum_time << " | " << par_sum_time << " | " << (seq_sum_time / par_sum_time) << " |\n";
    cout << "| Average | " << seq_avg_time << " | " << par_avg_time << " | " << (seq_avg_time / par_avg_time) << " |\n";
    cout << "+---------+------------------+------------------+----------+\n";

    return 0;
}