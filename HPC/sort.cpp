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
    uniform_int_distribution<> dis(1, 100000); // Random numbers between 1 and 1000
    for (int i = 0; i < n; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

// Sequential Bubble Sort
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; ++i) {
        // Odd-even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
        // Even-odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Merge function for Merge Sort
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left, k = 0; i <= right; ++i, ++k) {
        arr[i] = temp[k];
    }
}

// Sequential Merge Sort
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                parallelMergeSort(arr, left, mid);
                #pragma omp task
                parallelMergeSort(arr, mid + 1, right);
                #pragma omp taskwait
            }
        }
        merge(arr, left, mid, right);
    }
}

int main() {
    int n;
    char choice;

    // Input array length
    cout << "Enter the length of the array: ";
    cin >> n;

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

    // Create copies of the array for each sorting algorithm
    vector<int> arr_seq_bubble = arr;
    vector<int> arr_par_bubble = arr;
    vector<int> arr_seq_merge = arr;
    vector<int> arr_par_merge = arr;

    // Set number of threads
    omp_set_num_threads(4);

    // Measure execution times
    double start_time, end_time;
    double seq_bubble_time, par_bubble_time, seq_merge_time, par_merge_time;

    // Sequential Bubble Sort
    start_time = omp_get_wtime();
    sequentialBubbleSort(arr_seq_bubble);
    end_time = omp_get_wtime();
    seq_bubble_time = end_time - start_time;

    // Parallel Bubble Sort
    start_time = omp_get_wtime();
    parallelBubbleSort(arr_par_bubble);
    end_time = omp_get_wtime();
    par_bubble_time = end_time - start_time;

    // Sequential Merge Sort
    start_time = omp_get_wtime();
    sequentialMergeSort(arr_seq_merge, 0, n - 1);
    end_time = omp_get_wtime();
    seq_merge_time = end_time - start_time;

    // Parallel Merge Sort
    start_time = omp_get_wtime();
    parallelMergeSort(arr_par_merge, 0, n - 1);
    end_time = omp_get_wtime();
    par_merge_time = end_time - start_time;

    // Output results in a table
    cout << "\nExecution Time Comparison (in seconds):\n";
    cout << "+-------------------+------------------+------------------+\n";
    cout << "| Algorithm         | Sequential Time  | Parallel Time    |\n";
    cout << "+-------------------+------------------+------------------+\n";
    cout << "| Bubble Sort       | " << fixed << setprecision(6) << seq_bubble_time << " | " << par_bubble_time << " |\n";
    cout << "| Merge Sort        | " << seq_merge_time << " | " << par_merge_time << " |\n";
    cout << "+-------------------+------------------+------------------+\n";

    return 0;
}