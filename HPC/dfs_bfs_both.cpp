#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <omp.h>

using namespace std;

// Graph class using adjacency list
class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    Graph(int vertices) : V(vertices) {
        adj.resize(V);
    }

    // Add an edge to the graph
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // Undirected graph
    }

    // Sequential BFS
    void sequentialBFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Sequential BFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            cout << node << " ";

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

    // Parallel BFS
    void parallelBFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        while (!q.empty()) {
            int levelSize = q.size(); // Number of nodes at current level

            #pragma omp parallel for
            for (int i = 0; i < levelSize; i++) {
                int node;
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                }

                #pragma omp critical
                {
                    cout << node << " ";
                }

                vector<int> neighbors;
                #pragma omp critical
                {
                    neighbors = adj[node];
                }

                for (int neighbor : neighbors) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

    // Sequential DFS
    void sequentialDFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        cout << "Sequential DFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        sequentialDFSUtil(start, visited);

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

    // Parallel DFS
    void parallelDFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        #pragma omp parallel
        {
            #pragma omp single
            {
                parallelDFSUtil(start, visited);
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

private:
    // Utility function for sequential DFS
    void sequentialDFSUtil(int node, vector<bool>& visited) {
        visited[node] = true;
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                sequentialDFSUtil(neighbor, visited);
            }
        }
    }

    // Utility function for parallel DFS
    void parallelDFSUtil(int node, vector<bool>& visited) {
        #pragma omp critical
        {
            if (!visited[node]) {
                visited[node] = true;
                cout << node << " ";
            }
        }

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                #pragma omp task
                {
                    parallelDFSUtil(neighbor, visited);
                }
            }
        }

        #pragma omp taskwait
    }
};

int main() {
    int V, E;

    // Input number of vertices and edges
    cout << "Enter the number of vertices: ";
    cin >> V;
    cout << "Enter the number of edges: ";
    cin >> E;

    // Create graph
    Graph g(V);

    // Input edges
    cout << "Enter " << E << " edges (u v pairs, 0-based indexing):\n";
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        if (u < 0 || u >= V || v < 0 || v >= V) {
            cout << "Invalid vertex index! Exiting.\n";
            return 1;
        }
        g.addEdge(u, v);
    }

    // Set number of threads
    omp_set_num_threads(4);

    // Variables to store execution times
    double seq_bfs_time, par_bfs_time, seq_dfs_time, par_dfs_time;

    // Run sequential BFS
    g.sequentialBFS(0, seq_bfs_time);

    // Run parallel BFS
    g.parallelBFS(0, par_bfs_time);

    // Run sequential DFS
    g.sequentialDFS(0, seq_dfs_time);

    // Run parallel DFS
    g.parallelDFS(0, par_dfs_time);

    // Output execution time comparison table
    cout << "\nExecution Time Comparison (in seconds):\n";
    cout << "+---------+------------------+------------------+----------+\n";
    cout << "| Algorithm | Sequential Time  | Parallel Time    | Speedup  |\n";
    cout << "+---------+------------------+------------------+----------+\n";
    cout << "| BFS     | " << fixed << setprecision(6) << seq_bfs_time << " | " << par_bfs_time << " | " << (seq_bfs_time / par_bfs_time) << " |\n";
    cout << "| DFS     | " << seq_dfs_time << " | " << par_dfs_time << " | " << (seq_dfs_time / par_dfs_time) << " |\n";
    cout << "+---------+------------------+------------------+----------+\n";

    return 0;
}