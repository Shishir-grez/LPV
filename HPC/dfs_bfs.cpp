#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>  // Added for setprecision
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

    // Parallel BFS
    void parallelBFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        queue<int> q;

        // Initialize BFS
        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        while (!q.empty()) {
            int levelSize = q.size(); // Number of nodes at current level

            // Process all nodes at the current level in parallel
            #pragma omp parallel for
            for (int i = 0; i < levelSize; i++) {
                int node;
                // Critical section to safely dequeue a node
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                }

                // Print the node (in parallel, order may vary)
                #pragma omp critical
                {
                    cout << node << " ";
                }

                // Explore neighbors
                vector<int> neighbors;
                // Copy neighbors to local vector to avoid shared access
                #pragma omp critical
                {
                    neighbors = adj[node];
                }

                // Process neighbors
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

    // Parallel DFS using tasks
    void parallelDFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        // Start DFS from the root node
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
    // Utility function for parallel DFS
    void parallelDFSUtil(int node, vector<bool>& visited) {
        // Mark node as visited and print it
        #pragma omp critical
        {
            if (!visited[node]) {
                visited[node] = true;
                cout << node << " ";
            }
        }

        // Process neighbors
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                // Create a task for each unvisited neighbor
                #pragma omp task
                {
                    parallelDFSUtil(neighbor, visited);
                }
            }
        }

        // Wait for all tasks in this subtree to complete
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
    double bfs_time, dfs_time;

    // Run parallel BFS
    g.parallelBFS(0, bfs_time);

    // Run parallel DFS
    g.parallelDFS(0, dfs_time);

    // Output execution times
    cout << "\nExecution Times:\n";
    cout << "Parallel BFS: " << fixed << setprecision(6) << bfs_time << " seconds\n";
    cout << "Parallel DFS: " << dfs_time << " seconds\n";

    return 0;
}
