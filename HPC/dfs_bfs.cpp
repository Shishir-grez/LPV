#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <atomic>
#include <random>
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

        int count = 0; // Limit output to 10 nodes
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            if (count < 10) {
                cout << node << " ";
                count++;
            } else if (count == 10) {
                cout << "...";
                count++;
            }

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

        int count = 0; // Limit output to 10 nodes
        while (!q.empty()) {
            int levelSize = q.size(); // Number of nodes at current level
            vector<int> level_nodes(levelSize);
            
            // Extract all nodes at current level
            for (int i = 0; i < levelSize; i++) {
                level_nodes[i] = q.front();
                q.pop();
                
                if (count < 10) {
                    cout << level_nodes[i] << " ";
                    count++;
                } else if (count == 10) {
                    cout << "...";
                    count++;
                }
            }
            
            // Process all nodes at current level in parallel
            vector<int> next_frontier;
            #pragma omp parallel for
            for (int i = 0; i < levelSize; i++) {
                int node = level_nodes[i];
                vector<int> local_next;
                
                for (int neighbor : adj[node]) {
                    bool not_visited = false;
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            not_visited = true;
                        }
                    }
                    
                    if (not_visited) {
                        local_next.push_back(neighbor);
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
                }
            }
            
            // Add next level nodes to queue
            for (int node : next_frontier) {
                q.push(node);
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

    // Sequential DFS - Iterative implementation to avoid stack overflow
    void sequentialDFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        stack<int> s;
        
        cout << "Sequential DFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing
        
        s.push(start);
        int count = 0; // Limit output to 10 nodes
        
        while (!s.empty()) {
            int node = s.top();
            s.pop();
            
            if (!visited[node]) {
                visited[node] = true;
                
                if (count < 10) {
                    cout << node << " ";
                    count++;
                } else if (count == 10) {
                    cout << "...";
                    count++;
                }
                
                // Push neighbors in reverse order to maintain similar traversal to recursive DFS
                for (int i = adj[node].size() - 1; i >= 0; --i) {
                    int neighbor = adj[node][i];
                    if (!visited[neighbor]) {
                        s.push(neighbor);
                    }
                }
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

    // Parallel DFS with task-based parallelism
    void parallelDFS(int start, double& execution_time) {
        vector<bool> visited(V, false);
        cout << "Parallel DFS starting from vertex " << start << ": ";

        double start_time = omp_get_wtime(); // Start timing

        // Use a global counter instead of passing atomic by reference
        int global_count = 0;
        
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Mark start as visited
                visited[start] = true;
                
                // Print the start node
                cout << start << " ";
                global_count = 1;
                
                // Process the first level of neighbors using tasks
                // This avoids passing the atomic counter by reference
                for (int neighbor : adj[start]) {
                    #pragma omp task firstprivate(neighbor)
                    {
                        stack<int> s;
                        s.push(neighbor);
                        
                        while (!s.empty()) {
                            int current = s.top();
                            s.pop();
                            
                            bool is_new_node = false;
                            #pragma omp critical(visited)
                            {
                                if (!visited[current]) {
                                    visited[current] = true;
                                    is_new_node = true;
                                }
                            }
                            
                            if (is_new_node) {
                                int node_count;
                               #pragma omp atomic capture
                                node_count = global_count++;
                                
                                if (node_count < 10) {
                                    #pragma omp critical(cout)
                                    {
                                        cout << current << " ";
                                    }
                                } else if (node_count == 10) {
                                    #pragma omp critical(cout)
                                    {
                                        cout << "...";
                                    }
                                }
                                
                                // Add unvisited neighbors to stack
                                vector<int> unvisited;
                                for (int next : adj[current]) {
                                    bool not_visited = false;
                                    #pragma omp critical(visited)
                                    {
                                        not_visited = !visited[next];
                                    }
                                    
                                    if (not_visited) {
                                        unvisited.push_back(next);
                                    }
                                }
                                
                                // Push neighbors in reverse order to match sequential DFS traversal order
                                for (int i = unvisited.size() - 1; i >= 0; --i) {
                                    s.push(unvisited[i]);
                                }
                            }
                        }
                    }
                }
                
                // Wait for all tasks to complete
                #pragma omp taskwait
            }
        }

        double end_time = omp_get_wtime(); // End timing
        execution_time = end_time - start_time;

        cout << endl;
    }

private:
};

// Function to generate a large random graph
void generateLargeGraph(Graph& g, int V) {
    // Ensure connectivity with a spanning tree (chain structure)
    for (int i = 1; i < V; ++i) {
        g.addEdge(i, i - 1); // Connect vertex i to i-1
    }

    // Add additional random edges to reach ~10*V edges
    int target_edges = 10 * V;
    int current_edges = V - 1; // From spanning tree
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, V - 1);

    // Use a set to avoid duplicate edges
    vector<vector<bool>> edge_exists(V, vector<bool>(V, false));
    for (int i = 0; i < V; ++i) edge_exists[i][i] = true; // No self-loops
    for (int i = 1; i < V; ++i) {
        edge_exists[i][i - 1] = edge_exists[i - 1][i] = true; // Spanning tree edges
    }

    while (current_edges < target_edges) {
        int u = dis(gen);
        int v = dis(gen);
        if (u != v && !edge_exists[u][v]) {
            g.addEdge(u, v);
            edge_exists[u][v] = edge_exists[v][u] = true;
            current_edges++;
        }
    }
}

int main() {
    int V, E;
    char choice;

    // Input choice for random or manual graph
    cout << "Generate random large graph? (y/n): ";
    cin >> choice;

    Graph* g;
    if (choice == 'y' || choice == 'Y') {
        cout << "Enter number of vertices for random graph (recommended: 50-5000): ";
        cin >> V;
        g = new Graph(V);
        generateLargeGraph(*g, V);
        E = 10 * V; // Approximate number of edges
        cout << "Generated graph with " << V << " vertices and ~" << E << " edges\n";
    } else {
        // Manual input
        cout << "Enter the number of vertices: ";
        cin >> V;
        cout << "Enter the number of edges: ";
        cin >> E;

        g = new Graph(V);

        cout << "Enter " << E << " edges (u v pairs, 0-based indexing):\n";
        for (int i = 0; i < E; ++i) {
            int u, v;
            cin >> u >> v;
            if (u < 0 || u >= V || v < 0 || v >= V) {
                cout << "Invalid vertex index! Exiting.\n";
                return 1;
            }
            g->addEdge(u, v);
        }
    }

    // Set number of threads
    int num_threads = omp_get_max_threads();
    cout << "Using " << num_threads << " threads\n";
    omp_set_num_threads(num_threads);

    // Variables to store execution times
    double seq_bfs_time, par_bfs_time, seq_dfs_time, par_dfs_time;

    // Run sequential BFS
    g->sequentialBFS(0, seq_bfs_time);

    // Run parallel BFS
    g->parallelBFS(0, par_bfs_time);

    // Run sequential DFS
    g->sequentialDFS(0, seq_dfs_time);

    // Run parallel DFS
    g->parallelDFS(0, par_dfs_time);

    // Output execution time comparison table
    cout << "\nExecution Time Comparison (in seconds):\n";
    cout << "+---------+------------------+------------------+----------+\n";
    cout << "| Algorithm | Sequential Time  | Parallel Time    | Speedup  |\n";
    cout << "+---------+------------------+------------------+----------+\n";
double bfs_speedup = (par_bfs_time > 0) ? (seq_bfs_time / par_bfs_time) : 0.0;
double dfs_speedup = (par_dfs_time > 0) ? (seq_dfs_time / par_dfs_time) : 0.0;

cout << "| BFS     | " << fixed << setprecision(6) << seq_bfs_time << " | " << par_bfs_time << " | " << bfs_speedup << " |\n";
cout << "| DFS     | " << seq_dfs_time << " | " << par_dfs_time << " | " << dfs_speedup << " |\n";
    cout << "+---------+------------------+------------------+----------+\n";

    delete g;
    return 0;
}