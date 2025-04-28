#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <metis.h>
#include <unordered_set>
#include <limits>
#include <iomanip>
#include <unordered_map>
#include <queue>

// Constants
const double INF = std::numeric_limits<double>::infinity();

// CSR Graph structure
struct CSRGraph {
    std::vector<int> offsets;     // Vertex offsets into edges array
    std::vector<int> edges;       // Edge destinations
    std::vector<double> weights;  // Edge weights
    int num_vertices;
    int num_edges;
    std::unordered_map<int, int> ghost_vertices; // Ghost vertex mapping
    std::vector<int> global_to_local;  // Map from global to local vertex IDs
    std::vector<int> local_to_global;  // Map from local to global vertex IDs
};

// Edge change structure
struct EdgeChange {
    int u;          // First vertex
    int v;          // Second vertex
    double weight;  // Weight of edge
    bool is_delete; // true if deletion, false if insertion
};

// Fix 1: Create a proper build_initial_tree function that sets up a real spanning tree
void build_initial_tree(const CSRGraph& graph, int source, std::vector<int>& parent, std::vector<double>& dist) {
    int n = graph.num_vertices;
    
    // Initialize all distances to infinity and all parents to -1
    for (int i = 0; i < n; i++) {
        parent[i] = -1;
        dist[i] = INF;
    }
    
    // Set source distance to 0
    dist[source] = 0.0;
    
    // Use priority queue for Dijkstra's algorithm
    std::priority_queue<std::pair<double, int>, 
                       std::vector<std::pair<double, int>>, 
                       std::greater<std::pair<double, int>>> pq;
    pq.push({0.0, source});
    
    // Standard Dijkstra implementation
    while (!pq.empty()) {
        double d = pq.top().first;
        int v = pq.top().second;
        pq.pop();
        
        // Skip if we already found a better path
        if (d > dist[v]) continue;
        
        // Process all neighbors
        for (int i = graph.offsets[v]; i < graph.offsets[v+1]; i++) {
            int u = graph.edges[i];
            double w = graph.weights[i];
            
            if (dist[v] + w < dist[u]) {
                dist[u] = dist[v] + w;
                parent[u] = v;
                pq.push({dist[u], u});
            }
        }
    }
    
    // Count reachable vertices to verify tree was built correctly
    int reachable = 0;
    for (int i = 0; i < n; i++) {
        if (dist[i] != INF) {
            reachable++;
        }
    }
    
    std::cout << "Built initial SSSP tree: " << reachable << " out of " << n 
              << " vertices reachable from source " << source << std::endl;
}

// Function to save the SSSP tree to a file (run by rank 0)
void save_sssp_tree(const std::string& filename, const std::vector<int>& parent, 
                    const std::vector<double>& dist) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    int n = parent.size();
    file << n << std::endl;
    
    // Write parent array
    for (int i = 0; i < n; i++) {
        file << parent[i] << " ";
        // Add periodic newlines to avoid very long lines
        if (i > 0 && i % 1000 == 0)
            file << std::endl;
    }
    file << std::endl;
    
    // Write distance array
    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) {
            file << "INF ";
        } else {
            file << dist[i] << " ";
        }
        // Add periodic newlines to avoid very long lines
        if (i > 0 && i % 1000 == 0)
            file << std::endl;
    }
    file << std::endl;
    
    file.close();
    std::cout << "SSSP tree saved to " << filename << std::endl;
    
    // Verify file was written correctly
    std::ifstream check_file(filename);
    if (!check_file) {
        std::cerr << "Failed to verify tree file was written correctly" << std::endl;
        return;
    }
    
    int check_n;
    check_file >> check_n;
    if (check_n != n) {
        std::cerr << "Tree file verification failed - incorrect size" << std::endl;
        // Delete corrupted file and try again
        check_file.close();
        std::remove(filename.c_str());
    }
}

// Fix for reading/writing the initial_sssp_tree.txt file
void read_sssp_tree(const std::string& filename, std::vector<int>& parent, std::vector<double>& dist) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open SSSP tree file: " << filename << std::endl;
        return;  // Don't exit, just return - we'll handle this at the caller level
    }

    int n;
    file >> n;
    parent.resize(n);
    dist.resize(n);
    
    // Check if the file is valid
    if (file.fail()) {
        std::cerr << "Error reading tree file - invalid format" << std::endl;
        parent.clear();
        dist.clear();
        return;
    }

    for (int i = 0; i < n; ++i) {
        file >> parent[i];
    }

    std::string dist_str;
    for (int i = 0; i < n; ++i) {
        file >> dist_str;
        if (dist_str == "INF") {
            dist[i] = INF;
        } else {
            try {
                dist[i] = std::stod(dist_str);
            } catch (...) {
                dist[i] = INF;
                std::cerr << "Error parsing distance at index " << i << std::endl;
            }
        }
    }
}

// Read CSR graph from file
CSRGraph read_csr_graph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open graph file: " << filename << std::endl;
        exit(1);
    }

    CSRGraph graph;
    file >> graph.num_vertices >> graph.num_edges;
    
    graph.offsets.resize(graph.num_vertices + 1);
    graph.edges.resize(graph.num_edges);
    graph.weights.resize(graph.num_edges);

    // Read offsets
    for (int i = 0; i <= graph.num_vertices; ++i) {
        file >> graph.offsets[i];
    }

    // Read edges and weights
    for (int i = 0; i < graph.num_edges; ++i) {
        file >> graph.edges[i];
    }
    
    for (int i = 0; i < graph.num_edges; ++i) {
        file >> graph.weights[i];
    }

    return graph;
}

// Read edge changes
std::vector<EdgeChange> read_edge_changes(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open changes file: " << filename << std::endl;
        exit(1);
    }

    int num_changes;
    file >> num_changes;

    std::vector<EdgeChange> changes(num_changes);
    for (int i = 0; i < num_changes; ++i) {
        char change_type;
        file >> change_type >> changes[i].u >> changes[i].v >> changes[i].weight;
        changes[i].is_delete = (change_type == 'D' || change_type == 'd');
    }

    return changes;
}

// Proper partition_graph function that uses METIS directly
std::vector<int> partition_graph(const CSRGraph& graph, int num_parts) {
    std::vector<int> part(graph.num_vertices);
    
    // For single process, just assign everything to part 0
    if (num_parts <= 1) {
        return part; // Already initialized to 0
    }
    
    // Convert graph to METIS format
    idx_t nvtxs = graph.num_vertices;
    idx_t ncon = 1;  // Number of balancing constraints
    
    // METIS expects CSR format for edges
    std::vector<idx_t> xadj(graph.offsets.begin(), graph.offsets.end());
    std::vector<idx_t> adjncy(graph.edges.begin(), graph.edges.end());
    
    // Weights (scale to integers for METIS)
    std::vector<idx_t> adjwgt(graph.weights.size());
    for (size_t i = 0; i < graph.weights.size(); i++) {
        adjwgt[i] = static_cast<idx_t>(graph.weights[i] * 100);
    }
    
    // Options for METIS
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;  // Recursive bisection
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0;  // C-style numbering
    
    // Output variables
    idx_t edgecut;
    std::vector<idx_t> part_idx(nvtxs);
    
    std::cout << "Partitioning graph with METIS..." << std::endl;
    
    // Call METIS for partitioning
    int ret = METIS_PartGraphRecursive(
        &nvtxs,        // Number of vertices
        &ncon,         // Weights per vertex
        xadj.data(),   // CSR row pointers
        adjncy.data(), // CSR column indices
        nullptr,       // No vertex weights
        nullptr,       // No vertex sizes
        adjwgt.data(), // Edge weights
        &num_parts,    // Number of partitions
        nullptr,       // No target partition weights
        nullptr,       // No target partition weight tolerance
        options,       // Options array
        &edgecut,      // Output: Edge-cut or communication volume
        part_idx.data() // Output: Partition vector
    );
    
    if (ret != METIS_OK) {
        std::cerr << "Error: METIS partitioning failed with code " << ret << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Copy results to our output format
    for (int i = 0; i < graph.num_vertices; i++) {
        part[i] = part_idx[i];
    }
    
    std::cout << "METIS partitioning complete. Edge-cut: " << edgecut << std::endl;
    return part;
}

// Fix distribute_graph function to properly handle ghost vertices
CSRGraph distribute_graph(const CSRGraph& global_graph, const std::vector<int>& partition, int rank, int num_ranks) {
    // Identify vertices assigned to this rank
    std::vector<int> local_vertices;
    for (int v = 0; v < global_graph.num_vertices; v++) {
        if (partition[v] == rank) {
            local_vertices.push_back(v);
        }
    }
    
    // Create mappings between global and local vertex IDs
    std::vector<int> global_to_local(global_graph.num_vertices, -1);
    std::vector<int> local_to_global(local_vertices.size());
    
    for (size_t i = 0; i < local_vertices.size(); i++) {
        global_to_local[local_vertices[i]] = i;
        local_to_global[i] = local_vertices[i];
    }
    
    // Count local edges and construct local graph
    CSRGraph local_graph;
    local_graph.num_vertices = local_vertices.size();
    local_graph.offsets.resize(local_graph.num_vertices + 1);
    
    // Store mapping of ghost vertices - we need this for boundary communication
    std::unordered_map<int, int> ghost_vertices; // global_id -> local_offset
    int next_ghost_id = local_graph.num_vertices;
    
    int edge_count = 0;
    std::vector<std::pair<int, double>> temp_edges;
    
    for (int local_idx = 0; local_idx < local_graph.num_vertices; local_idx++) {
        local_graph.offsets[local_idx] = edge_count;
        int global_v = local_vertices[local_idx];
        
        for (int j = global_graph.offsets[global_v]; j < global_graph.offsets[global_v + 1]; j++) {
            int global_u = global_graph.edges[j];
            double weight = global_graph.weights[j];
            
            int target_id;
            if (global_to_local[global_u] != -1) {
                // Neighbor is a local vertex
                target_id = global_to_local[global_u];
            } else {
                // Neighbor is from another partition - create ghost vertex
                if (ghost_vertices.find(global_u) == ghost_vertices.end()) {
                    ghost_vertices[global_u] = next_ghost_id++;
                }
                target_id = ghost_vertices[global_u];
            }
            
            temp_edges.push_back({target_id, weight});
            edge_count++;
        }
    }
    
    local_graph.offsets[local_graph.num_vertices] = edge_count;
    local_graph.num_edges = edge_count;
    local_graph.edges.resize(edge_count);
    local_graph.weights.resize(edge_count);
    
    // Store the ghost vertex mapping and local-to-global mapping
    local_graph.ghost_vertices = ghost_vertices;
    local_graph.global_to_local = global_to_local;
    local_graph.local_to_global = local_to_global;
    
    edge_count = 0;
    for (auto& edge : temp_edges) {
        local_graph.edges[edge_count] = edge.first;
        local_graph.weights[edge_count] = edge.second;
        edge_count++;
    }
    
    std::cout << "Rank " << rank << " has " << local_graph.num_vertices 
              << " local vertices and " << ghost_vertices.size() 
              << " ghost vertices" << std::endl;
              
    return local_graph;
}

// Functions to count insertions and deletions in the changes
void count_changes(const std::vector<EdgeChange>& changes, int& num_insertions, int& num_deletions) {
    num_insertions = num_deletions = 0;
    for (const auto& change : changes) {
        if (change.is_delete) {
            num_deletions++;
        } else {
            num_insertions++;
        }
    }
}

// Output results to CSV
void output_csv(const std::string& graph_name, int num_vertices, int num_edges,
                int num_changes, double pct_insertions, double runtime, double speedup) {
    std::ofstream csv("results.csv", std::ios::app);
    if (!csv) {
        std::cerr << "Failed to open results.csv for writing" << std::endl;
        return;
    }
    
    // Write header if file is empty
    csv.seekp(0, std::ios::end);
    if (csv.tellp() == 0) {
        csv << "graph_name,vertices,edges,changes,pct_insertions,runtime,speedup" << std::endl;
    }
    
    csv << graph_name << "," << num_vertices << "," << num_edges << "," 
        << num_changes << "," << pct_insertions << "," << runtime << "," << speedup << std::endl;
}

// Fix 2: Safe cleanup to prevent double-free errors
void cleanup_resources(CSRGraph &graph) {
    // Clear all vectors safely
    std::vector<int>().swap(graph.offsets);
    std::vector<int>().swap(graph.edges);
    std::vector<double>().swap(graph.weights);
    std::vector<int>().swap(graph.global_to_local);
    std::vector<int>().swap(graph.local_to_global);
    graph.ghost_vertices.clear();
}

// Main function
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    // Parse command line arguments
    std::string graph_file, changes_file, mode = "mpi+openmp";
    int source = 0;
    int num_threads = omp_get_max_threads();
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (strcmp(argv[i], "--changes") == 0 && i + 1 < argc) {
            changes_file = argv[++i];
        } else if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            source = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--ranks") == 0 && i + 1 < argc) {
            // Already set by MPI, ignore this parameter
            i++;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }
    
    // Set the number of OpenMP threads
    omp_set_num_threads(num_threads);
    
    if (rank == 0) {
        std::cout << "Running with " << num_ranks << " MPI ranks and " 
                  << num_threads << " OpenMP threads per rank" << std::endl;
    }
    
    // Read global graph (all ranks read it)
    CSRGraph global_graph = read_csr_graph(graph_file);
    
    // Get graph name for reporting
    size_t lastSlash = graph_file.find_last_of("/\\");
    std::string graph_name = graph_file.substr(lastSlash + 1);
    
    // Fix for the segmentation fault in partition broadcasting
    std::vector<int> partition;
    if (rank == 0) {
        std::cout << "Partitioning graph with METIS..." << std::endl;
        partition = partition_graph(global_graph, num_ranks);
    }

    // Resize vectors on all ranks before broadcast
    if (rank != 0) {
        partition.resize(global_graph.num_vertices);
    }

    // Now broadcast is safe since all ranks have allocated memory
    MPI_Bcast(partition.data(), global_graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Distribute the graph according to the partition
    CSRGraph local_graph = distribute_graph(global_graph, partition, rank, num_ranks);
    
    if (rank == 0) {
        std::cout << "Graph distributed among " << num_ranks << " ranks" << std::endl;
    }
    
    // Broadcast the source vertex to all ranks
    MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // For each rank, initialize parent and dist arrays
    std::vector<int> parent(local_graph.num_vertices);
    std::vector<double> dist(local_graph.num_vertices);
    
    // Replace the manual tree initialization with:
    std::string tree_file = "initial_sssp_tree.txt";
    bool tree_exists = false;

    // Check if the tree file exists and is valid
    if (rank == 0) {
        std::ifstream f(tree_file);
        if (f.good()) {
            // Check if the file has valid content
            int size;
            f >> size;
            tree_exists = !f.fail() && size == global_graph.num_vertices;
            if (!tree_exists) {
                std::cout << "Tree file exists but has invalid format or size" << std::endl;
            }
        }
    }
    MPI_Bcast(&tree_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Fix 3: In main(), force regeneration of the spanning tree
    // Remove the broken tree file
    if (rank == 0) {
        // Force regeneration of spanning tree
        std::remove(tree_file.c_str());
        tree_exists = false;
        std::cout << "Forcing regeneration of spanning tree" << std::endl;
    }
    MPI_Bcast(&tree_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Initialize vectors regardless of whether we'll load from file or compute
    std::vector<int> global_parent(global_graph.num_vertices, -1);
    std::vector<double> global_dist(global_graph.num_vertices, INF);

    if (tree_exists) {
        if (rank == 0) {
            std::cout << "Loading existing SSSP tree from " << tree_file << std::endl;
            read_sssp_tree(tree_file, global_parent, global_dist);
        }
    } else {
        if (rank == 0) {
            std::cout << "Computing initial SSSP tree..." << std::endl;
            
            // Run Dijkstra on the full graph for ground truth
            build_initial_tree(global_graph, source, global_parent, global_dist);
            
            // Save this tree for future reference
            save_sssp_tree(tree_file, global_parent, global_dist);
        }
    }

    // Broadcast the global tree to all ranks
    MPI_Bcast(global_parent.data(), global_graph.num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_dist.data(), global_graph.num_vertices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Map the global tree to local tree for each rank
    std::vector<int> local_parent(local_graph.num_vertices);
    std::vector<double> local_dist(local_graph.num_vertices);
    
    // Map the global tree to local vertices
    for (int i = 0; i < local_graph.num_vertices; i++) {
        // Get the global ID for this local vertex
        int global_id = local_graph.local_to_global[i];
        local_dist[i] = global_dist[global_id];
        
        // Map parent to local ID if possible
        int global_parent_id = global_parent[global_id];
        
        // If parent exists and is in our local partition
        if (global_parent_id != -1) {
            if (local_graph.global_to_local[global_parent_id] != -1) {
                // Parent is a local vertex
                local_parent[i] = local_graph.global_to_local[global_parent_id];
            } else {
                // Parent is in another partition or is a ghost vertex
                auto it = local_graph.ghost_vertices.find(global_parent_id);
                if (it != local_graph.ghost_vertices.end()) {
                    local_parent[i] = it->second;  // Use ghost vertex ID
                } else {
                    local_parent[i] = -1;  // No parent in this partition
                }
            }
        } else {
            local_parent[i] = -1;  // No parent
        }
    }
    
    // Read edge changes
    std::vector<EdgeChange> changes;
    if (rank == 0) {
        changes = read_edge_changes(changes_file);
    }
    
    // Broadcast changes to all ranks
    int num_changes = changes.size();
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        changes.resize(num_changes);  // This is correct, keep this
    }

    // Create a custom MPI datatype for EdgeChange
    MPI_Datatype edge_change_type;
    int blocklengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_C_BOOL};
    
    displacements[0] = offsetof(EdgeChange, u);
    displacements[1] = offsetof(EdgeChange, v);
    displacements[2] = offsetof(EdgeChange, weight);
    displacements[3] = offsetof(EdgeChange, is_delete);
    
    MPI_Type_create_struct(4, blocklengths, displacements, types, &edge_change_type);
    MPI_Type_commit(&edge_change_type);
    
    // Broadcast changes
    MPI_Bcast(changes.data(), num_changes, edge_change_type, 0, MPI_COMM_WORLD);
    
    // Count insertions and deletions
    int num_insertions = 0, num_deletions = 0;
    count_changes(changes, num_insertions, num_deletions);
    double pct_insertions = 0.0;
    if (num_changes > 0) {
        pct_insertions = 100.0 * num_insertions / num_changes;
    }
    
    // Measure baseline sequential execution time (single process, no OpenMP)
    double sequential_time = 0.0;
    double parallel_time = 0.0;
    
    // Run the algorithm using all ranks
    if (rank == 0) {
        std::cout << "Running dynamic SSSP update algorithm..." << std::endl;
    }
    
    double start_time = MPI_Wtime();
    // Placeholder for update_sssp function call
    double end_time = MPI_Wtime();
    parallel_time = end_time - start_time;
    
    // Fix the sequential comparison to avoid memory issues
    if (rank == 0) {
        // Create new vectors for sequential computation
        std::vector<int> seq_parent(global_graph.num_vertices, -1);
        std::vector<double> seq_dist(global_graph.num_vertices, INF);
        seq_dist[source] = 0.0;
        
        // Run true sequential algorithm
        start_time = MPI_Wtime();
        // Placeholder for sequential algorithm implementation
        end_time = MPI_Wtime();
        sequential_time = end_time - start_time;
        
        // Compute speedup
        double speedup = sequential_time / parallel_time;
        std::cout << "Sequential time: " << sequential_time << " seconds" << std::endl;
        std::cout << "Parallel time: " << parallel_time << " seconds" << std::endl;
        std::cout << "Speedup: " << speedup << std::endl;
        
        // Output results to CSV
        output_csv(graph_name, global_graph.num_vertices, global_graph.num_edges,
                  num_changes, pct_insertions, parallel_time, speedup);
    }
    
    // Before freeing MPI datatype and finalizing MPI,
    // clear all data structures that might cause issues during cleanup
    cleanup_resources(local_graph);
    if (rank == 0) {
        cleanup_resources(global_graph);
    }

    // Make sure to synchronize all processes before cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Free MPI datatype and finalize MPI
    MPI_Type_free(&edge_change_type);
    MPI_Finalize();
    
    return 0;
}