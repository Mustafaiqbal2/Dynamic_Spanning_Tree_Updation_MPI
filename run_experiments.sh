#!/bin/bash

# Directory containing graph datasets
DATASET_DIR="./datasets"
mkdir -p $DATASET_DIR

# List of number of ranks to test
RANKS=(8)
# RANKS=(1 2 4 8 16 32 64 128)
# List of thread counts per rank
THREADS=( 8)
# THREADS=(1 2 4 8 16 32 64)

# Source vertex
SOURCE=0

# Directory for results
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# Function to prepare datasets for our program
prepare_datasets() {
    echo "Converting datasets to appropriate formats..."
    
    # Process LiveJournal dataset - just creating CSR, no METIS files needed
    if [ -f "$DATASET_DIR/soc-LiveJournal1.txt" ] && [ ! -f "./graph.csr" ]; then
        echo "Processing soc-LiveJournal1.txt..."
        python3 -c "
import numpy as np
import os

# Parse edges from LiveJournal dataset - limit to 50K edges for testing
edges = []
vertices = set()
edge_limit = 50000  # Limit to prevent crashes during testing

print('Reading edges from LiveJournal dataset')
count = 0
with open('$DATASET_DIR/soc-LiveJournal1.txt', 'r') as f:
    for line in f:
        # Skip comment lines
        if line.startswith('#'):
            continue
        
        count += 1
        if count > edge_limit:
            break
            
        parts = line.strip().split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            vertices.add(u)
            vertices.add(v)

# Map vertices to consecutive IDs starting from 0
vertex_map = {v: i for i, v in enumerate(sorted(vertices))}
n_vertices = len(vertex_map)
n_edges = len(edges)

print(f'Creating graph with {n_vertices} vertices and {n_edges} edges')

# Create adjacency lists
adj_lists = [[] for _ in range(n_vertices)]
for u, v in edges:
    u_mapped = vertex_map[u]
    v_mapped = vertex_map[v]
    # Add both directions for undirected graph
    adj_lists[u_mapped].append(v_mapped)
    adj_lists[v_mapped].append(u_mapped)

# Create CSR format for our program
offsets = [0]
edges_flat = []
weights = []

for adj in adj_lists:
    offsets.append(offsets[-1] + len(adj))
    edges_flat.extend(adj)
    weights.extend([1.0] * len(adj))  # Unit weights

# Write CSR file
with open('./graph.csr', 'w') as f:
    f.write(f'{n_vertices} {len(edges_flat)}\\n')
    
    for offset in offsets:
        f.write(f'{offset} ')
    f.write('\\n')
    
    for edge in edges_flat:
        f.write(f'{edge} ')
    f.write('\\n')
    
    for weight in weights:
        f.write(f'{weight:.1f} ')
    f.write('\\n')

print('CSR file created successfully')
"
    fi
    
    # Generate changes file
    if [ -f "./graph.csr" ] && [ ! -f "./graph.changes" ]; then
        echo "Generating changes..."
        python3 -c "
import numpy as np

# Read the CSR file to get graph information
with open('./graph.csr', 'r') as f:
    firstline = f.readline().strip().split()
    n_vertices = int(firstline[0])
    n_edges = int(firstline[1])
    
    # Skip offsets and edges
    f.readline()  # Skip offsets
    f.readline()  # Skip edges

# Generate changes (1% of edges or at least 50, max 500)
n_changes = max(min(int(n_edges * 0.01), 500), 50)
print(f'Generating {n_changes} changes for graph with {n_vertices} vertices and {n_edges} edges')

# 70% insertions, 30% deletions
n_insert = int(n_changes * 0.7)
n_delete = n_changes - n_insert

# Generate random changes
changes = []

# Insertions
for _ in range(n_insert):
    u = np.random.randint(0, n_vertices)
    v = np.random.randint(0, n_vertices)
    if u != v:  # Avoid self-loops
        weight = np.random.uniform(0.1, 5.0)  # Random weight
        changes.append(('I', u, v, weight))
        
# Deletions
for _ in range(n_delete):
    u = np.random.randint(0, n_vertices)
    v = np.random.randint(0, n_vertices)
    if u != v:  # Avoid self-loops
        weight = 1.0  # Weight doesn't matter for deletions
        changes.append(('D', u, v, weight))

# Write changes to file
with open('./graph.changes', 'w') as f:
    f.write(f'{len(changes)}\\n')
    for op, u, v, w in changes:
        f.write(f'{op} {u} {v} {w:.6f}\\n')

print('Changes file created successfully')
"
    fi
    
    echo "All dataset preparations completed."
}

# Function to run weak scaling experiment
run_weak_scaling() {
    echo "Running weak scaling experiments..."
    for ranks in "${RANKS[@]}"; do
        for threads in "${THREADS[@]}"; do
            echo "Running with ranks=$ranks, threads=$threads"
            timeout 600 mpirun --oversubscribe -np $ranks --map-by socket:PE=$threads ./dssp_update \
                --graph ./graph.csr \
                --changes ./graph.changes \
                --source $SOURCE \
                --threads $threads \
                --mode mpi+openmp 2>&1 | tee -a $RESULTS_DIR/weak_scaling_${ranks}_${threads}.log

            # Add a short delay between runs
            sleep 2
        done
    done
}

# Function to run strong scaling experiment
run_strong_scaling() {
    echo "Running strong scaling experiments..."
    for ranks in "${RANKS[@]}"; do
        for threads in "${THREADS[@]}"; do
            echo "Running with ranks=$ranks, threads=$threads"
            timeout 600 mpirun -np $ranks ./dssp_update \
                --graph ./graph.csr \
                --changes ./graph.changes \
                --source $SOURCE \
                --threads $threads \
                --mode mpi+openmp 2>&1 | tee -a $RESULTS_DIR/strong_scaling_${ranks}_${threads}.log
            
            # Add a short delay between runs
            sleep 2
        done
    done
}

# Check for required dependencies
check_requirements() {
    echo "Checking required packages..."
    
    # Check for Python with NumPy
    if ! command -v python3 &> /dev/null || ! python3 -c "import numpy" &> /dev/null; then
        echo "Python3 with NumPy is required. Please install with:"
        echo "sudo apt-get install python3 python3-numpy"
        exit 1
    fi
    
    # Check for MPI
    if ! command -v mpirun &> /dev/null; then
        echo "MPI is required. Please install with:"
        echo "sudo apt-get install openmpi-bin libopenmpi-dev"
        exit 1
    fi
}

# Make sure the executable exists
check_executable() {
    if [ ! -f "./dssp_update" ]; then
        echo "Executable ./dssp_update not found. Building it now..."
        make
        if [ ! -f "./dssp_update" ]; then
            echo "Failed to build executable. Please check compilation errors."
            exit 1
        fi
    fi
}

# Main execution
check_requirements
check_executable
prepare_datasets

# Run experiments
if [ "$1" == "weak" ]; then
    run_weak_scaling
elif [ "$1" == "strong" ]; then
    run_strong_scaling
else
    echo "Usage: $0 [weak|strong]"
    echo "Running both by default..."
    run_weak_scaling
    run_strong_scaling
fi

echo "All experiments completed. Results saved in $RESULTS_DIR"