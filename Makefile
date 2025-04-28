CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -fopenmp -O3
METIS_DIR = /usr/local

# Try to detect METIS installation
ifneq ($(wildcard $(METIS_DIR)/include/metis.h),)
	METIS_INCLUDE = -I$(METIS_DIR)/include
	METIS_LIB = -L$(METIS_DIR)/lib -lmetis
else ifneq ($(wildcard /usr/include/metis.h),)
	METIS_INCLUDE = -I/usr/include
	METIS_LIB = -lmetis
else ifneq ($(wildcard /usr/local/include/metis.h),)
	METIS_INCLUDE = -I/usr/local/include
	METIS_LIB = -L/usr/local/lib -lmetis
else
	$(error METIS library not found. Please install METIS or specify METIS_DIR)
endif

INCLUDES = $(METIS_INCLUDE)
LIBS = $(METIS_LIB) -lm

# Target executable name
TARGET = dssp_update

# Source files
SRC = main.cpp

# Object files
OBJ = $(SRC:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

# Create directories for results and scripts if they don't exist
setup:
	mkdir -p results
	mkdir -p datasets
	mkdir -p scripts
	chmod +x scripts/*.sh

# Run experiments
run-weak:
	./run_experiments.sh weak

run-strong:
	./run_experiments.sh strong

.PHONY: all clean setup run-weak run-strong