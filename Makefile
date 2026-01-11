# Makefile for hw2_sift project
# Compiler and flags
MPICXX = mpicxx
MPIFLAGS = -std=c++17 -O3 -fopenmp
INCLUDES = -I.

# Source files
SOURCES = hw2.cpp sift.cpp image.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = hw2

# STB image library files
STB_HEADERS = stb_image.h stb_image_write.h

# Default target
all: $(TARGET)

# Build targets
$(TARGET): $(OBJECTS)
	$(MPICXX) $(MPIFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files to object files
hw2.o: hw2.cpp sift.hpp image.hpp
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c hw2.cpp

sift.o: sift.cpp sift.hpp image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c sift.cpp

image.o: image.cpp image.hpp $(STB_HEADERS)
	$(MPICXX) $(MPIFLAGS) $(INCLUDES) -c image.cpp

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(MPI_OBJECTS) $(TARGET) result.jpg

.PHONY: all clean
