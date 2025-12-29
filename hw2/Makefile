CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3

hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -O3 -fopenmp -msse4.2

CXXFLAGS = $(CFLAGS)
TARGETS = hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
		rm -f $(TARGETS) $(TARGETS:=.o)