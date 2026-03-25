# MOSP CUDA-transition Makefile

CXX       := g++
NVCC      := nvcc

CXXFLAGS  := -std=c++17 -Wall -Wextra -Iheaders
NVCCFLAGS := -std=c++17 -Iheaders -arch=sm_70
LDFLAGS   := -lstdc++fs

SRCDIR    := src
BINDIR    := bin
BUILDDIR  := build

APP       := $(BINDIR)/main

CPP_SRCS  := \
	$(SRCDIR)/main.cpp \
	$(SRCDIR)/generateGraph.cpp \
	$(SRCDIR)/read.cpp \
	$(SRCDIR)/Dijkstra.cpp \
	$(SRCDIR)/csr.cpp

CU_SRCS   := \
	$(SRCDIR)/cuda_print.cu

CPP_OBJS  := $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRCS))
CU_OBJS   := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(CU_SRCS))

OBJS      := $(CPP_OBJS) $(CU_OBJS)

.PHONY: all clean run

all: $(APP)

$(BINDIR) $(BUILDDIR):
	mkdir -p $@

$(APP): $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(BINDIR) $(BUILDDIR)

run: $(APP)
	./$(APP)