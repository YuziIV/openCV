# Compiler settings
CXX = g++
CXXFLAGS = -Wall -std=c++11 `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`

# Build target
TARGET = assignment

all: $(TARGET)

$(TARGET): assignment.cpp
	$(CXX) $(CXXFLAGS) assignment.cpp -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
