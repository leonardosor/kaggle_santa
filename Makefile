# Makefile for Kaggle Santa with Go integration

.PHONY: all build-go test-go test-integration clean help

# Default target
all: build-go test-go test-integration

# Build Go shared library
build-go:
	@echo "Building Go geometry library..."
	cd golib/cmd/libgeometry && go build -buildmode=c-shared -o libgeometry.so main.go
	@echo "✓ Go library built successfully"

# Run Go unit tests
test-go:
	@echo "Running Go unit tests..."
	cd golib && go test -v
	@echo "✓ Go tests passed"

# Run Go benchmarks
benchmark-go:
	@echo "Running Go benchmarks..."
	cd golib && go test -bench=. -benchmem
	@echo "✓ Benchmarks complete"

# Run Python-Go integration tests
test-integration: build-go
	@echo "Running Python-Go integration tests..."
	python3 golib/test_integration.py
	@echo "✓ Integration tests passed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f golib/cmd/libgeometry/libgeometry.so
	rm -f golib/cmd/libgeometry/libgeometry.h
	@echo "✓ Clean complete"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build Go library and run all tests (default)"
	@echo "  build-go         - Build Go shared library"
	@echo "  test-go          - Run Go unit tests"
	@echo "  benchmark-go     - Run Go benchmarks"
	@echo "  test-integration - Run Python-Go integration tests"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
