# Define the base directory for your project
PROJECT_DIR := $(CURDIR)

# Default target to build the program
all: build

# Build target for a specific day and program
build: $(PROJECT_DIR)/day$(day)/$(program).out

$(PROJECT_DIR)/day$(day)/$(program).out: $(PROJECT_DIR)/day$(day)/$(program).cu
	@echo "Building program $(program) for day$(day)..."
	@nvcc -o $@ $<
	@echo "Build completed for $(program).out"

# Target to run the program (removes .cu extension, adds .out)
run: $(PROJECT_DIR)/day$(day)/$(program).out
	@echo "Running $(program).out..."
	@./day$(day)/$(program).out

# Target to clean the .out files from a specific day
clean:
	@echo "Cleaning up .out files for day$(day)..."
	@rm -f $(PROJECT_DIR)/day$(day)/*.out
	@echo "Clean completed for day$(day)"

# Help target to display usage instructions
help:
	@echo "Usage instructions for Makefile:"
	@echo ""
	@echo "make day=<day> program=<program>    Build the program <program>.cu for day <day>"
	@echo "make run day=<day> program=<program> Run the compiled <program>.out for day <day>"
	@echo "make clean day=<day>               Clean all .out files for day <day>"
	@echo ""
	@echo "Examples:"
	@echo "make day=1 program=addition        # Build addition.cu for day 1"
	@echo "make run day=1 program=addition    # Run addition.out for day 1"
	@echo "make clean day=1                  # Clean up .out files for day 1"

# Allow passin
