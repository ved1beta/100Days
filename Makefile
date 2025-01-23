PROJECT_DIR := $(CURDIR)

COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_RED := \033[31m

all: build

build: $(PROJECT_DIR)/$(dir)/$(program).out

$(PROJECT_DIR)/$(dir)/$(program).out: $(PROJECT_DIR)/$(dir)/$(program).cu
	@echo  "$(COLOR_YELLOW)Building program $(program) in directory $(dir)...$(COLOR_RESET)"
	@nvcc -o $@ $<
	@echo  "$(COLOR_GREEN)Build completed for $(program).out in $(dir)$(COLOR_RESET)"

run: $(PROJECT_DIR)/$(dir)/$(program).out
	@echo  "$(COLOR_BLUE)Running $(program).out in directory $(dir)...$(COLOR_RESET)"
	@./$(dir)/$(program).out

clean:
	@echo  "$(COLOR_RED)Cleaning up .out files in directory $(dir)...$(COLOR_RESET)"
	@rm -f $(PROJECT_DIR)/$(dir)/*.out
	@echo  "$(COLOR_GREEN)Clean completed for directory $(dir)$(COLOR_RESET)"

cleanall:
	@echo  "$(COLOR_RED)Cleaning up all .out files in all directories...$(COLOR_RESET)"
	@find $(PROJECT_DIR) -type f -name "*.out" -exec rm -f {} \;
	@echo  "$(COLOR_GREEN)Cleanall completed for all directories$(COLOR_RESET)"

help:
	@echo  "$(COLOR_BLUE)Usage instructions for Makefile:$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)make dir=<dir> program=<program>$(COLOR_RESET)      # Build the program <program>.cu in directory <dir>"
	@echo "$(COLOR_YELLOW)make run dir=<dir> program=<program>$(COLOR_RESET)  # Run the compiled <program>.out in directory <dir>"
	@echo "$(COLOR_YELLOW)make clean dir=<dir>$(COLOR_RESET)                  # Clean all .out files in directory <dir>"
	@echo "$(COLOR_YELLOW)make cleanall$(COLOR_RESET)                         # Clean all .out files in all directories"
	@echo ""
	@echo "$(COLOR_BLUE)Examples:$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)make dir=day1 program=addition$(COLOR_RESET)        # Build addition.cu in day1"
	@echo "$(COLOR_GREEN)make run dir=day1 program=addition$(COLOR_RESET)    # Run addition.out in day1"
	@echo "$(COLOR_GREEN)make clean dir=day1$(COLOR_RESET)                   # Clean up .out files in day1"
	@echo "$(COLOR_GREEN)make cleanall$(COLOR_RESET)                         # Clean all .out files in all directories"