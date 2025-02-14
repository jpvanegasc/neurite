all: main.o

COMPILER = gcc -std=c11
MAIN = main.c
ALL = $(MAIN) neural_network.o training.o

run: main.o
	time ./main.o

main.o: $(ALL)
	$(COMPILER) -O2 -Wall $^ -o $@

neural_network.o : neural_network.c neural_network.h
	$(COMPILER) -O2 -Wall -c $< -o $@

training.o : training.c training.h
	$(COMPILER) -O2 -Wall -c $< -o $@

# Debugging and profiling
DEBUG = -g -Wall -fsanitize=address -fsanitize=leak -fsanitize=undefined
VAL = --track-origins=yes --leak-check=full
CACH = --tool=cachegrind

.PHONY:
.SILENT:

assembler: $(ALL) ## Disassemble the project
	$(COMPILER) -g -S $(MAIN) -o main.S

parallel: $(ALL) ## Run using OpenMP
	$(COMPILER) -fopenmp -O2 $< -o parallel.o
	time ./parallel.o

debug: $(ALL) ## Compile and run with debugging
	$(COMPILER) $(DEBUG) $< -o debug.o
	./debug.o

valgrind: $(ALL) ## Memory leaks
	$(COMPILER) -g $< -o valgrind.o
	valgrind $(VAL) ./valgrind.o

cachegrind: $(ALL) ## Cache misses
	$(COMPILER) -g $< -o cachegrind.o
	valgrind $(CACH) ./cachegrind.o

gprof: $(ALL) ## Cross-platform performance analysis
	$(COMPILER) -Wall -pg $< -o gprof.o
	./gprof.o
	gprof gprof.o gmon.out > analysis.txt
	most analysis.txt

perf: $(ALL) ## Linux performance analysis (includes call stack)
	$(COMPILER) -Wall -pg $< -o perf.o
	perf record ./perf.o ; perf report

clean: ## Remove compiled files
	rm -f *.o *.txt *.out* debug main *.data

help: ## Show this help message
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
