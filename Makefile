get-mnist:
	@mkdir -p example/mnist
	@cd example/mnist && wget https://pjreddie.com/media/files/mnist_train.csv
	@cd example/mnist && wget https://pjreddie.com/media/files/mnist_test.csv

run-fuzzy-art:
	@go mod tidy
	@cd example/fuzzy_art && go run .

run-default-artmap-2:
	@go mod tidy
	@cd example/default_artmap_2 && go run .

.PHONY: test
test:
	go test ./... -race

# By using `-run=^$$`, it effectively selects no tests to run.
# This is often used in conjunction with other flags like `-bench`
# to ensure that only benchmarks are executed, and no regular tests are run.
.PHONY: bench
bench:
	go test . -run=^$$ -bench=. -benchmem -cpuprofile cpu.prof -count 5

.PHONY: trace
trace:
	go test . -run=^$$ -bench=. -trace=trace.out

# call this when running the example
get-trace:
	curl localhost:5555/debug/pprof/trace?milliseconds=1 > trace.out && go tool trace trace.out