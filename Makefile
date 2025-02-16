get-mnist:
	@mkdir -p example/mnist
	@cd example/mnist && wget https://pjreddie.com/media/files/mnist_train.csv
	@cd example/mnist && wget https://pjreddie.com/media/files/mnist_test.csv

run-example:
	@go mod tidy
	@cd example && go run .