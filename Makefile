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