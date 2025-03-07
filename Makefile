get-mnist:
	@mkdir -p testdata
	@cd testdata && wget https://pjreddie.com/media/files/mnist_train.csv
	@cd testdata && wget https://pjreddie.com/media/files/mnist_test.csv

run:
	@go mod tidy
	@cd example/fuzzy_art && go run .