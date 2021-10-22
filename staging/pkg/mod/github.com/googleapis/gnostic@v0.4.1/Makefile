
all:
	./COMPILE-PROTOS.sh
	go get ./...
	go install ./...
	cd extensions/sample; make

test:
	go test ./... -v
