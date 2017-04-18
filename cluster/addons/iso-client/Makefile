all: binary docker

binary:
	go build -v -o coreaffinity-isolator

docker: all
	docker build -t iso-client:latest .

clean:
	rm -rf coreaffinity-isolator > /dev/null 2>&1
