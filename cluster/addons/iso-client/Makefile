ifeq ($(IMAGE_NAME),)
IMAGE_NAME:=cgroup-cpuset-cpus-isolator
endif

all: binary docker

binary:
	go build -v -o coreaffinity-isolator

docker: binary
	docker build -t ${IMAGE_NAME}:latest .

clean:
	rm -rf coreaffinity-isolator > /dev/null 2>&1
