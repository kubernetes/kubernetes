all: push

# 0.0 shouldn't clobber any released builds
TAG = 0.0
PREFIX = bprashanth/servicelb

server: service_loadbalancer.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' -o service_loadbalancer ./service_loadbalancer.go

container: server
	docker build -t $(PREFIX):$(TAG) .

push: container
	docker push $(PREFIX):$(TAG)

clean:
	rm -f service_loadbalancer
