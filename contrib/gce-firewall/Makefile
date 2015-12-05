.PHONY: all gce-firewall server client container push clean test

TAG = 1.1
PREFIX = artfulcoder
all: container

server: config-server/config-server.go
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -o server -a -installsuffix cgo --ldflags '-w' config-server/config-server.go

client: config-client/config-client.go config-client/firewall.go config-client/iptables.go
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -o client -a -installsuffix cgo --ldflags '-w' config-client/config-client.go config-client/firewall.go config-client/iptables.go

test-server: config-test/http-server.go
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -o test-server -a -installsuffix cgo --ldflags '-w' config-test/http-server.go

container: config-server config-client
	docker build -t $(PREFIX)/gce-firewall:$(TAG) .

push:
	gcloud docker push $(PREFIX)/gce-firewall:$(TAG)

clean:
	rm -f server client test-server
