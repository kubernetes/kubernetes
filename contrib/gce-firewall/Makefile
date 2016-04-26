.PHONY: all gce-firewall server client container push clean

TAG = 1.1
PREFIX = artfulcoder
all: container

server: firewall-rules-server/*.go common/*.go
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -o server -a -installsuffix cgo --ldflags '-w' firewall-rules-server/main.go

client: firewall-rules-client/*.go common/*.go
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -o client -a -installsuffix cgo --ldflags '-w' firewall-rules-client/main.go firewall-rules-client/firewall.go firewall-rules-client/iptables.go

container: server client
	docker build -t $(PREFIX)/gce-firewall:$(TAG) .

push: container
	gcloud docker push $(PREFIX)/gce-firewall:$(TAG)

clean:
	rm -f server client
