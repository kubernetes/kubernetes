.PHONY: build push vet test clean

TAG = 0.5
REPO = uluyol/kube-diurnal

BIN = dc

dc: dc.go time.go
	CGO_ENABLED=0 godep go build -a -installsuffix cgo -o dc dc.go time.go

vet:
	godep go vet .

test:
	godep go test .

build: $(BIN)
	docker build -t $(REPO):$(TAG) .

push:
	docker push $(REPO):$(TAG)

clean:
	rm -f $(BIN)
