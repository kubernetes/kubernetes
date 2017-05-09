.PHONY: build push

all:	proxy_sustain service_sustain

TAG = 1.0

proxy_sustain:	proxy_sustain.go
		go build proxy_sustain.go

service_sustain:	service_sustain.go
		GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' service_sustain.go

build:	service_sustain
		docker build -t satnam6502/sustain:$(TAG) .

push:
	docker push satnam6502/sustain:$(TAG)

mkpod:
	kubectl create -f sustain-pod.yaml

delpod:
	kubectl delete pod sustain

logs:
	kubectl logs sustain
