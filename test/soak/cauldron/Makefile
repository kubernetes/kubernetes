.PHONY: cauldron

TAG = 1.0

cauldron:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 godep go build -a -installsuffix cgo --ldflags '-w' cauldron.go

container:	cauldron
	docker build -t kubernetes/cauldron:$(TAG) .

push:
	docker push kubernetes/cauldron:$(TAG)

rc:
	kubectl create --validate -f cauldron-rc.yaml

stop:
	kubectl delete rc cauldron

get:
	kubectl get rc,pods -l app=cauldron

scale20:
	kubectl scale rc cauldron --replicas=20

scale1:
	kubectl scale rc cauldron --replicas=1

clean:
	rm -rf cauldron
