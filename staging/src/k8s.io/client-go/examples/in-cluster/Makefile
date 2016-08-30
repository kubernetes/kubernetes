DEFAULTREPO := gcr.io/google-containers
REPO ?= $(DEFAULTREPO)
all:
	-kubectl delete deployment example
	go build main.go
	docker build -t $(REPO)/client-go-example .
	gcloud docker push $(REPO)/client-go-example
	kubectl run example --image=$(REPO)/client-go-example --image-pull-policy=Always
