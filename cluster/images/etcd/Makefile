.PHONY:	clean build push

IMAGE = etcd
TAG = 2.2.1
ETCD_VERSION = 2.2.1
OUTPUT_DIR = $(IMAGE)-v$(ETCD_VERSION)-linux-amd64

clean:
	rm -rf $(OUTPUT_DIR) $(IMAGE)-v$(ETCD_VERSION)-linux-amd64.tar.gz etcd etcdctl

build: clean
	curl -L -O https://github.com/coreos/etcd/releases/download/v$(ETCD_VERSION)/$(IMAGE)-v$(ETCD_VERSION)-linux-amd64.tar.gz
	tar xzvf $(IMAGE)-v$(ETCD_VERSION)-linux-amd64.tar.gz
	cp $(OUTPUT_DIR)/etcd .
	cp $(OUTPUT_DIR)/etcdctl .
	docker build -t gcr.io/google_containers/$(IMAGE):$(TAG) .

push: build
	gcloud docker push gcr.io/google_containers/$(IMAGE):$(TAG)

all:	push
