REGISTRY?=gcr.io/google.com/mml-dev
ARCH?=amd64
TEMP_DIR:=$(shell mktemp -d)
VERSION=3.0.14

IMAGE=$(REGISTRY)/etcd-test-$(ARCH):$(VERSION)

build:
	cp -pr ./* $(TEMP_DIR)
	cp -p ../etcd/migrate-if-needed.sh $(TEMP_DIR)
	cd $(TEMP_DIR) && \
		perl -i.bak -pe 's/\@VERSION\@/$(VERSION)/g' Dockerfile
	cd $(TEMP_DIR) && chmod a+rx test-migration.sh migrate-if-needed.sh
	docker build -t $(IMAGE) $(TEMP_DIR)
	rm -rf "$(TEMP_DIR)"

push: build
	gcloud docker -- push $(IMAGE)
