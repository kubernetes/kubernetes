ifndef TAGS
TAGS := daemon
endif

ifndef PKGS
PKGS := $(shell go list ./... 2>&1 | grep -v 'github.com/libopenstorage/openstorage/vendor')
endif

ifeq ($(BUILD_TYPE),debug)
BUILDFLAGS := -gcflags "-N -l"
endif

ifdef HAVE_BTRFS
TAGS+=btrfs_noversion have_btrfs
endif

ifdef HAVE_CHAINFS
TAGS+=have_chainfs
endif

ifndef PROTOC
PROTOC = protoc
endif

ifndef PROTOS_PATH
PROTOS_PATH = $(GOPATH)/src
endif

ifndef PROTOSRC_PATH
PROTOSRC_PATH = $(PROTOS_PATH)/github.com/libopenstorage/openstorage
endif

export GO15VENDOREXPERIMENT=1

all: build

deps:
	GO15VENDOREXPERIMENT=0 go get -d -v $(PKGS)

update-deps:
	GO15VENDOREXPERIMENT=0 go get -d -v -u -f $(PKGS)

test-deps:
	GO15VENDOREXPERIMENT=0 go get -d -v -t $(PKGS)

update-test-deps:
	GO15VENDOREXPERIMENT=0 go get -tags "$(TAGS)" -d -v -t -u -f $(PKGS)

vendor-update:
	GO15VENDOREXPERIMENT=0 GOOS=linux GOARCH=amd64 go get -tags "daemon btrfs_noversion have_btrfs have_chainfs" -d -v -t -u -f $(shell go list ./... 2>&1 | grep -v 'github.com/libopenstorage/openstorage/vendor')

vendor-without-update:
	go get -v github.com/kardianos/govendor
	rm -rf vendor
	govendor init
	GOOS=linux GOARCH=amd64 govendor add +external
	GOOS=linux GOARCH=amd64 govendor update +vendor
	GOOS=linux GOARCH=amd64 govendor add +external
	GOOS=linux GOARCH=amd64 govendor update +vendor

vendor: vendor-update vendor-without-update

build:
	go build -tags "$(TAGS)" $(BUILDFLAGS) $(PKGS)

install:
	go install -tags "$(TAGS)" $(PKGS)

proto:
	go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway
	@echo "Generating protobuf definitions from api/api.proto"
	$(PROTOC) -I $(PROTOSRC_PATH) $(PROTOSRC_PATH)/api/api.proto --go_out=plugins=grpc:.
	@echo "Generating grpc protobuf definitions from pkg/flexvolume/flexvolume.proto"
	$(PROTOC) -I/usr/local/include -I$(PROTOSRC_PATH) -I$(PROTOS_PATH)/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis --go_out=plugins=grpc:. $(PROTOSRC_PATH)/pkg/flexvolume/flexvolume.proto
	$(PROTOC) -I/usr/local/include -I$(PROTOSRC_PATH) -I$(PROTOS_PATH)/github.com/grpc-ecosystem/grpc-gateway/third_party/googleapis --grpc-gateway_out=logtostderr=true:. $(PROTOSRC_PATH)/pkg/flexvolume/flexvolume.proto
	@echo "Generating protobuf definitions from pkg/jsonpb/testing/testing.proto"
	$(PROTOC) -I $(PROTOSRC_PATH) $(PROTOSRC_PATH)/pkg/jsonpb/testing/testing.proto --go_out=plugins=grpc:.

lint:
	go get -v github.com/golang/lint/golint
	for file in $$(find . -name '*.go' | grep -v vendor | grep -v '\.pb\.go' | grep -v '\.pb\.gw\.go'); do \
		golint $${file}; \
		if [ -n "$$(golint $${file})" ]; then \
			exit 1; \
		fi; \
	done

vet:
	go vet $(PKGS)

errcheck:
	go get -v github.com/kisielk/errcheck
	errcheck -tags "$(TAGS)" $(PKGS)

pretest: lint vet errcheck

test:
	go test -tags "$(TAGS)" $(TESTFLAGS) $(PKGS)

docker-build-osd-dev:
	docker build -t openstorage/osd-dev -f Dockerfile.osd-dev .

docker-build: docker-build-osd-dev
	docker run \
		--privileged \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		-e "TAGS=$(TAGS)" \
		-e "PKGS=$(PKGS)" \
		-e "BUILDFLAGS=$(BUILDFLAGS)" \
		openstorage/osd-dev \
			make build

docker-test: docker-build-osd-dev
	docker run \
		--privileged \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY \
		-e "TAGS=$(TAGS)" \
		-e "PKGS=$(PKGS)" \
		-e "BUILDFLAGS=$(BUILDFLAGS)" \
		-e "TESTFLAGS=$(TESTFLAGS)" \
		openstorage/osd-dev \
			make test

docker-build-osd-internal:
	rm -rf _tmp
	mkdir -p _tmp
	go build -a -tags "$(TAGS)" -o _tmp/osd cmd/osd/main.go
	docker build -t openstorage/osd -f Dockerfile.osd .

docker-build-osd: docker-build-osd-dev
	docker run \
		-v /var/run/docker.sock:/var/run/docker.sock \
		-e "TAGS=$(TAGS)" \
		-e "PKGS=$(PKGS)" \
		-e "BUILDFLAGS=$(BUILDFLAGS)" \
		openstorage/osd-dev \
			make docker-build-osd-internal

launch: docker-build-osd
	docker run \
		--privileged \
		-d \
		-v $(shell pwd):/etc \
		-v /run/docker/plugins:/run/docker/plugins \
		-v /var/lib/osd/:/var/lib/osd/ \
		-p 9005:9005 \
		openstorage/osd -d -f /etc/config.yaml

# must set HAVE_BTRFS
launch-local-btrfs: install
	sudo bash -x etc/btrfs/init.sh
	sudo $(shell which osd) -d -f etc/config/config_btrfs.yaml

install-flexvolume:
	go install -a -tags "$(TAGS)" github.com/libopenstorage/openstorage/pkg/flexvolume github.com/libopenstorage/openstorage/pkg/flexvolume/cmd/flexvolume

install-flexvolume-plugin: install-flexvolume
	sudo rm -rf /usr/libexec/kubernetes/kubelet/volume/exec-plugins/openstorage~openstorage
	sudo mkdir -p /usr/libexec/kubernetes/kubelet/volume/exec-plugins/openstorage~openstorage
	sudo chmod 777 /usr/libexec/kubernetes/kubelet/volume/exec-plugins/openstorage~openstorage
	cp $(GOPATH)/bin/flexvolume /usr/libexec/kubernetes/kubelet/volume/exec-plugins/openstorage~openstorage/openstorage

clean:
	go clean -i $(PKGS)

.PHONY: \
	all \
	deps \
	update-deps \
	test-deps \
	update-test-deps \
	vendor-update \
	vendor-without-update \
	vendor \
	build \
	install \
	proto \
	lint \
	vet \
	errcheck \
	pretest \
	test \
	docker-build-osd-dev \
	docker-build \
	docker-test \
	docker-build-osd-internal \
	docker-build-osd \
	launch \
	launch-local-btrfs \
	install-flexvolume-plugin \
	clean
