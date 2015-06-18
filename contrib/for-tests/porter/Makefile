# Use:
#
# `make porter` will build porter.
# `make tag` will suggest a tag.
# `make container` will build a container-- you must supply a tag.
# `make push` will push the container-- you must supply a tag.

REPO ?= gcr.io/google_containers

porter: porter.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./porter.go

.tag: porter
	md5sum porter | cut -d " " -f 1 > .tag

tag: .tag
	@echo "Suggest using TAG=$(shell cat .tag)"
	@echo "$$ make container TAG=$(shell cat .tag)"
	@echo "or"
	@echo "$$ make push TAG=$(shell cat .tag)"

container:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	docker build -t $(REPO)/porter:$(TAG) .

push:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	gcloud docker push $(REPO)/porter:$(TAG)

clean:
	rm -f porter
	rm -f .tag
