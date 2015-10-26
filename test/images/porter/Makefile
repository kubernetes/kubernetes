# Use:
#
# `make porter` will build porter.
# `make tag` will suggest a tag.
# `make image` will build an image-- you must supply a tag.
# `make push` will push the image-- you must supply a tag.

# This image does not tag in the normal way
# TAG =
PREFIX = gcr.io/google_containers
SUGGESTED_TAG = $(shell git rev-parse --verify HEAD)

porter: porter.go
	CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w' ./porter.go

tag:
	@echo "If all relevant changes are committed, suggest using TAG=$(SUGGESTED_TAG)"
	@echo "$$ make container TAG=$(SUGGESTED_TAG)"
	@echo "or"
	@echo "$$ make push TAG=$(SUGGESTED_TAG)"

container: image

image:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' after committing changes to see a suggestion))
	docker build -t $(PREFIX)/porter:$(TAG) .

push:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' after committing changes to see a suggestion))
	gcloud docker push $(PREFIX)/porter:$(TAG)

clean:
	rm -f porter
