# Use:
#
# `make kubectl` will build kubectl.
# `make tag` will suggest a tag.
# `make container` will build a container-- you must supply a tag.
# `make push` will push the container-- you must supply a tag.

kubectl:
	KUBE_STATIC_OVERRIDES="kubectl" ../../hack/build-go.sh cmd/kubectl; cp ../../_output/local/bin/linux/amd64/kubectl .

.tag: kubectl
	./kubectl version -c | grep -o 'GitVersion:"[^"]*"' | cut -f 2 -d '"' > .tag

tag: .tag
	@echo "Suggest using TAG=$(shell cat .tag)"
	@echo "$$ make container TAG=$(shell cat .tag)"
	@echo "or"
	@echo "$$ make push TAG=$(shell cat .tag)"

container:
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	docker build -t gcr.io/google_containers/kubectl:$(TAG) .

push: container
	$(if $(TAG),,$(error TAG is not defined. Use 'make tag' to see a suggestion))
	gcloud docker push gcr.io/google_containers/kubectl:$(TAG)

clean:
	rm -f kubectl
	rm -f .tag
