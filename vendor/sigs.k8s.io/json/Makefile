.PHONY: default build test benchmark fmt vet

default: build

build:
	go build ./...

test:
	go test sigs.k8s.io/json/...

benchmark:
	go test sigs.k8s.io/json -bench . -benchmem

fmt:
	go mod tidy
	gofmt -s -w *.go

vet:
	go vet sigs.k8s.io/json

	@echo "checking for external dependencies"
	@deps=$$(go list -f '{{ if not (or .Standard .Module.Main) }}{{.ImportPath}}{{ end }}' -deps sigs.k8s.io/json/... || true); \
	if [ -n "$${deps}" ]; then \
		echo "only stdlib dependencies allowed, found:"; \
		echo "$${deps}"; \
		exit 1; \
	fi

	@echo "checking for unsafe use"
	@unsafe=$$(go list -f '{{.ImportPath}} depends on {{.Imports}}' sigs.k8s.io/json/... | grep unsafe || true); \
	if [ -n "$${unsafe}" ]; then \
		echo "no dependencies on unsafe allowed, found:"; \
		echo "$${unsafe}"; \
		exit 1; \
	fi
