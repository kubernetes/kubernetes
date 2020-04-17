test:
	[ -z "`gofmt -s -w -l -e .`" ]
	go vet
	ginkgo -p -r --randomizeAllSpecs --failOnPending --randomizeSuites --race

.PHONY: test
