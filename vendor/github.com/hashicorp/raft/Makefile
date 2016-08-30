DEPS = $(go list -f '{{range .TestImports}}{{.}} {{end}}' ./...)

test:
	go test -timeout=5s ./...

integ: test
	INTEG_TESTS=yes go test -timeout=3s -run=Integ ./...

deps:
	go get -d -v ./...
	echo $(DEPS) | xargs -n1 go get -d

cov:
	INTEG_TESTS=yes gocov test github.com/hashicorp/raft | gocov-html > /tmp/coverage.html
	open /tmp/coverage.html

.PHONY: test cov integ deps
