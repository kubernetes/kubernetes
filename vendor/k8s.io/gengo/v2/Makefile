all:
	go build ./...

test:
	GODEBUG=gotypesalias=0 go test -race ./... -count=1
	GODEBUG=gotypesalias=1 go test -race ./... -count=1

# We verify for the maximum version of the go directive as 1.20
# here because the oldest go directive that exists on our supported
# release branches in k/k is 1.20.
verify:
	GODEBUG=gotypesalias=0 ./hack/verify-examples.sh
	GODEBUG=gotypesalias=1 ./hack/verify-examples.sh
	./hack/verify-go-directive.sh 1.20
