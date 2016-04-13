# Building and Testing cAdvisor

**Note**: cAdvisor only builds on Linux since it uses Linux-only APIs.

## Installing Dependencies

cAdvisor is written in the [Go](http://golang.org) programming language. If you haven't set up a Go development environment, please follow [these instructions](http://golang.org/doc/code.html) to install go tool and set up GOPATH. Ensure that your version of Go is at least 1.3. Note that the version of Go in package repositories of some operating systems is outdated, so please [download](https://golang.org/dl/) the latest version.

**Note**: cAdvisor requires Go 1.5 to build.

After setting up Go, you should be able to `go get` cAdvisor as expected (we use `-d` to only download):

```
$ go get -d github.com/google/cadvisor
```

We use `godep` so you will need to get that as well:

```
$ go get github.com/tools/godep
```

## Building from Source

At this point you can build cAdvisor from the source folder:

```
$GOPATH/src/github.com/google/cadvisor $ godep go build .
```

or run only unit tests:

```
$GOPATH/src/github.com/google/cadvisor $ godep go test ./... -test.short
```

For integration tests, see the [integration testing](integration_testing.md) page.

## Running Built Binary

Now you can run the built binary:

```
$GOPATH/src/github.com/google/cadvisor $ sudo ./cadvisor
```
