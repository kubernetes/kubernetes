[![CircleCI](https://circleci.com/gh/charithe/durationcheck.svg?style=svg)](https://circleci.com/gh/charithe/durationcheck)



Duration Check
===============

A Go linter to detect cases where two `time.Duration` values are being multiplied in possibly erroneous ways.

Consider the following (highly contrived) code:

```go
func waitForSeconds(someDuration time.Duration) {
	timeToWait := someDuration * time.Second
	fmt.Printf("Waiting for %s\n", timeToWait)
}

func main() {
	waitForSeconds(5) // waits for 5 seconds
	waitForSeconds(5 * time.Second) // waits for 1388888h 53m 20s
}
```

Both invocations of the function are syntactically correct but the second one is probably not what most people want.
In this contrived example it is quite easy to spot the mistake. However, if the incorrect `waitForSeconds` invocation is
nested deep within a complex piece of code that runs in the background, the mistake could go unnoticed for months (which
is exactly what happened in a production backend system of fairly well-known software service). 


See the [test cases](testdata/src/a/a.go) for more examples of the types of errors detected by the linter.


Installation
-------------

Requires Go 1.11 or above.

```
go get -u github.com/charithe/durationcheck/cmd/durationcheck
```

Usage
-----

Invoke `durationcheck` with your package name

```
durationcheck ./...
# or
durationcheck github.com/you/yourproject/...
```
