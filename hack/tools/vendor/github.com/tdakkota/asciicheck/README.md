# asciicheck [![Go Report Card](https://goreportcard.com/badge/github.com/tdakkota/asciicheck)](https://goreportcard.com/report/github.com/tdakkota/asciicheck) [![codecov](https://codecov.io/gh/tdakkota/asciicheck/branch/master/graph/badge.svg)](https://codecov.io/gh/tdakkota/asciicheck) ![Go](https://github.com/tdakkota/asciicheck/workflows/Go/badge.svg)
Simple linter to check that your code does not contain non-ASCII identifiers

# Install
  
```
go get -u github.com/tdakkota/asciicheck/cmd/asciicheck
```

# Reason to use
So, do you see this code? Looks correct, isn't it?

```go
package main

import "fmt"

type TеstStruct struct{}

func main() {
	s := TestStruct{}
	fmt.Println(s)
}
```
But if you try to run it, you will get an error:
```
./prog.go:8:7: undefined: TestStruct
```
What? `TestStruct` is defined above, but compiler thinks diffrent. Why?

**Answer**:
Because `TestStruct` is not `TеstStruct`.
```
type TеstStruct struct{}
      ^ this 'e' (U+0435) is not 'e' (U+0065)
```

# Usage
asciicheck uses [`singlechecker`](https://pkg.go.dev/golang.org/x/tools/go/analysis/singlechecker) package to run:

```
asciicheck: checks that all code identifiers does not have non-ASCII symbols in the name

Usage: asciicheck [-flag] [package]


Flags:
  -V	print version and exit
  -all
    	no effect (deprecated)
  -c int
    	display offending line with this many lines of context (default -1)
  -cpuprofile string
    	write CPU profile to this file
  -debug string
    	debug flags, any subset of "fpstv"
  -fix
    	apply all suggested fixes
  -flags
    	print analyzer flags in JSON
  -json
    	emit JSON output
  -memprofile string
    	write memory profile to this file
  -source
    	no effect (deprecated)
  -tags string
    	no effect (deprecated)
  -trace string
    	write trace log to this file
  -v	no effect (deprecated)
```
