# gocyclo

[![PkgGoDev](https://pkg.go.dev/badge/github.com/fzipp/gocyclo)](https://pkg.go.dev/github.com/fzipp/gocyclo)
![Build Status](https://github.com/fzipp/gocyclo/workflows/build/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/github.com/fzipp/gocyclo)](https://goreportcard.com/report/github.com/fzipp/gocyclo)

Gocyclo calculates
[cyclomatic complexities](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
of functions in Go source code.

Cyclomatic complexity is a
[code quality metric](https://en.wikipedia.org/wiki/Software_metric)
which can be used to identify code that needs refactoring.
It measures the number of linearly independent paths through a function's
source code.

The cyclomatic complexity of a function is calculated according to the
following rules:

```
 1 is the base complexity of a function
+1 for each 'if', 'for', 'case', '&&' or '||'
```

A function with a higher cyclomatic complexity requires more test cases to
cover all possible paths and is potentially harder to understand. The
complexity can be reduced by applying common refactoring techniques that lead
to smaller functions.

## Installation

To install the `gocyclo` command, run

```
$ go install github.com/fzipp/gocyclo/cmd/gocyclo@latest
```

and put the resulting binary in one of your PATH directories if
`$GOPATH/bin` isn't already in your PATH.

## Usage

```
Calculate cyclomatic complexities of Go functions.
Usage:
    gocyclo [flags] <Go file or directory> ...

Flags:
    -over N               show functions with complexity > N only and
                          return exit code 1 if the set is non-empty
    -top N                show the top N most complex functions only
    -avg, -avg-short      show the average complexity over all functions;
                          the short option prints the value without a label
    -total, -total-short  show the total complexity for all functions;
                          the short option prints the value without a label
    -ignore REGEX         exclude files matching the given regular expression

The output fields for each line are:
<complexity> <package> <function> <file:line:column>
```

## Examples

```
$ gocyclo .
$ gocyclo main.go
$ gocyclo -top 10 src/
$ gocyclo -over 25 docker
$ gocyclo -avg .
$ gocyclo -top 20 -ignore "_test|Godeps|vendor/" .
$ gocyclo -over 3 -avg gocyclo/
```

Example output:

```
9 gocyclo (*complexityVisitor).Visit complexity.go:30:1
8 main main cmd/gocyclo/main.go:53:1
7 gocyclo (*fileAnalyzer).analyzeDecl analyze.go:96:1
4 gocyclo Analyze analyze.go:24:1
4 gocyclo parseDirectives directives.go:27:1
4 gocyclo (Stats).SortAndFilter stats.go:52:1
Average: 2.72
```

Note that the average is calculated over all analyzed functions,
not just the printed ones.

### Ignoring individual functions

Individual functions can be ignored with a `gocyclo:ignore` directive:

```
//gocyclo:ignore
func f1() {
	// ...
}
    
//gocyclo:ignore
var f2 = func() {
	// ...
}
```

## License

This project is free and open source software licensed under the
[BSD 3-Clause License](LICENSE).
