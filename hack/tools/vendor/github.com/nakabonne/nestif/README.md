# nestif

[![Go Doc](https://img.shields.io/badge/godoc-reference-blue.svg?style=flat-square)](http://godoc.org/github.com/nakabonne/nestif)

Reports complex nested if statements in Go code, by calculating its complexities based on the rules defined by the [Cognitive Complexity white paper by G. Ann Campbell](https://www.sonarsource.com/docs/CognitiveComplexity.pdf).

It helps you find if statements that make your code hard to read, and clarifies which parts to refactor.

## Installation

### By go get

```
go get github.com/nakabonne/nestif/cmd/nestif
```

### By golangci-lint

`nestif` is already integrated with [golangci-lint](https://github.com/golangci/golangci-lint). Please refer to the instructions there and enable it.

## Usage

### Quick Start

```bash
nestif
```

The `...` glob operator is supported, and the above is an equivalent of:

```bash
nestif ./...
```

One or more files and directories can be specified in a single command:

```bash
nestif dir/foo.go dir2 dir3/...
```

Packages can be specified as well:

```bash
nestif github.com/foo/bar example.com/bar/baz
```

### Options

```
usage: nestif [<flag> ...] <Go files or directories or packages> ...
  -e, --exclude-dirs strings   regexps of directories to be excluded for checking; comma-separated list
      --json                   emit json format
      --min int                minimum complexity to show (default 1)
      --top int                show only the top N most complex if statements (default 10)
  -v, --verbose                verbose output
```

### Example

Let's say you write:

```go
package main

func _() {
    if foo {
        if bar {
        }
    }

    if baz == "baz" {
        if qux {
            if quux {
            }
        }
    }
}
```

And give it to nestif:

```console
$ nestif foo.go
foo.go:9:2: `if baz == "baz"` is nested (complexity: 3)
foo.go:4:2: `if foo` is nested (complexity: 1)
```

Note that the results are sorted in descending order of complexity. In addition, it shows only the top 10 most complex if statements by default, and you can specify how many to show with `-top` flag.

### Rules

It calculates the complexities of if statements according to the nesting rules of Cognitive Complexity.
Since the more deeply-nested your code gets, the harder it can be to reason about, it assesses a nesting increment for it:

```go
if condition1 {
    if condition2 { // +1
        if condition3 { // +2
            if condition4 { // +3
            }
        }
    }
}
```

`else` and `else if` increase complexity by one wherever they are because the mental cost has already been paid when reading the if:

```go
if condition1 {
    if condition2 { // +1
        if condition3 { // +2
        } else if condition4 { // +1
	} else { // +1
	    if condition5 { // +3
	    }
        }
    }
}
```

## Inspired by

- [uudashr/gocognit](https://github.com/uudashr/gocognit)
- [fzipp/gocyclo](https://github.com/fzipp/gocyclo)

## Further reading

Please see the [Cognitive Complexity: A new way of measuring understandability](https://www.sonarsource.com/docs/CognitiveComplexity.pdf) white paper by G. Ann Campbell for more detail on Cognitive Complexity.
