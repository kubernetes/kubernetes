# containedctx

[![test_and_lint](https://github.com/sivchari/containedctx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sivchari/containedctx/actions/workflows/ci.yml)

containedctx is a linter that detects struct contained context.Context field. 
This is discouraged technique in favour of passing context as first argument of method or function. 
For rationale please read [Contexts and structs](https://go.dev/blog/context-and-structs) the Go blog post.

## Instruction

```sh
go install github.com/sivchari/containedctx/cmd/containedctx
```

## Usage

```go
package main

import "context"

type ok struct {
	i int
	s string
}

type ng struct {
	ctx context.Context
}

type empty struct{}
```

```console
go vet -vettool=(which containedctx) ./...

# a
./main.go:11:2: found a struct that contains a context.Context field
```


## CI

### CircleCI

```yaml
- run:
    name: install containedctx
    command: go install github.com/sivchari/containedctx/cmd/containedctx

- run:
    name: run containedctx
    command: go vet -vettool=`which containedctx` ./...
```

### GitHub Actions

```yaml
- name: install containedctx
  run: go install github.com/sivchari/containedctx/cmd/containedctx

- name: run containedctx
  run: go vet -vettool=`which containedctx` ./...
```
