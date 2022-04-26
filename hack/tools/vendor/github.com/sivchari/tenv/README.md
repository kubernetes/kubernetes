# tenv
tenv is analyzer that detects using os.Setenv instead of t.Setenv since Go1.17

[![test_and_lint](https://github.com/sivchari/tenv/actions/workflows/workflows.yml/badge.svg?branch=main)](https://github.com/sivchari/tenv/actions/workflows/workflows.yml)

## Instruction

```sh
go install github.com/sivchari/tenv/cmd/tenv
```

## Usage

```go
package sandbox_test

import (
        "os"
        "testing"
)

var (
        e = os.Setenv("a", "b")
        _ = e
)

func setup() {
        os.Setenv("a", "b")
        err := os.Setenv("a", "b")
        if err != nil {
                _ = err
        }
}

func TestF(t *testing.T) {
        setup()
        os.Setenv("a", "b")
        if err := os.Setenv("a", "b"); err != nil {
                _ = err
        }
}
```

### fish

```console
go vet -vettool=(which tenv) sandbox_test.go

# command-line-arguments
./sandbox_test.go:9:2: variable e is not using t.Setenv
./sandbox_test.go:14:2: func setup is not using t.Setenv
./sandbox_test.go:15:2: func setup is not using t.Setenv
./sandbox_test.go:23:2: func TestF is not using t.Setenv
./sandbox_test.go:24:2: func TestF is not using t.Setenv
```

### bash

```console
$ go vet -vettool=`which tenv` main.go

# command-line-arguments
./sandbox_test.go:9:2: variable e is not using t.Setenv
./sandbox_test.go:14:2: func setup is not using t.Setenv
./sandbox_test.go:15:2: func setup is not using t.Setenv
./sandbox_test.go:23:2: func TestF is not using t.Setenv
./sandbox_test.go:24:2: func TestF is not using t.Setenv
```

### option

t.Setenv can use since Go1.17.
This linter diagnostics, if Go version is since 1.17.
But, if you wanna exec this linter in prior Go1.17, you can use it that you set `-tenv.f` flag.

e.g.

### fish

```console
go vet -vettool=(which tenv) -tenv.f sandbox_test.go
```

### bash

```console
go vet -vettool=`which tenv` -tenv.f main.go
```

## CI

### CircleCI

```yaml
- run:
    name: Install tenv
    command: go install github.com/sivchari/tenv

- run:
    name: Run tenv
    command: go vet -vettool=`which tenv` ./...
```

### GitHub Actions

```yaml
- name: Install tenv
  run: go install github.com/sivchari/tenv

- name: Run tenv
  run: go vet -vettool=`which tenv` ./...
```
