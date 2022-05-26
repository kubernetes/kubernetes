[![CircleCI](https://circleci.com/gh/sylvia7788/contextcheck.svg?style=svg)](https://circleci.com/gh/sylvia7788/contextcheck)


# contextcheck

`contextcheck` is a static analysis tool, it is used to check the function whether use a non-inherited context, which will result in a broken call link.

For example:

```go
func call1(ctx context.Context) {
    ...

    ctx = getNewCtx(ctx)
    call2(ctx) // OK

    call2(context.Background()) // Non-inherited new context, use function like `context.WithXXX` instead

    call3() // Function `call3` should pass the context parameter
    ...
}

func call2(ctx context.Context) {
    ...
}

func call3() {
    ctx := context.TODO()
    call2(ctx)
}

func getNewCtx(ctx context.Context) (newCtx context.Context) {
    ...
    return
}
```

## Installation

You can get `contextcheck` by `go get` command.

```bash
$ go get -u github.com/sylvia7788/contextcheck
```

or build yourself.

```bash
$ make build
$ make install
```

## Usage

Invoke `contextcheck` with your package name

```bash
$ contextcheck ./...
$ # or
$ go vet -vettool=`which contextcheck` ./...
```
