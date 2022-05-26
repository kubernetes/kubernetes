[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/astcast)](https://goreportcard.com/report/github.com/go-toolsmith/astcast)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/astcast?status.svg)](https://godoc.org/github.com/go-toolsmith/astcast)

# astcast

Package astcast wraps type assertion operations in such way that you don't have
to worry about nil pointer results anymore.

## Installation

```bash
go get -v github.com/go-toolsmith/astcast
```

## Example

```go
package main

import (
	"fmt"

	"github.com/go-toolsmith/astcast"
	"github.com/go-toolsmith/strparse"
)

func main() {
	x := strparse.Expr(`(foo * bar) + 1`)

	// x type is ast.Expr, we want to access bar operand
	// that is a RHS of the LHS of the addition.
	// Note that addition LHS (X field) is has parenthesis,
	// so we have to remove them too.

	add := astcast.ToBinaryExpr(x)
	mul := astcast.ToBinaryExpr(astcast.ToParenExpr(add.X).X)
	bar := astcast.ToIdent(mul.Y)
	fmt.Printf("%T %s\n", bar, bar.Name) // => *ast.Ident bar

	// If argument has different dynamic type,
	// non-nil sentinel object of requested type is returned.
	// Those sentinel objects are exported so if you need
	// to know whether it was a nil interface value of
	// failed type assertion, you can compare returned
	// object with such a sentinel.

	y := astcast.ToCallExpr(strparse.Expr(`x`))
	if y == astcast.NilCallExpr {
		fmt.Println("it is a sentinel, type assertion failed")
	}
}
```

Without `astcast`, you would have to do a lots of type assertions:

```go
package main

import (
	"fmt"

	"github.com/go-toolsmith/strparse"
)

func main() {
	x := strparse.Expr(`(foo * bar) + 1`)

	add, ok := x.(*ast.BinaryExpr)
	if !ok || add == nil {
		return
	}
	additionLHS, ok := add.X.(*ast.ParenExpr)
	if !ok || additionLHS == nil {
		return
	}
	mul, ok := additionLHS.X.(*ast.BinaryExpr)
	if !ok || mul == nil {
		return
	}
	bar, ok := mul.Y.(*ast.Ident)
	if !ok || bar == nil {
		return
	}
	fmt.Printf("%T %s\n", bar, bar.Name)
}
```
