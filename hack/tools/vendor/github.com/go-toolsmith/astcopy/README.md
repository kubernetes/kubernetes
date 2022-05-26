[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/astcopy)](https://goreportcard.com/report/github.com/go-toolsmith/astcopy)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/astcopy?status.svg)](https://godoc.org/github.com/go-toolsmith/astcopy)
[![Build Status](https://travis-ci.org/go-toolsmith/astcopy.svg?branch=master)](https://travis-ci.org/go-toolsmith/astcopy)

# astcopy

Package astcopy implements Go AST reflection-free deep copy operations.

## Installation:

```bash
go get github.com/go-toolsmith/astcopy
```

## Example

```go
package main

import (
	"fmt"
	"go/ast"
	"go/token"

	"github.com/go-toolsmith/astcopy"
	"github.com/go-toolsmith/astequal"
	"github.com/go-toolsmith/strparse"
)

func main() {
	x := strparse.Expr(`1 + 2`).(*ast.BinaryExpr)
	y := astcopy.BinaryExpr(x)
	fmt.Println(astequal.Expr(x, y)) // => true

	// Now modify x and make sure y is not modified.
	z := astcopy.BinaryExpr(y)
	x.Op = token.SUB
	fmt.Println(astequal.Expr(y, z)) // => true
	fmt.Println(astequal.Expr(x, y)) // => false
}
```
