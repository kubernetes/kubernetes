[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/strparse)](https://goreportcard.com/report/github.com/go-toolsmith/strparse)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/strparse?status.svg)](https://godoc.org/github.com/go-toolsmith/strparse)
[![Build Status](https://travis-ci.org/go-toolsmith/strparse.svg?branch=master)](https://travis-ci.org/go-toolsmith/strparse)


# strparse

Package strparse provides convenience wrappers around `go/parser` for simple
expression, statement and declaretion parsing from string.

## Installation

```bash
go get github.com/go-toolsmith/strparse
```

## Example

```go
package main

import (
	"go-toolsmith/astequal"
	"go-toolsmith/strparse"
)

func main() {
	// Comparing AST strings for equallity (note different spacing):
	x := strparse.Expr(`1 + f(v[0].X)`)
	y := strparse.Expr(` 1+f( v[0].X ) `)
	fmt.Println(astequal.Expr(x, y)) // => true
}

```
