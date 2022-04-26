[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/astequal)](https://goreportcard.com/report/github.com/go-toolsmith/astequal)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/astequal?status.svg)](https://godoc.org/github.com/go-toolsmith/astequal)
[![Build Status](https://travis-ci.org/go-toolsmith/astequal.svg?branch=master)](https://travis-ci.org/go-toolsmith/astequal)


# astequal

Package astequal provides AST (deep) equallity check operations.

## Installation:

```bash
go get github.com/go-toolsmith/astequal
```

## Example

```go
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"log"
	"reflect"

	"github.com/go-toolsmith/astequal"
)

func main() {
	const code = `
		package foo

		func main() {
			x := []int{1, 2, 3}
			x := []int{1, 2, 3}
		}`

	fset := token.NewFileSet()
	pkg, err := parser.ParseFile(fset, "string", code, 0)
	if err != nil {
		log.Fatalf("parse error: %+v", err)
	}

	fn := pkg.Decls[0].(*ast.FuncDecl)
	x := fn.Body.List[0]
	y := fn.Body.List[1]

	// Reflect DeepEqual will fail due to different Pos values.
	// astequal only checks whether two nodes describe AST.
	fmt.Println(reflect.DeepEqual(x, y)) // => false
	fmt.Println(astequal.Node(x, y))     // => true
	fmt.Println(astequal.Stmt(x, y))     // => true
}
```

## Performance

`astequal` outperforms reflection-based comparison by a big margin:

```
BenchmarkEqualExpr/astequal.Expr-8       5000000     298 ns/op       0 B/op   0 allocs/op
BenchmarkEqualExpr/astequal.Node-8       3000000     409 ns/op       0 B/op   0 allocs/op
BenchmarkEqualExpr/reflect.DeepEqual-8     50000   38898 ns/op   10185 B/op   156 allocs/op
```
