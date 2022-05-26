[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/astp)](https://goreportcard.com/report/github.com/go-toolsmith/astp)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/astp?status.svg)](https://godoc.org/github.com/go-toolsmith/astp)
[![Build Status](https://travis-ci.org/go-toolsmith/astp.svg?branch=master)](https://travis-ci.org/go-toolsmith/astp)


# astp

Package astp provides AST predicates.

## Installation:

```bash
go get github.com/go-toolsmith/astp
```

## Example

```go
package main

import (
	"fmt"

	"github.com/go-toolsmith/astp"
	"github.com/go-toolsmith/strparse"
)

func main() {
	if astp.IsIdent(strparse.Expr(`x`)) {
		fmt.Println("ident")
	}
	if astp.IsBlockStmt(strparse.Stmt(`{f()}`)) {
		fmt.Println("block stmt")
	}
	if astp.IsGenDecl(strparse.Decl(`var x int = 10`)) {
		fmt.Println("gen decl")
	}
}
```
