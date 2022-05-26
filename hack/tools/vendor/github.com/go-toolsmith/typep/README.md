[![Go Report Card](https://goreportcard.com/badge/github.com/go-toolsmith/typep)](https://goreportcard.com/report/github.com/go-toolsmith/typep)
[![GoDoc](https://godoc.org/github.com/go-toolsmith/typep?status.svg)](https://godoc.org/github.com/go-toolsmith/typep)
[![Build Status](https://travis-ci.org/go-toolsmith/typep.svg?branch=master)](https://travis-ci.org/go-toolsmith/typep)

# typep

Package typep provides type predicates.

## Installation:

```bash
go get -v github.com/go-toolsmith/typep
```

## Example

```go
package main

import (
	"fmt"

	"github.com/go-toolsmith/typep"
	"github.com/go-toolsmith/strparse"
)

func main() {
	floatTyp := types.Typ[types.Float32]
	intTyp := types.Typ[types.Int]
	ptr := types.NewPointer(intTyp)
	arr := types.NewArray(intTyp, 64)
	fmt.Println(typep.HasFloatProp(floatTyp)) // => true
	fmt.Println(typep.HasFloatProp(intTyp))   // => false
	fmt.Println(typep.IsPointer(ptr))         // => true
	fmt.Println(typep.IsArray(arr))           // => true
}
```
