# Gonum BLAS [![GoDoc](https://godoc.org/gonum.org/v1/gonum/blas?status.svg)](https://godoc.org/gonum.org/v1/gonum/blas)

A collection of packages to provide BLAS functionality for the [Go programming
language](http://golang.org)

## Installation
```sh
  go get gonum.org/v1/gonum/blas/...
```

## Packages

### blas

Defines [BLAS API](http://www.netlib.org/blas/blast-forum/cinterface.pdf) split in several
interfaces.

### blas/gonum

Go implementation of the BLAS API (incomplete, implements the `float32` and `float64` API).

### blas/blas64 and blas/blas32

Wrappers for an implementation of the double (i.e., `float64`) and single (`float32`)
precision real parts of the BLAS API.

```Go
package main

import (
	"fmt"

	"gonum.org/v1/gonum/blas/blas64"
)

func main() {
	v := blas64.Vector{Inc: 1, Data: []float64{1, 1, 1}}
	fmt.Println("v has length:", blas64.Nrm2(len(v.Data), v))
}
```

### blas/cblas128 and blas/cblas64

Wrappers for an implementation of the double (i.e., `complex128`) and single (`complex64`) 
precision complex parts of the blas API.

Currently blas/cblas64 and blas/cblas128 require gonum.org/v1/netlib/blas.
