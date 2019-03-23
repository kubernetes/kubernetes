# Gonum floats [![GoDoc](https://godoc.org/gonum.org/v1/gonum/floats?status.svg)](https://godoc.org/gonum.org/v1/gonum/floats)

Package floats provides a set of helper routines for dealing with slices of float64.
The functions avoid allocations to allow for use within tight loops without garbage collection overhead.
