# go-safetemp
[![Godoc](https://godoc.org/github.com/hashcorp/go-safetemp?status.svg)](https://godoc.org/github.com/hashicorp/go-safetemp)

Functions for safely working with temporary directories and files.

## Why?

The Go standard library provides the excellent `ioutil` package for
working with temporary directories and files. This library builds on top
of that to provide safe abstractions above that.
