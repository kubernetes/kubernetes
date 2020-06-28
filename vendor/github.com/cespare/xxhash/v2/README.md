# xxhash

[![GoDoc](https://godoc.org/github.com/cespare/xxhash?status.svg)](https://godoc.org/github.com/cespare/xxhash)
[![Build Status](https://travis-ci.org/cespare/xxhash.svg?branch=master)](https://travis-ci.org/cespare/xxhash)

xxhash is a Go implementation of the 64-bit
[xxHash](http://cyan4973.github.io/xxHash/) algorithm, XXH64. This is a
high-quality hashing algorithm that is much faster than anything in the Go
standard library.

This package provides a straightforward API:

```
func Sum64(b []byte) uint64
func Sum64String(s string) uint64
type Digest struct{ ... }
    func New() *Digest
```

The `Digest` type implements hash.Hash64. Its key methods are:

```
func (*Digest) Write([]byte) (int, error)
func (*Digest) WriteString(string) (int, error)
func (*Digest) Sum64() uint64
```

This implementation provides a fast pure-Go implementation and an even faster
assembly implementation for amd64.

## Compatibility

This package is in a module and the latest code is in version 2 of the module.
You need a version of Go with at least "minimal module compatibility" to use
github.com/cespare/xxhash/v2:

* 1.9.7+ for Go 1.9
* 1.10.3+ for Go 1.10
* Go 1.11 or later

I recommend using the latest release of Go.

## Benchmarks

Here are some quick benchmarks comparing the pure-Go and assembly
implementations of Sum64.

| input size | purego | asm |
| --- | --- | --- |
| 5 B   |  979.66 MB/s |  1291.17 MB/s  |
| 100 B | 7475.26 MB/s | 7973.40 MB/s  |
| 4 KB  | 17573.46 MB/s | 17602.65 MB/s |
| 10 MB | 17131.46 MB/s | 17142.16 MB/s |

These numbers were generated on Ubuntu 18.04 with an Intel i7-8700K CPU using
the following commands under Go 1.11.2:

```
$ go test -tags purego -benchtime 10s -bench '/xxhash,direct,bytes'
$ go test -benchtime 10s -bench '/xxhash,direct,bytes'
```

## Projects using this package

- [InfluxDB](https://github.com/influxdata/influxdb)
- [Prometheus](https://github.com/prometheus/prometheus)
- [FreeCache](https://github.com/coocood/freecache)
