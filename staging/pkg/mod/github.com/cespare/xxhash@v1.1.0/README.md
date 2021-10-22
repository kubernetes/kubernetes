# xxhash

[![GoDoc](https://godoc.org/github.com/cespare/xxhash?status.svg)](https://godoc.org/github.com/cespare/xxhash)

xxhash is a Go implementation of the 64-bit
[xxHash](http://cyan4973.github.io/xxHash/) algorithm, XXH64. This is a
high-quality hashing algorithm that is much faster than anything in the Go
standard library.

The API is very small, taking its cue from the other hashing packages in the
standard library:

    $ go doc github.com/cespare/xxhash                                                                                                                                                                                              !
    package xxhash // import "github.com/cespare/xxhash"

    Package xxhash implements the 64-bit variant of xxHash (XXH64) as described
    at http://cyan4973.github.io/xxHash/.

    func New() hash.Hash64
    func Sum64(b []byte) uint64
    func Sum64String(s string) uint64

This implementation provides a fast pure-Go implementation and an even faster
assembly implementation for amd64.

## Benchmarks

Here are some quick benchmarks comparing the pure-Go and assembly
implementations of Sum64 against another popular Go XXH64 implementation,
[github.com/OneOfOne/xxhash](https://github.com/OneOfOne/xxhash):

| input size | OneOfOne | cespare (purego) | cespare |
| --- | --- | --- | --- |
| 5 B   |  416 MB/s | 720 MB/s |  872 MB/s  |
| 100 B | 3980 MB/s | 5013 MB/s | 5252 MB/s  |
| 4 KB  | 12727 MB/s | 12999 MB/s | 13026 MB/s |
| 10 MB | 9879 MB/s | 10775 MB/s | 10913 MB/s  |

These numbers were generated with:

```
$ go test -benchtime 10s -bench '/OneOfOne,'
$ go test -tags purego -benchtime 10s -bench '/xxhash,'
$ go test -benchtime 10s -bench '/xxhash,'
```

## Projects using this package

- [InfluxDB](https://github.com/influxdata/influxdb)
- [Prometheus](https://github.com/prometheus/prometheus)
