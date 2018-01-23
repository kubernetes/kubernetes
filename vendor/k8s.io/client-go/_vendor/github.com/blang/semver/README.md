semver for golang [![Build Status](https://drone.io/github.com/blang/semver/status.png)](https://drone.io/github.com/blang/semver/latest) [![GoDoc](https://godoc.org/github.com/blang/semver?status.png)](https://godoc.org/github.com/blang/semver) [![Coverage Status](https://img.shields.io/coveralls/blang/semver.svg)](https://coveralls.io/r/blang/semver?branch=master)
======

semver is a [Semantic Versioning](http://semver.org/) library written in golang. It fully covers spec version `2.0.0`.

Usage
-----
```bash
$ go get github.com/blang/semver
```
Note: Always vendor your dependencies or fix on a specific version tag.

```go
import github.com/blang/semver
v1, err := semver.Make("1.0.0-beta")
v2, err := semver.Make("2.0.0-beta")
v1.Compare(v2)
```

Also check the [GoDocs](http://godoc.org/github.com/blang/semver).

Why should I use this lib?
-----

- Fully spec compatible
- No reflection
- No regex
- Fully tested (Coverage >99%)
- Readable parsing/validation errors
- Fast (See [Benchmarks](#benchmarks))
- Only Stdlib
- Uses values instead of pointers
- Many features, see below


Features
-----

- Parsing and validation at all levels
- Comparator-like comparisons
- Compare Helper Methods
- InPlace manipulation
- Sortable (implements sort.Interface)
- database/sql compatible (sql.Scanner/Valuer)
- encoding/json compatible (json.Marshaler/Unmarshaler)


Example
-----

Have a look at full examples in [examples/main.go](examples/main.go)

```go
import github.com/blang/semver

v, err := semver.Make("0.0.1-alpha.preview+123.github")
fmt.Printf("Major: %d\n", v.Major)
fmt.Printf("Minor: %d\n", v.Minor)
fmt.Printf("Patch: %d\n", v.Patch)
fmt.Printf("Pre: %s\n", v.Pre)
fmt.Printf("Build: %s\n", v.Build)

// Prerelease versions array
if len(v.Pre) > 0 {
    fmt.Println("Prerelease versions:")
    for i, pre := range v.Pre {
        fmt.Printf("%d: %q\n", i, pre)
    }
}

// Build meta data array
if len(v.Build) > 0 {
    fmt.Println("Build meta data:")
    for i, build := range v.Build {
        fmt.Printf("%d: %q\n", i, build)
    }
}

v001, err := semver.Make("0.0.1")
// Compare using helpers: v.GT(v2), v.LT, v.GTE, v.LTE
v001.GT(v) == true
v.LT(v001) == true
v.GTE(v) == true
v.LTE(v) == true

// Or use v.Compare(v2) for comparisons (-1, 0, 1):
v001.Compare(v) == 1
v.Compare(v001) == -1
v.Compare(v) == 0

// Manipulate Version in place:
v.Pre[0], err = semver.NewPRVersion("beta")
if err != nil {
    fmt.Printf("Error parsing pre release version: %q", err)
}

fmt.Println("\nValidate versions:")
v.Build[0] = "?"

err = v.Validate()
if err != nil {
    fmt.Printf("Validation failed: %s\n", err)
}
```

Benchmarks
-----

    BenchmarkParseSimple         5000000      328    ns/op    49 B/op   1 allocs/op
    BenchmarkParseComplex        1000000     2105    ns/op   263 B/op   7 allocs/op
    BenchmarkParseAverage        1000000     1301    ns/op   168 B/op   4 allocs/op
    BenchmarkStringSimple       10000000      130    ns/op     5 B/op   1 allocs/op
    BenchmarkStringLarger        5000000      280    ns/op    32 B/op   2 allocs/op
    BenchmarkStringComplex       3000000      512    ns/op    80 B/op   3 allocs/op
    BenchmarkStringAverage       5000000      387    ns/op    47 B/op   2 allocs/op
    BenchmarkValidateSimple    500000000        7.92 ns/op     0 B/op   0 allocs/op
    BenchmarkValidateComplex     2000000      923    ns/op     0 B/op   0 allocs/op
    BenchmarkValidateAverage     5000000      452    ns/op     0 B/op   0 allocs/op
    BenchmarkCompareSimple     100000000       11.2  ns/op     0 B/op   0 allocs/op
    BenchmarkCompareComplex     50000000       40.9  ns/op     0 B/op   0 allocs/op
    BenchmarkCompareAverage     50000000       43.8  ns/op     0 B/op   0 allocs/op
    BenchmarkSort                5000000      436    ns/op   259 B/op   2 allocs/op

See benchmark cases at [semver_test.go](semver_test.go)


Motivation
-----

I simply couldn't find any lib supporting the full spec. Others were just wrong or used reflection and regex which i don't like.


Contribution
-----

Feel free to make a pull request. For bigger changes create a issue first to discuss about it.


License
-----

See [LICENSE](LICENSE) file.
