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
- Ranges `>=1.0.0 <2.0.0 || >=3.0.0 !3.0.1-beta.1`
- Sortable (implements sort.Interface)
- database/sql compatible (sql.Scanner/Valuer)
- encoding/json compatible (json.Marshaler/Unmarshaler)

Ranges
------

A `Range` is a set of conditions which specify which versions satisfy the range.

A condition is composed of an operator and a version. The supported operators are:

- `<1.0.0` Less than `1.0.0`
- `<=1.0.0` Less than or equal to `1.0.0`
- `>1.0.0` Greater than `1.0.0`
- `>=1.0.0` Greater than or equal to `1.0.0`
- `1.0.0`, `=1.0.0`, `==1.0.0` Equal to `1.0.0`
- `!1.0.0`, `!=1.0.0` Not equal to `1.0.0`. Excludes version `1.0.0`.

A `Range` can link multiple `Ranges` separated by space:

Ranges can be linked by logical AND:

  - `>1.0.0 <2.0.0` would match between both ranges, so `1.1.1` and `1.8.7` but not `1.0.0` or `2.0.0`
  - `>1.0.0 <3.0.0 !2.0.3-beta.2` would match every version between `1.0.0` and `3.0.0` except `2.0.3-beta.2`

Ranges can also be linked by logical OR:

  - `<2.0.0 || >=3.0.0` would match `1.x.x` and `3.x.x` but not `2.x.x`

AND has a higher precedence than OR. It's not possible to use brackets.

Ranges can be combined by both AND and OR

  - `>1.0.0 <2.0.0 || >3.0.0 !4.2.1` would match `1.2.3`, `1.9.9`, `3.1.1`, but not `4.2.1`, `2.1.1`

Range usage:

```
v, err := semver.Parse("1.2.3")
range, err := semver.ParseRange(">1.0.0 <2.0.0 || >=3.0.0")
if range(v) {
    //valid
}

```

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

    BenchmarkParseSimple-4           5000000    390    ns/op    48 B/op   1 allocs/op
    BenchmarkParseComplex-4          1000000   1813    ns/op   256 B/op   7 allocs/op
    BenchmarkParseAverage-4          1000000   1171    ns/op   163 B/op   4 allocs/op
    BenchmarkStringSimple-4         20000000    119    ns/op    16 B/op   1 allocs/op
    BenchmarkStringLarger-4         10000000    206    ns/op    32 B/op   2 allocs/op
    BenchmarkStringComplex-4         5000000    324    ns/op    80 B/op   3 allocs/op
    BenchmarkStringAverage-4         5000000    273    ns/op    53 B/op   2 allocs/op
    BenchmarkValidateSimple-4      200000000      9.33 ns/op     0 B/op   0 allocs/op
    BenchmarkValidateComplex-4       3000000    469    ns/op     0 B/op   0 allocs/op
    BenchmarkValidateAverage-4       5000000    256    ns/op     0 B/op   0 allocs/op
    BenchmarkCompareSimple-4       100000000     11.8  ns/op     0 B/op   0 allocs/op
    BenchmarkCompareComplex-4       50000000     30.8  ns/op     0 B/op   0 allocs/op
    BenchmarkCompareAverage-4       30000000     41.5  ns/op     0 B/op   0 allocs/op
    BenchmarkSort-4                  3000000    419    ns/op   256 B/op   2 allocs/op
    BenchmarkRangeParseSimple-4      2000000    850    ns/op   192 B/op   5 allocs/op
    BenchmarkRangeParseAverage-4     1000000   1677    ns/op   400 B/op  10 allocs/op
    BenchmarkRangeParseComplex-4      300000   5214    ns/op  1440 B/op  30 allocs/op
    BenchmarkRangeMatchSimple-4     50000000     25.6  ns/op     0 B/op   0 allocs/op
    BenchmarkRangeMatchAverage-4    30000000     56.4  ns/op     0 B/op   0 allocs/op
    BenchmarkRangeMatchComplex-4    10000000    153    ns/op     0 B/op   0 allocs/op

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
