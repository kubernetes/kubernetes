# Versioning Library for Go
[![Build Status](https://circleci.com/gh/hashicorp/go-version/tree/master.svg?style=svg)](https://circleci.com/gh/hashicorp/go-version/tree/master)
[![GoDoc](https://godoc.org/github.com/hashicorp/go-version?status.svg)](https://godoc.org/github.com/hashicorp/go-version)

go-version is a library for parsing versions and version constraints,
and verifying versions against a set of constraints. go-version
can sort a collection of versions properly, handles prerelease/beta
versions, can increment versions, etc.

Versions used with go-version must follow [SemVer](http://semver.org/).

## Installation and Usage

Package documentation can be found on
[GoDoc](http://godoc.org/github.com/hashicorp/go-version).

Installation can be done with a normal `go get`:

```
$ go get github.com/hashicorp/go-version
```

#### Version Parsing and Comparison

```go
v1, err := version.NewVersion("1.2")
v2, err := version.NewVersion("1.5+metadata")

// Comparison example. There is also GreaterThan, Equal, and just
// a simple Compare that returns an int allowing easy >=, <=, etc.
if v1.LessThan(v2) {
    fmt.Printf("%s is less than %s", v1, v2)
}
```

#### Version Constraints

```go
v1, err := version.NewVersion("1.2")

// Constraints example.
constraints, err := version.NewConstraint(">= 1.0, < 1.4")
if constraints.Check(v1) {
	fmt.Printf("%s satisfies constraints %s", v1, constraints)
}
```

#### Version Sorting

```go
versionsRaw := []string{"1.1", "0.7.1", "1.4-beta", "1.4", "2"}
versions := make([]*version.Version, len(versionsRaw))
for i, raw := range versionsRaw {
    v, _ := version.NewVersion(raw)
    versions[i] = v
}

// After this, the versions are properly sorted
sort.Sort(version.Collection(versions))
```

## Issues and Contributing

If you find an issue with this library, please report an issue. If you'd
like, we welcome any contributions. Fork this library and submit a pull
request.
