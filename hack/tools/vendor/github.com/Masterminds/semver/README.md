# SemVer

The `semver` package provides the ability to work with [Semantic Versions](http://semver.org) in Go. Specifically it provides the ability to:

* Parse semantic versions
* Sort semantic versions
* Check if a semantic version fits within a set of constraints
* Optionally work with a `v` prefix

[![Stability:
Active](https://masterminds.github.io/stability/active.svg)](https://masterminds.github.io/stability/active.html)
[![Build Status](https://travis-ci.org/Masterminds/semver.svg)](https://travis-ci.org/Masterminds/semver) [![Build status](https://ci.appveyor.com/api/projects/status/jfk66lib7hb985k8/branch/master?svg=true&passingText=windows%20build%20passing&failingText=windows%20build%20failing)](https://ci.appveyor.com/project/mattfarina/semver/branch/master) [![GoDoc](https://godoc.org/github.com/Masterminds/semver?status.svg)](https://godoc.org/github.com/Masterminds/semver) [![Go Report Card](https://goreportcard.com/badge/github.com/Masterminds/semver)](https://goreportcard.com/report/github.com/Masterminds/semver)

If you are looking for a command line tool for version comparisons please see
[vert](https://github.com/Masterminds/vert) which uses this library.

## Parsing Semantic Versions

To parse a semantic version use the `NewVersion` function. For example,

```go
    v, err := semver.NewVersion("1.2.3-beta.1+build345")
```

If there is an error the version wasn't parseable. The version object has methods
to get the parts of the version, compare it to other versions, convert the
version back into a string, and get the original string. For more details
please see the [documentation](https://godoc.org/github.com/Masterminds/semver).

## Sorting Semantic Versions

A set of versions can be sorted using the [`sort`](https://golang.org/pkg/sort/)
package from the standard library. For example,

```go
    raw := []string{"1.2.3", "1.0", "1.3", "2", "0.4.2",}
    vs := make([]*semver.Version, len(raw))
	for i, r := range raw {
		v, err := semver.NewVersion(r)
		if err != nil {
			t.Errorf("Error parsing version: %s", err)
		}

		vs[i] = v
	}

	sort.Sort(semver.Collection(vs))
```

## Checking Version Constraints

Checking a version against version constraints is one of the most featureful
parts of the package.

```go
    c, err := semver.NewConstraint(">= 1.2.3")
    if err != nil {
        // Handle constraint not being parseable.
    }

    v, _ := semver.NewVersion("1.3")
    if err != nil {
        // Handle version not being parseable.
    }
    // Check if the version meets the constraints. The a variable will be true.
    a := c.Check(v)
```

## Basic Comparisons

There are two elements to the comparisons. First, a comparison string is a list
of comma separated and comparisons. These are then separated by || separated or
comparisons. For example, `">= 1.2, < 3.0.0 || >= 4.2.3"` is looking for a
comparison that's greater than or equal to 1.2 and less than 3.0.0 or is
greater than or equal to 4.2.3.

The basic comparisons are:

* `=`: equal (aliased to no operator)
* `!=`: not equal
* `>`: greater than
* `<`: less than
* `>=`: greater than or equal to
* `<=`: less than or equal to

## Working With Pre-release Versions

Pre-releases, for those not familiar with them, are used for software releases
prior to stable or generally available releases. Examples of pre-releases include
development, alpha, beta, and release candidate releases. A pre-release may be
a version such as `1.2.3-beta.1` while the stable release would be `1.2.3`. In the
order of precidence, pre-releases come before their associated releases. In this
example `1.2.3-beta.1 < 1.2.3`.

According to the Semantic Version specification pre-releases may not be
API compliant with their release counterpart. It says,

> A pre-release version indicates that the version is unstable and might not satisfy the intended compatibility requirements as denoted by its associated normal version.

SemVer comparisons without a pre-release comparator will skip pre-release versions.
For example, `>=1.2.3` will skip pre-releases when looking at a list of releases
while `>=1.2.3-0` will evaluate and find pre-releases.

The reason for the `0` as a pre-release version in the example comparison is
because pre-releases can only contain ASCII alphanumerics and hyphens (along with
`.` separators), per the spec. Sorting happens in ASCII sort order, again per the spec. The lowest character is a `0` in ASCII sort order (see an [ASCII Table](http://www.asciitable.com/))

Understanding ASCII sort ordering is important because A-Z comes before a-z. That
means `>=1.2.3-BETA` will return `1.2.3-alpha`. What you might expect from case
sensitivity doesn't apply here. This is due to ASCII sort ordering which is what
the spec specifies.

## Hyphen Range Comparisons

There are multiple methods to handle ranges and the first is hyphens ranges.
These look like:

* `1.2 - 1.4.5` which is equivalent to `>= 1.2, <= 1.4.5`
* `2.3.4 - 4.5` which is equivalent to `>= 2.3.4, <= 4.5`

## Wildcards In Comparisons

The `x`, `X`, and `*` characters can be used as a wildcard character. This works
for all comparison operators. When used on the `=` operator it falls
back to the pack level comparison (see tilde below). For example,

* `1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
* `>= 1.2.x` is equivalent to `>= 1.2.0`
* `<= 2.x` is equivalent to `< 3`
* `*` is equivalent to `>= 0.0.0`

## Tilde Range Comparisons (Patch)

The tilde (`~`) comparison operator is for patch level ranges when a minor
version is specified and major level changes when the minor number is missing.
For example,

* `~1.2.3` is equivalent to `>= 1.2.3, < 1.3.0`
* `~1` is equivalent to `>= 1, < 2`
* `~2.3` is equivalent to `>= 2.3, < 2.4`
* `~1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
* `~1.x` is equivalent to `>= 1, < 2`

## Caret Range Comparisons (Major)

The caret (`^`) comparison operator is for major level changes. This is useful
when comparisons of API versions as a major change is API breaking. For example,

* `^1.2.3` is equivalent to `>= 1.2.3, < 2.0.0`
* `^0.0.1` is equivalent to `>= 0.0.1, < 1.0.0`
* `^1.2.x` is equivalent to `>= 1.2.0, < 2.0.0`
* `^2.3` is equivalent to `>= 2.3, < 3`
* `^2.x` is equivalent to `>= 2.0.0, < 3`

# Validation

In addition to testing a version against a constraint, a version can be validated
against a constraint. When validation fails a slice of errors containing why a
version didn't meet the constraint is returned. For example,

```go
    c, err := semver.NewConstraint("<= 1.2.3, >= 1.4")
    if err != nil {
        // Handle constraint not being parseable.
    }

    v, _ := semver.NewVersion("1.3")
    if err != nil {
        // Handle version not being parseable.
    }

    // Validate a version against a constraint.
    a, msgs := c.Validate(v)
    // a is false
    for _, m := range msgs {
        fmt.Println(m)

        // Loops over the errors which would read
        // "1.3 is greater than 1.2.3"
        // "1.3 is less than 1.4"
    }
```

# Fuzzing

 [dvyukov/go-fuzz](https://github.com/dvyukov/go-fuzz) is used for fuzzing.

1. `go-fuzz-build`
2. `go-fuzz -workdir=fuzz`

# Contribute

If you find an issue or want to contribute please file an [issue](https://github.com/Masterminds/semver/issues)
or [create a pull request](https://github.com/Masterminds/semver/pulls).
