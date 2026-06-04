# SemVer

The `semver` package provides the ability to work with [Semantic Versions](http://semver.org) in Go. Specifically it provides the ability to:

* Parse semantic versions
* Sort semantic versions
* Check if a semantic version fits within a set of constraints
* Optionally work with a `v` prefix

[![Stability:
Active](https://masterminds.github.io/stability/active.svg)](https://masterminds.github.io/stability/active.html)
[![](https://github.com/Masterminds/semver/workflows/Tests/badge.svg)](https://github.com/Masterminds/semver/actions)
[![GoDoc](https://img.shields.io/static/v1?label=godoc&message=reference&color=blue)](https://pkg.go.dev/github.com/Masterminds/semver/v3)
[![Go Report Card](https://goreportcard.com/badge/github.com/Masterminds/semver)](https://goreportcard.com/report/github.com/Masterminds/semver)

## Package Versions

Note, import `github.com/Masterminds/semver/v3` to use the latest version.

There are three major versions fo the `semver` package.

* 3.x.x is the stable and active version. This version is focused on constraint
  compatibility for range handling in other tools from other languages. It has
  a similar API to the v1 releases. The development of this version is on the master
  branch. The documentation for this version is below.
* 2.x was developed primarily for [dep](https://github.com/golang/dep). There are
  no tagged releases and the development was performed by [@sdboyer](https://github.com/sdboyer).
  There are API breaking changes from v1. This version lives on the [2.x branch](https://github.com/Masterminds/semver/tree/2.x).
* 1.x.x is the original release. It is no longer maintained. You should use the
  v3 release instead. You can read the documentation for the 1.x.x release
  [here](https://github.com/Masterminds/semver/blob/release-1/README.md).

## Parsing Semantic Versions

There are two functions that can parse semantic versions. The `StrictNewVersion`
function only parses valid version 2 semantic versions as outlined in the
specification. The `NewVersion` function attempts to coerce a version into a
semantic version and parse it. For example, if there is a leading v or a version
listed without all 3 parts (e.g. `v1.2`) it will attempt to coerce it into a valid
semantic version (e.g., 1.2.0). In both cases a `Version` object is returned
that can be sorted, compared, and used in constraints.

When parsing a version an error is returned if there is an issue parsing the
version. For example,

    v, err := semver.NewVersion("1.2.3-beta.1+build345")

The version object has methods to get the parts of the version, compare it to
other versions, convert the version back into a string, and get the original
string. Getting the original string is useful if the semantic version was coerced
into a valid form.

There are package level variables that affect how `NewVersion` handles parsing.

- `CoerceNewVersion` is `true` by default. When set to `true` it coerces non-compliant
  versions into SemVer. For example, allowing a leading 0 in a major, minor, or patch
  part. This enables the use of CalVer in versions even when not compliant with SemVer.
  When set to `false` less coercion work is done.
- `DetailedNewVersionErrors` provides more detailed errors. It only has an affect when
  `CoerceNewVersion` is set to `false`. When `DetailedNewVersionErrors` is set to `true`
  it can provide some more insight into why a version is invalid. Setting
  `DetailedNewVersionErrors` to `false` is faster on performance but provides less
  detailed error messages if a version fails to parse.

## Sorting Semantic Versions

A set of versions can be sorted using the `sort` package from the standard library.
For example,

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

There are two methods for comparing versions. One uses comparison methods on
`Version` instances and the other uses `Constraints`. There are some important
differences to notes between these two methods of comparison.

1. When two versions are compared using functions such as `Compare`, `LessThan`,
   and others it will follow the specification and always include pre-releases
   within the comparison. It will provide an answer that is valid with the
   comparison section of the spec at https://semver.org/#spec-item-11
2. When constraint checking is used for checks or validation it will follow a
   different set of rules that are common for ranges with tools like npm/js
   and Rust/Cargo. This includes considering pre-releases to be invalid if the
   ranges does not include one. If you want to have it include pre-releases a
   simple solution is to include `-0` in your range.
3. Constraint ranges can have some complex rules including the shorthand use of
   ~ and ^. For more details on those see the options below.

There are differences between the two methods or checking versions because the
comparison methods on `Version` follow the specification while comparison ranges
are not part of the specification. Different packages and tools have taken it
upon themselves to come up with range rules. This has resulted in differences.
For example, npm/js and Cargo/Rust follow similar patterns while PHP has a
different pattern for ^. The comparison features in this package follow the
npm/js and Cargo/Rust lead because applications using it have followed similar
patters with their versions.

Checking a version against version constraints is one of the most featureful
parts of the package.

```go
c, err := semver.NewConstraint(">= 1.2.3")
if err != nil {
    // Handle constraint not being parsable.
}

v, err := semver.NewVersion("1.3")
if err != nil {
    // Handle version not being parsable.
}
// Check if the version meets the constraints. The variable a will be true.
a := c.Check(v)
```

### Basic Comparisons

There are two elements to the comparisons. First, a comparison string is a list
of space or comma separated AND comparisons. These are then separated by || (OR)
comparisons. For example, `">= 1.2 < 3.0.0 || >= 4.2.3"` is looking for a
comparison that's greater than or equal to 1.2 and less than 3.0.0 or is
greater than or equal to 4.2.3.

The basic comparisons are:

* `=`: equal (aliased to no operator)
* `!=`: not equal
* `>`: greater than
* `<`: less than
* `>=`: greater than or equal to
* `<=`: less than or equal to

### Working With Prerelease Versions

Pre-releases, for those not familiar with them, are used for software releases
prior to stable or generally available releases. Examples of pre-releases include
development, alpha, beta, and release candidate releases. A pre-release may be
a version such as `1.2.3-beta.1` while the stable release would be `1.2.3`. In the
order of precedence, pre-releases come before their associated releases. In this
example `1.2.3-beta.1 < 1.2.3`.

According to the Semantic Version specification, pre-releases may not be
API compliant with their release counterpart. It says,

> A pre-release version indicates that the version is unstable and might not satisfy the intended compatibility requirements as denoted by its associated normal version.

SemVer's comparisons using constraints without a pre-release comparator will skip
pre-release versions. For example, `>=1.2.3` will skip pre-releases when looking
at a list of releases while `>=1.2.3-0` will evaluate and find pre-releases.

The reason for the `0` as a pre-release version in the example comparison is
because pre-releases can only contain ASCII alphanumerics and hyphens (along with
`.` separators), per the spec. Sorting happens in ASCII sort order, again per the
spec. The lowest character is a `0` in ASCII sort order
(see an [ASCII Table](http://www.asciitable.com/))

Understanding ASCII sort ordering is important because A-Z comes before a-z. That
means `>=1.2.3-BETA` will return `1.2.3-alpha`. What you might expect from case
sensitivity doesn't apply here. This is due to ASCII sort ordering which is what
the spec specifies.

The `Constraints` instance returned from `semver.NewConstraint()` has a property
`IncludePrerelease` that, when set to true, will return prerelease versions when calls
to `Check()` and `Validate()` are made.

### Hyphen Range Comparisons

There are multiple methods to handle ranges and the first is hyphens ranges.
These look like:

* `1.2 - 1.4.5` which is equivalent to `>= 1.2 <= 1.4.5`
* `2.3.4 - 4.5` which is equivalent to `>= 2.3.4 <= 4.5`

Note that `1.2-1.4.5` without whitespace is parsed completely differently; it's
parsed as a single constraint `1.2.0` with _prerelease_ `1.4.5`.

### Wildcards In Comparisons

The `x`, `X`, and `*` characters can be used as a wildcard character. This works
for all comparison operators. When used on the `=` operator it falls
back to the patch level comparison (see tilde below). For example,

* `1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
* `>= 1.2.x` is equivalent to `>= 1.2.0`
* `<= 2.x` is equivalent to `< 3`
* `*` is equivalent to `>= 0.0.0`

### Tilde Range Comparisons (Patch)

The tilde (`~`) comparison operator is for patch level ranges when a minor
version is specified and major level changes when the minor number is missing.
For example,

* `~1.2.3` is equivalent to `>= 1.2.3, < 1.3.0`
* `~1` is equivalent to `>= 1, < 2`
* `~2.3` is equivalent to `>= 2.3, < 2.4`
* `~1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
* `~1.x` is equivalent to `>= 1, < 2`

### Caret Range Comparisons (Major)

The caret (`^`) comparison operator is for major level changes once a stable
(1.0.0) release has occurred. Prior to a 1.0.0 release the minor versions acts
as the API stability level. This is useful when comparisons of API versions as a
major change is API breaking. For example,

* `^1.2.3` is equivalent to `>= 1.2.3, < 2.0.0`
* `^1.2.x` is equivalent to `>= 1.2.0, < 2.0.0`
* `^2.3` is equivalent to `>= 2.3, < 3`
* `^2.x` is equivalent to `>= 2.0.0, < 3`
* `^0.2.3` is equivalent to `>=0.2.3 <0.3.0`
* `^0.2` is equivalent to `>=0.2.0 <0.3.0`
* `^0.0.3` is equivalent to `>=0.0.3 <0.0.4`
* `^0.0` is equivalent to `>=0.0.0 <0.1.0`
* `^0` is equivalent to `>=0.0.0 <1.0.0`

## Validation

In addition to testing a version against a constraint, a version can be validated
against a constraint. When validation fails a slice of errors containing why a
version didn't meet the constraint is returned. For example,

```go
c, err := semver.NewConstraint("<= 1.2.3, >= 1.4")
if err != nil {
    // Handle constraint not being parseable.
}

v, err := semver.NewVersion("1.3")
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

## Contribute

If you find an issue or want to contribute please file an [issue](https://github.com/Masterminds/semver/issues)
or [create a pull request](https://github.com/Masterminds/semver/pulls).

## Security

Security is an important consideration for this project. The project currently
uses the following tools to help discover security issues:

* [CodeQL](https://codeql.github.com)
* [gosec](https://github.com/securego/gosec)
* Daily Fuzz testing

If you believe you have found a security vulnerability you can privately disclose
it through the [GitHub security page](https://github.com/Masterminds/semver/security).
