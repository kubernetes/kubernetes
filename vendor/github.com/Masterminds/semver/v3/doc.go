/*
Package semver provides the ability to work with Semantic Versions (http://semver.org) in Go.

Specifically it provides the ability to:

    * Parse semantic versions
    * Sort semantic versions
    * Check if a semantic version fits within a set of constraints
    * Optionally work with a `v` prefix

Parsing Semantic Versions

There are two functions that can parse semantic versions. The `StrictNewVersion`
function only parses valid version 2 semantic versions as outlined in the
specification. The `NewVersion` function attempts to coerce a version into a
semantic version and parse it. For example, if there is a leading v or a version
listed without all 3 parts (e.g. 1.2) it will attempt to coerce it into a valid
semantic version (e.g., 1.2.0). In both cases a `Version` object is returned
that can be sorted, compared, and used in constraints.

When parsing a version an optional error can be returned if there is an issue
parsing the version. For example,

    v, err := semver.NewVersion("1.2.3-beta.1+build345")

The version object has methods to get the parts of the version, compare it to
other versions, convert the version back into a string, and get the original
string. For more details please see the documentation
at https://godoc.org/github.com/Masterminds/semver.

Sorting Semantic Versions

A set of versions can be sorted using the `sort` package from the standard library.
For example,

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

Checking Version Constraints and Comparing Versions

There are two methods for comparing versions. One uses comparison methods on
`Version` instances and the other is using Constraints. There are some important
differences to notes between these two methods of comparison.

1. When two versions are compared using functions such as `Compare`, `LessThan`,
   and others it will follow the specification and always include prereleases
   within the comparison. It will provide an answer valid with the comparison
   spec section at https://semver.org/#spec-item-11
2. When constraint checking is used for checks or validation it will follow a
   different set of rules that are common for ranges with tools like npm/js
   and Rust/Cargo. This includes considering prereleases to be invalid if the
   ranges does not include on. If you want to have it include pre-releases a
   simple solution is to include `-0` in your range.
3. Constraint ranges can have some complex rules including the shorthard use of
   ~ and ^. For more details on those see the options below.

There are differences between the two methods or checking versions because the
comparison methods on `Version` follow the specification while comparison ranges
are not part of the specification. Different packages and tools have taken it
upon themselves to come up with range rules. This has resulted in differences.
For example, npm/js and Cargo/Rust follow similar patterns which PHP has a
different pattern for ^. The comparison features in this package follow the
npm/js and Cargo/Rust lead because applications using it have followed similar
patters with their versions.

Checking a version against version constraints is one of the most featureful
parts of the package.

    c, err := semver.NewConstraint(">= 1.2.3")
    if err != nil {
        // Handle constraint not being parsable.
    }

    v, err := semver.NewVersion("1.3")
    if err != nil {
        // Handle version not being parsable.
    }
    // Check if the version meets the constraints. The a variable will be true.
    a := c.Check(v)

Basic Comparisons

There are two elements to the comparisons. First, a comparison string is a list
of comma or space separated AND comparisons. These are then separated by || (OR)
comparisons. For example, `">= 1.2 < 3.0.0 || >= 4.2.3"` is looking for a
comparison that's greater than or equal to 1.2 and less than 3.0.0 or is
greater than or equal to 4.2.3. This can also be written as
`">= 1.2, < 3.0.0 || >= 4.2.3"`

The basic comparisons are:

    * `=`: equal (aliased to no operator)
    * `!=`: not equal
    * `>`: greater than
    * `<`: less than
    * `>=`: greater than or equal to
    * `<=`: less than or equal to

Hyphen Range Comparisons

There are multiple methods to handle ranges and the first is hyphens ranges.
These look like:

    * `1.2 - 1.4.5` which is equivalent to `>= 1.2, <= 1.4.5`
    * `2.3.4 - 4.5` which is equivalent to `>= 2.3.4 <= 4.5`

Wildcards In Comparisons

The `x`, `X`, and `*` characters can be used as a wildcard character. This works
for all comparison operators. When used on the `=` operator it falls
back to the tilde operation. For example,

    * `1.2.x` is equivalent to `>= 1.2.0 < 1.3.0`
    * `>= 1.2.x` is equivalent to `>= 1.2.0`
    * `<= 2.x` is equivalent to `<= 3`
    * `*` is equivalent to `>= 0.0.0`

Tilde Range Comparisons (Patch)

The tilde (`~`) comparison operator is for patch level ranges when a minor
version is specified and major level changes when the minor number is missing.
For example,

    * `~1.2.3` is equivalent to `>= 1.2.3 < 1.3.0`
    * `~1` is equivalent to `>= 1, < 2`
    * `~2.3` is equivalent to `>= 2.3 < 2.4`
    * `~1.2.x` is equivalent to `>= 1.2.0 < 1.3.0`
    * `~1.x` is equivalent to `>= 1 < 2`

Caret Range Comparisons (Major)

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

Validation

In addition to testing a version against a constraint, a version can be validated
against a constraint. When validation fails a slice of errors containing why a
version didn't meet the constraint is returned. For example,

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
*/
package semver
