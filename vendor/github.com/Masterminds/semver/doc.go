/*
Package semver provides the ability to work with Semantic Versions (http://semver.org) in Go.

Specifically it provides the ability to:

    * Parse semantic versions
    * Sort semantic versions
    * Check if a semantic version fits within a set of constraints
    * Optionally work with a `v` prefix

Parsing Semantic Versions

To parse a semantic version use the `NewVersion` function. For example,

    v, err := semver.NewVersion("1.2.3-beta.1+build345")

If there is an error the version wasn't parseable. The version object has methods
to get the parts of the version, compare it to other versions, convert the
version back into a string, and get the original string. For more details
please see the documentation at https://godoc.org/github.com/Masterminds/semver.

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

Checking Version Constraints

Checking a version against version constraints is one of the most featureful
parts of the package.

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

Basic Comparisons

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

Hyphen Range Comparisons

There are multiple methods to handle ranges and the first is hyphens ranges.
These look like:

    * `1.2 - 1.4.5` which is equivalent to `>= 1.2, <= 1.4.5`
    * `2.3.4 - 4.5` which is equivalent to `>= 2.3.4, <= 4.5`

Wildcards In Comparisons

The `x`, `X`, and `*` characters can be used as a wildcard character. This works
for all comparison operators. When used on the `=` operator it falls
back to the pack level comparison (see tilde below). For example,

    * `1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
    * `>= 1.2.x` is equivalent to `>= 1.2.0`
    * `<= 2.x` is equivalent to `<= 3`
    * `*` is equivalent to `>= 0.0.0`

Tilde Range Comparisons (Patch)

The tilde (`~`) comparison operator is for patch level ranges when a minor
version is specified and major level changes when the minor number is missing.
For example,

    * `~1.2.3` is equivalent to `>= 1.2.3, < 1.3.0`
    * `~1` is equivalent to `>= 1, < 2`
    * `~2.3` is equivalent to `>= 2.3, < 2.4`
    * `~1.2.x` is equivalent to `>= 1.2.0, < 1.3.0`
    * `~1.x` is equivalent to `>= 1, < 2`

Caret Range Comparisons (Major)

The caret (`^`) comparison operator is for major level changes. This is useful
when comparisons of API versions as a major change is API breaking. For example,

    * `^1.2.3` is equivalent to `>= 1.2.3, < 2.0.0`
    * `^1.2.x` is equivalent to `>= 1.2.0, < 2.0.0`
    * `^2.3` is equivalent to `>= 2.3, < 3`
    * `^2.x` is equivalent to `>= 2.0.0, < 3`
*/
package semver
