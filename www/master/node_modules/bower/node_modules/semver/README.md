semver(1) -- The semantic versioner for npm
===========================================

## Usage

    $ npm install semver

    semver.valid('1.2.3') // '1.2.3'
    semver.valid('a.b.c') // null
    semver.clean('  =v1.2.3   ') // '1.2.3'
    semver.satisfies('1.2.3', '1.x || >=2.5.0 || 5.0.0 - 7.2.3') // true
    semver.gt('1.2.3', '9.8.7') // false
    semver.lt('1.2.3', '9.8.7') // true

As a command-line utility:

    $ semver -h

    Usage: semver <version> [<version> [...]] [-r <range> | -i <inc> | -d <dec>]
    Test if version(s) satisfy the supplied range(s), and sort them.

    Multiple versions or ranges may be supplied, unless increment
    or decrement options are specified.  In that case, only a single
    version may be used, and it is incremented by the specified level

    Program exits successfully if any valid version satisfies
    all supplied ranges, and prints all satisfying versions.

    If no versions are valid, or ranges are not satisfied,
    then exits failure.

    Versions are printed in ascending order, so supplying
    multiple versions to the utility will just sort them.

## Versions

A "version" is described by the `v2.0.0` specification found at
<http://semver.org/>.

A leading `"="` or `"v"` character is stripped off and ignored.

## Ranges

The following range styles are supported:

* `1.2.3` A specific version.  When nothing else will do.  Must be a full
  version number, with major, minor, and patch versions specified.
  Note that build metadata is still ignored, so `1.2.3+build2012` will
  satisfy this range.
* `>1.2.3` Greater than a specific version.
* `<1.2.3` Less than a specific version.  If there is no prerelease
  tag on the version range, then no prerelease version will be allowed
  either, even though these are technically "less than".
* `>=1.2.3` Greater than or equal to.  Note that prerelease versions
  are NOT equal to their "normal" equivalents, so `1.2.3-beta` will
  not satisfy this range, but `2.3.0-beta` will.
* `<=1.2.3` Less than or equal to.  In this case, prerelease versions
  ARE allowed, so `1.2.3-beta` would satisfy.
* `1.2.3 - 2.3.4` := `>=1.2.3 <=2.3.4`
* `~1.2.3` := `>=1.2.3-0 <1.3.0-0`  "Reasonably close to `1.2.3`".  When
  using tilde operators, prerelease versions are supported as well,
  but a prerelease of the next significant digit will NOT be
  satisfactory, so `1.3.0-beta` will not satisfy `~1.2.3`.
* `^1.2.3` := `>=1.2.3-0 <2.0.0-0`  "Compatible with `1.2.3`".  When
  using caret operators, anything from the specified version (including
  prerelease) will be supported up to, but not including, the next
  major version (or its prereleases). `1.5.1` will satisfy `^1.2.3`,
  while `1.2.2` and `2.0.0-beta` will not.
* `^0.1.3` := `>=0.1.3-0 <0.2.0-0` "Compatible with `0.1.3`". `0.x.x` versions are
  special: the first non-zero component indicates potentially breaking changes,
  meaning the caret operator matches any version with the same first non-zero
  component starting at the specified version.
* `^0.0.2` := `=0.0.2` "Only the version `0.0.2` is considered compatible"
* `~1.2` := `>=1.2.0-0 <1.3.0-0` "Any version starting with `1.2`"
* `^1.2` := `>=1.2.0-0 <2.0.0-0` "Any version compatible with `1.2`"
* `1.2.x` := `>=1.2.0-0 <1.3.0-0` "Any version starting with `1.2`"
* `1.2.*` Same as `1.2.x`.
* `1.2` Same as `1.2.x`.
* `~1` := `>=1.0.0-0 <2.0.0-0` "Any version starting with `1`"
* `^1` := `>=1.0.0-0 <2.0.0-0` "Any version compatible with `1`"
* `1.x` := `>=1.0.0-0 <2.0.0-0` "Any version starting with `1`"
* `1.*` Same as `1.x`.
* `1` Same as `1.x`.
* `*` Any version whatsoever.
* `x` Same as `*`.
* `""` (just an empty string) Same as `*`.


Ranges can be joined with either a space (which implies "and") or a
`||` (which implies "or").

## Functions

All methods and classes take a final `loose` boolean argument that, if
true, will be more forgiving about not-quite-valid semver strings.
The resulting output will always be 100% strict, of course.

Strict-mode Comparators and Ranges will be strict about the SemVer
strings that they parse.

* `valid(v)`: Return the parsed version, or null if it's not valid.
* `inc(v, release)`: Return the version incremented by the release
  type (`major`,   `premajor`, `minor`, `preminor`, `patch`,
  `prepatch`, or `prerelease`), or null if it's not valid
  * `premajor` in one call will bump the version up to the next major
    version and down to a prerelease of that major version.
    `preminor`, and `prepatch` work the same way.
  * If called from a non-prerelease version, the `prerelease` will work the
    same as `prepatch`. It increments the patch version, then makes a
    prerelease. If the input version is already a prerelease it simply
    increments it.

### Comparison

* `gt(v1, v2)`: `v1 > v2`
* `gte(v1, v2)`: `v1 >= v2`
* `lt(v1, v2)`: `v1 < v2`
* `lte(v1, v2)`: `v1 <= v2`
* `eq(v1, v2)`: `v1 == v2` This is true if they're logically equivalent,
  even if they're not the exact same string.  You already know how to
  compare strings.
* `neq(v1, v2)`: `v1 != v2` The opposite of `eq`.
* `cmp(v1, comparator, v2)`: Pass in a comparison string, and it'll call
  the corresponding function above.  `"==="` and `"!=="` do simple
  string comparison, but are included for completeness.  Throws if an
  invalid comparison string is provided.
* `compare(v1, v2)`: Return `0` if `v1 == v2`, or `1` if `v1` is greater, or `-1` if
  `v2` is greater.  Sorts in ascending order if passed to `Array.sort()`.
* `rcompare(v1, v2)`: The reverse of compare.  Sorts an array of versions
  in descending order when passed to `Array.sort()`.


### Ranges

* `validRange(range)`: Return the valid range or null if it's not valid
* `satisfies(version, range)`: Return true if the version satisfies the
  range.
* `maxSatisfying(versions, range)`: Return the highest version in the list
  that satisfies the range, or `null` if none of them do.
* `gtr(version, range)`: Return `true` if version is greater than all the
  versions possible in the range.
* `ltr(version, range)`: Return `true` if version is less than all the
  versions possible in the range.
* `outside(version, range, hilo)`: Return true if the version is outside
  the bounds of the range in either the high or low direction.  The
  `hilo` argument must be either the string `'>'` or `'<'`.  (This is
  the function called by `gtr` and `ltr`.)

Note that, since ranges may be non-contiguous, a version might not be
greater than a range, less than a range, *or* satisfy a range!  For
example, the range `1.2 <1.2.9 || >2.0.0` would have a hole from `1.2.9`
until `2.0.0`, so the version `1.2.10` would not be greater than the
range (because `2.0.1` satisfies, which is higher), nor less than the
range (since `1.2.8` satisfies, which is lower), and it also does not
satisfy the range.

If you want to know if a version satisfies or does not satisfy a
range, use the `satisfies(version, range)` function.
