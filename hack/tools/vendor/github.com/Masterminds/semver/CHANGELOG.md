# 1.5.0 (2019-09-11)

## Added

- #103: Add basic fuzzing for `NewVersion()` (thanks @jesse-c)

## Changed

- #82: Clarify wildcard meaning in range constraints and update tests for it (thanks @greysteil)
- #83: Clarify caret operator range for pre-1.0.0 dependencies (thanks @greysteil)
- #72: Adding docs comment pointing to vert for a cli
- #71: Update the docs on pre-release comparator handling
- #89: Test with new go versions (thanks @thedevsaddam)
- #87: Added $ to ValidPrerelease for better validation (thanks @jeremycarroll)

## Fixed

- #78: Fix unchecked error in example code (thanks @ravron)
- #70: Fix the handling of pre-releases and the 0.0.0 release edge case
- #97: Fixed copyright file for proper display on GitHub
- #107: Fix handling prerelease when sorting alphanum and num 
- #109: Fixed where Validate sometimes returns wrong message on error

# 1.4.2 (2018-04-10)

## Changed
- #72: Updated the docs to point to vert for a console appliaction
- #71: Update the docs on pre-release comparator handling

## Fixed
- #70: Fix the handling of pre-releases and the 0.0.0 release edge case

# 1.4.1 (2018-04-02)

## Fixed
- Fixed #64: Fix pre-release precedence issue (thanks @uudashr)

# 1.4.0 (2017-10-04)

## Changed
- #61: Update NewVersion to parse ints with a 64bit int size (thanks @zknill)

# 1.3.1 (2017-07-10)

## Fixed
- Fixed #57: number comparisons in prerelease sometimes inaccurate

# 1.3.0 (2017-05-02)

## Added
- #45: Added json (un)marshaling support (thanks @mh-cbon)
- Stability marker. See https://masterminds.github.io/stability/

## Fixed
- #51: Fix handling of single digit tilde constraint (thanks @dgodd)

## Changed
- #55: The godoc icon moved from png to svg

# 1.2.3 (2017-04-03)

## Fixed
- #46: Fixed 0.x.x and 0.0.x in constraints being treated as *

# Release 1.2.2 (2016-12-13)

## Fixed
- #34: Fixed issue where hyphen range was not working with pre-release parsing.

# Release 1.2.1 (2016-11-28)

## Fixed
- #24: Fixed edge case issue where constraint "> 0" does not handle "0.0.1-alpha"
  properly.

# Release 1.2.0 (2016-11-04)

## Added
- #20: Added MustParse function for versions (thanks @adamreese)
- #15: Added increment methods on versions (thanks @mh-cbon)

## Fixed
- Issue #21: Per the SemVer spec (section 9) a pre-release is unstable and
  might not satisfy the intended compatibility. The change here ignores pre-releases
  on constraint checks (e.g., ~ or ^) when a pre-release is not part of the
  constraint. For example, `^1.2.3` will ignore pre-releases while
  `^1.2.3-alpha` will include them.

# Release 1.1.1 (2016-06-30)

## Changed
- Issue #9: Speed up version comparison performance (thanks @sdboyer)
- Issue #8: Added benchmarks (thanks @sdboyer)
- Updated Go Report Card URL to new location
- Updated Readme to add code snippet formatting (thanks @mh-cbon)
- Updating tagging to v[SemVer] structure for compatibility with other tools.

# Release 1.1.0 (2016-03-11)

- Issue #2: Implemented validation to provide reasons a versions failed a
  constraint.

# Release 1.0.1 (2015-12-31)

- Fixed #1: * constraint failing on valid versions.

# Release 1.0.0 (2015-10-20)

- Initial release
