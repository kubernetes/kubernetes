package semver

import (
	"bytes"
	"fmt"
)

var rangeErrs = [...]string{
	"%s is less than the minimum of %s",
	"%s is less than or equal to the minimum of %s",
	"%s is greater than the maximum of %s",
	"%s is greater than or equal to the maximum of %s",
	"%s is specifically disallowed in %s",
	"%s has prerelease data, so is omitted by the range %s",
}

const (
	rerrLT = iota
	rerrLTE
	rerrGT
	rerrGTE
	rerrNE
	rerrPre
)

// MatchFailure is an interface for failures to find a Constraint match.
type MatchFailure interface {
	error

	// Pair returns the version and constraint that did not match prompting
	// the error.
	Pair() (v Version, c Constraint)
}

// RangeMatchFailure occurs when a version is not within a constraint range.
type RangeMatchFailure struct {
	v   Version
	rc  rangeConstraint
	typ int8
}

func (rce RangeMatchFailure) Error() string {
	return fmt.Sprintf(rangeErrs[rce.typ], rce.v, rce.rc)
}

// Pair returns the version and constraint that did not match. Part of the
// MatchFailure interface.
func (rce RangeMatchFailure) Pair() (v Version, r Constraint) {
	return rce.v, rce.rc
}

// VersionMatchFailure occurs when two versions do not match each other.
type VersionMatchFailure struct {
	v, other Version
}

func (vce VersionMatchFailure) Error() string {
	return fmt.Sprintf("%s is not equal to %s", vce.v, vce.other)
}

// Pair returns the two versions that did not match. Part of the
// MatchFailure interface.
func (vce VersionMatchFailure) Pair() (v Version, r Constraint) {
	return vce.v, vce.other
}

// MultiMatchFailure errors occur when there are multiple constraints a version
// is being checked against and there are failures.
type MultiMatchFailure []MatchFailure

func (mmf MultiMatchFailure) Error() string {
	var buf bytes.Buffer

	for k, e := range mmf {
		if k < len(mmf)-1 {
			fmt.Fprintf(&buf, "%s\n", e)
		} else {
			fmt.Fprintf(&buf, e.Error())
		}
	}

	return buf.String()
}
