package semver

import (
	"fmt"
	"sort"
	"strings"
)

type rangeConstraint struct {
	min, max               Version
	includeMin, includeMax bool
	excl                   []Version
}

func (rc rangeConstraint) Matches(v Version) error {
	var fail bool
	ispre := v.Prerelease() != ""

	rce := RangeMatchFailure{
		v:  v,
		rc: rc,
	}

	if !rc.minIsZero() {
		cmp := rc.min.Compare(v)
		if rc.includeMin {
			rce.typ = rerrLT
			fail = cmp == 1
		} else {
			rce.typ = rerrLTE
			fail = cmp != -1
		}

		if fail {
			return rce
		}
	}

	if !rc.maxIsInf() {
		cmp := rc.max.Compare(v)
		if rc.includeMax {
			rce.typ = rerrGT
			fail = cmp == -1
		} else {
			rce.typ = rerrGTE
			fail = cmp != 1

		}

		if fail {
			return rce
		}
	}

	for _, excl := range rc.excl {
		if excl.Equal(v) {
			rce.typ = rerrNE
			return rce
		}
	}

	// If the incoming version has prerelease info, it's usually a match failure
	// - unless all the numeric parts are equal between the incoming and the
	// minimum.
	if !fail && ispre && !numPartsEq(rc.min, v) {
		rce.typ = rerrPre
		return rce
	}

	return nil
}

func (rc rangeConstraint) dup() rangeConstraint {
	// Only need to do anything if there are some excludes
	if len(rc.excl) == 0 {
		return rc
	}

	var excl []Version
	excl = make([]Version, len(rc.excl))
	copy(excl, rc.excl)

	return rangeConstraint{
		min:        rc.min,
		max:        rc.max,
		includeMin: rc.includeMin,
		includeMax: rc.includeMax,
		excl:       excl,
	}
}

func (rc rangeConstraint) minIsZero() bool {
	return rc.min.special == zeroVersion
}

func (rc rangeConstraint) maxIsInf() bool {
	return rc.max.special == infiniteVersion
}

func (rc rangeConstraint) Intersect(c Constraint) Constraint {
	switch oc := c.(type) {
	case any:
		return rc
	case none:
		return None()
	case unionConstraint:
		return oc.Intersect(rc)
	case Version:
		if err := rc.Matches(oc); err != nil {
			return None()
		}
		return c
	case rangeConstraint:
		nr := rangeConstraint{
			min:        rc.min,
			max:        rc.max,
			includeMin: rc.includeMin,
			includeMax: rc.includeMax,
		}

		if !oc.minIsZero() {
			if nr.minIsZero() || nr.min.LessThan(oc.min) {
				nr.min = oc.min
				nr.includeMin = oc.includeMin
			} else if oc.min.Equal(nr.min) && !oc.includeMin {
				// intersection means we must follow the least inclusive
				nr.includeMin = false
			}
		}

		if !oc.maxIsInf() {
			if nr.maxIsInf() || nr.max.GreaterThan(oc.max) {
				nr.max = oc.max
				nr.includeMax = oc.includeMax
			} else if oc.max.Equal(nr.max) && !oc.includeMax {
				// intersection means we must follow the least inclusive
				nr.includeMax = false
			}
		}

		// Ensure any applicable excls from oc are included in nc
		for _, e := range append(rc.excl, oc.excl...) {
			if nr.Matches(e) == nil {
				nr.excl = append(nr.excl, e)
			}
		}

		if nr.minIsZero() || nr.maxIsInf() {
			return nr
		}

		if nr.min.Equal(nr.max) {
			// min and max are equal. if range is inclusive, return that
			// version; otherwise, none
			if nr.includeMin && nr.includeMax {
				return nr.min
			}
			return None()
		}

		if nr.min.GreaterThan(nr.max) {
			// min is greater than max - not possible, so we return none
			return None()
		}

		// range now fully validated, return what we have
		return nr

	default:
		panic("unknown type")
	}
}

func (rc rangeConstraint) Union(c Constraint) Constraint {
	switch oc := c.(type) {
	case any:
		return Any()
	case none:
		return rc
	case unionConstraint:
		return Union(rc, oc)
	case Version:
		if err := rc.Matches(oc); err == nil {
			return rc
		} else if len(rc.excl) > 0 { // TODO (re)checking like this is wasteful
			// ensure we don't have an excl-specific mismatch; if we do, remove
			// it and return that
			for k, e := range rc.excl {
				if e.Equal(oc) {
					excl := make([]Version, len(rc.excl)-1)

					if k == len(rc.excl)-1 {
						copy(excl, rc.excl[:k])
					} else {
						copy(excl, append(rc.excl[:k], rc.excl[k+1:]...))
					}

					return rangeConstraint{
						min:        rc.min,
						max:        rc.max,
						includeMin: rc.includeMin,
						includeMax: rc.includeMax,
						excl:       excl,
					}
				}
			}
		}

		if oc.LessThan(rc.min) {
			return unionConstraint{oc, rc.dup()}
		}
		if oc.Equal(rc.min) {
			ret := rc.dup()
			ret.includeMin = true
			return ret
		}
		if oc.Equal(rc.max) {
			ret := rc.dup()
			ret.includeMax = true
			return ret
		}
		// Only possibility left is gt
		return unionConstraint{rc.dup(), oc}
	case rangeConstraint:
		if (rc.minIsZero() && oc.maxIsInf()) || (rc.maxIsInf() && oc.minIsZero()) {
			rcl, ocl := len(rc.excl), len(oc.excl)
			// Quick check for open case
			if rcl == 0 && ocl == 0 {
				return Any()
			}

			// This is inefficient, but it's such an absurdly corner case...
			if len(dedupeExcls(rc.excl, oc.excl)) == rcl+ocl {
				// If deduped excludes are the same length as the individual
				// excludes, then they have no overlapping elements, so the
				// union knocks out the excludes and we're back to Any.
				return Any()
			}

			// There's at least some dupes, which are all we need to include
			nc := rangeConstraint{
				min: Version{special: zeroVersion},
				max: Version{special: infiniteVersion},
			}
			for _, e1 := range rc.excl {
				for _, e2 := range oc.excl {
					if e1.Equal(e2) {
						nc.excl = append(nc.excl, e1)
					}
				}
			}

			return nc
		} else if areAdjacent(rc, oc) {
			// Receiver adjoins the input from below
			nc := rc.dup()

			nc.max = oc.max
			nc.includeMax = oc.includeMax
			nc.excl = append(nc.excl, oc.excl...)

			return nc
		} else if areAdjacent(oc, rc) {
			// Input adjoins the receiver from below
			nc := oc.dup()

			nc.max = rc.max
			nc.includeMax = rc.includeMax
			nc.excl = append(nc.excl, rc.excl...)

			return nc

		} else if rc.MatchesAny(oc) {
			// Receiver and input overlap; form a new range accordingly.
			nc := rangeConstraint{
				min: Version{special: zeroVersion},
				max: Version{special: infiniteVersion},
			}

			// For efficiency, we simultaneously determine if either of the
			// ranges are supersets of the other, while also selecting the min
			// and max of the new range
			var info uint8

			const (
				lminlt uint8             = 1 << iota // left (rc) min less than right
				rminlt                               // right (oc) min less than left
				lmaxgt                               // left max greater than right
				rmaxgt                               // right max greater than left
				lsupr  = lminlt | lmaxgt             // left is superset of right
				rsupl  = rminlt | rmaxgt             // right is superset of left
			)

			// Pick the min
			if !rc.minIsZero() {
				if oc.minIsZero() || rc.min.GreaterThan(oc.min) || (rc.min.Equal(oc.min) && !rc.includeMin && oc.includeMin) {
					info |= rminlt
					nc.min = oc.min
					nc.includeMin = oc.includeMin
				} else {
					info |= lminlt
					nc.min = rc.min
					nc.includeMin = rc.includeMin
				}
			} else if !oc.minIsZero() {
				info |= lminlt
				nc.min = rc.min
				nc.includeMin = rc.includeMin
			}

			// Pick the max
			if !rc.maxIsInf() {
				if oc.maxIsInf() || rc.max.LessThan(oc.max) || (rc.max.Equal(oc.max) && !rc.includeMax && oc.includeMax) {
					info |= rmaxgt
					nc.max = oc.max
					nc.includeMax = oc.includeMax
				} else {
					info |= lmaxgt
					nc.max = rc.max
					nc.includeMax = rc.includeMax
				}
			} else if oc.maxIsInf() {
				info |= lmaxgt
				nc.max = rc.max
				nc.includeMax = rc.includeMax
			}

			// Reincorporate any excluded versions
			if info&lsupr != lsupr {
				// rc is not superset of oc, so must walk oc.excl
				for _, e := range oc.excl {
					if rc.Matches(e) != nil {
						nc.excl = append(nc.excl, e)
					}
				}
			}

			if info&rsupl != rsupl {
				// oc is not superset of rc, so must walk rc.excl
				for _, e := range rc.excl {
					if oc.Matches(e) != nil {
						nc.excl = append(nc.excl, e)
					}
				}
			}

			return nc
		} else {
			// Don't call Union() here b/c it would duplicate work
			uc := constraintList{rc, oc}
			sort.Sort(uc)
			return unionConstraint(uc)
		}
	}

	panic("unknown type")
}

// isSupersetOf computes whether the receiver rangeConstraint is a superset of
// the passed rangeConstraint.
//
// This is NOT a strict superset comparison, so identical ranges will both
// report being supersets of each other.
//
// Note also that this does *not* compare excluded versions - it only compares
// range endpoints.
func (rc rangeConstraint) isSupersetOf(rc2 rangeConstraint) bool {
	if !rc.minIsZero() {
		if rc2.minIsZero() || rc.min.GreaterThan(rc2.min) || (rc.min.Equal(rc2.min) && !rc.includeMin && rc2.includeMin) {
			return false
		}
	}

	if !rc.maxIsInf() {
		if rc2.maxIsInf() || rc.max.LessThan(rc2.max) || (rc.max.Equal(rc2.max) && !rc.includeMax && rc2.includeMax) {
			return false
		}
	}

	return true
}

func (rc rangeConstraint) String() string {
	return rc.toString(false)
}

func (rc rangeConstraint) ImpliedCaretString() string {
	return rc.toString(true)
}

func (rc rangeConstraint) toString(impliedCaret bool) string {
	var pieces []string

	// We need to trigger the standard verbose handling from various points, so
	// wrap it in a function.
	noshort := func() {
		if !rc.minIsZero() {
			if rc.includeMin {
				pieces = append(pieces, fmt.Sprintf(">=%s", rc.min))
			} else {
				pieces = append(pieces, fmt.Sprintf(">%s", rc.min))
			}
		}

		if !rc.maxIsInf() {
			if rc.includeMax {
				pieces = append(pieces, fmt.Sprintf("<=%s", rc.max))
			} else {
				pieces = append(pieces, fmt.Sprintf("<%s", rc.max))
			}
		}
	}

	// Handle the possibility that we might be able to express the range
	// with a caret or tilde, as we prefer those forms.
	var caretstr string
	if impliedCaret {
		caretstr = "%s"
	} else {
		caretstr = "^%s"
	}

	switch {
	case rc.minIsZero() && rc.maxIsInf():
		// This if is internal because it's useful to know for the other cases
		// that we don't have special values at both bounds
		if len(rc.excl) == 0 {
			// Shouldn't be possible to reach from anything that can be done
			// outside the package, but best to cover it and be safe
			return "*"
		}
	case rc.minIsZero(), rc.includeMax, !rc.includeMin:
		// tilde and caret could never apply here
		noshort()
	case !rc.maxIsInf() && rc.max.Minor() == 0 && rc.max.Patch() == 0: // basic caret
		if rc.min.Major() == rc.max.Major()-1 && rc.min.Major() != 0 {
			pieces = append(pieces, fmt.Sprintf(caretstr, rc.min))
		} else {
			// range is too wide for caret, need standard operators
			noshort()
		}
	case !rc.maxIsInf() && rc.max.Major() != 0 && rc.max.Patch() == 0: // basic tilde
		if rc.min.Minor() == rc.max.Minor()-1 && rc.min.Major() == rc.max.Major() {
			pieces = append(pieces, fmt.Sprintf("~%s", rc.min))
		} else {
			// range is too wide for tilde, need standard operators
			noshort()
		}
	case !rc.maxIsInf() && rc.max.Major() == 0 && rc.max.Patch() == 0 && rc.max.Minor() != 0:
		// below 1.0.0, tilde is meaningless but caret is shifted to the
		// right (so it basically behaves the same as tilde does above 1.0.0)
		if rc.min.Minor() == rc.max.Minor()-1 {
			pieces = append(pieces, fmt.Sprintf(caretstr, rc.min))
		} else {
			noshort()
		}
	default:
		noshort()
	}

	for _, e := range rc.excl {
		pieces = append(pieces, fmt.Sprintf("!=%s", e))
	}

	return strings.Join(pieces, ", ")
}

// areAdjacent tests two constraints to determine if they are adjacent,
// but non-overlapping.
//
// If either constraint is not a range, returns false. We still allow it at the
// type level, however, to make the check convenient elsewhere.
//
// Assumes the first range is less than the second; it is incumbent on the
// caller to arrange the inputs appropriately.
func areAdjacent(c1, c2 Constraint) bool {
	var rc1, rc2 rangeConstraint
	var ok bool
	if rc1, ok = c1.(rangeConstraint); !ok {
		return false
	}
	if rc2, ok = c2.(rangeConstraint); !ok {
		return false
	}

	if !rc1.max.Equal(rc2.min) {
		return false
	}

	return (rc1.includeMax && !rc2.includeMin) ||
		(!rc1.includeMax && rc2.includeMin)
}

func (rc rangeConstraint) MatchesAny(c Constraint) bool {
	if _, ok := rc.Intersect(c).(none); ok {
		return false
	}
	return true
}

func dedupeExcls(ex1, ex2 []Version) []Version {
	// TODO stupid inefficient, but these are really only ever going to be
	// small, so not worth optimizing right now
	var ret []Version
oloop:
	for _, e1 := range ex1 {
		for _, e2 := range ex2 {
			if e1.Equal(e2) {
				continue oloop
			}
		}
		ret = append(ret, e1)
	}

	return append(ret, ex2...)
}

func (rangeConstraint) _private() {}
func (rangeConstraint) _real()    {}
