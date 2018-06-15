package semver

import "strings"

type unionConstraint []realConstraint

func (uc unionConstraint) Matches(v Version) error {
	var uce MultiMatchFailure
	for _, c := range uc {
		err := c.Matches(v)
		if err == nil {
			return nil
		}
		uce = append(uce, err.(MatchFailure))

	}

	return uce
}

func (uc unionConstraint) Intersect(c2 Constraint) Constraint {
	var other []realConstraint

	switch tc2 := c2.(type) {
	case none:
		return None()
	case any:
		return uc
	case Version:
		return c2
	case rangeConstraint:
		other = append(other, tc2)
	case unionConstraint:
		other = c2.(unionConstraint)
	default:
		panic("unknown type")
	}

	var newc []Constraint
	// TODO there's a smarter way to do this than NxN, but...worth it?
	for _, c := range uc {
		for _, oc := range other {
			i := c.Intersect(oc)
			if !IsNone(i) {
				newc = append(newc, i)
			}
		}
	}

	return Union(newc...)
}

func (uc unionConstraint) MatchesAny(c Constraint) bool {
	for _, ic := range uc {
		if ic.MatchesAny(c) {
			return true
		}
	}
	return false
}

func (uc unionConstraint) Union(c Constraint) Constraint {
	return Union(uc, c)
}

func (uc unionConstraint) String() string {
	var pieces []string
	for _, c := range uc {
		pieces = append(pieces, c.String())
	}

	return strings.Join(pieces, " || ")
}

func (uc unionConstraint) ImpliedCaretString() string {
	var pieces []string
	for _, c := range uc {
		pieces = append(pieces, c.ImpliedCaretString())
	}

	return strings.Join(pieces, " || ")
}

func (unionConstraint) _private() {}

type constraintList []realConstraint

func (cl constraintList) Len() int {
	return len(cl)
}

func (cl constraintList) Swap(i, j int) {
	cl[i], cl[j] = cl[j], cl[i]
}

func (cl constraintList) Less(i, j int) bool {
	ic, jc := cl[i], cl[j]

	switch tic := ic.(type) {
	case Version:
		switch tjc := jc.(type) {
		case Version:
			return tic.LessThan(tjc)
		case rangeConstraint:
			if tjc.minIsZero() {
				return false
			}

			// Because we don't assume stable sort, always put versions ahead of
			// range mins if they're equal and includeMin is on
			if tjc.includeMin && tic.Equal(tjc.min) {
				return false
			}
			return tic.LessThan(tjc.min)
		}
	case rangeConstraint:
		switch tjc := jc.(type) {
		case Version:
			if tic.minIsZero() {
				return true
			}

			// Because we don't assume stable sort, always put versions ahead of
			// range mins if they're equal and includeMin is on
			if tic.includeMin && tjc.Equal(tic.min) {
				return false
			}
			return tic.min.LessThan(tjc)
		case rangeConstraint:
			if tic.minIsZero() {
				return true
			}
			if tjc.minIsZero() {
				return false
			}
			return tic.min.LessThan(tjc.min)
		}
	}

	panic("unreachable")
}

func (cl *constraintList) Push(x interface{}) {
	*cl = append(*cl, x.(realConstraint))
}

func (cl *constraintList) Pop() interface{} {
	o := *cl
	c := o[len(o)-1]
	*cl = o[:len(o)-1]
	return c
}
