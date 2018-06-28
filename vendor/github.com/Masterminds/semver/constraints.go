package semver

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
)

var constraintRegex *regexp.Regexp
var constraintRangeRegex *regexp.Regexp

const cvRegex string = `v?([0-9|x|X|\*]+)(\.[0-9|x|X|\*]+)?(\.[0-9|x|X|\*]+)?` +
	`(-([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?` +
	`(\+([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?`

func init() {
	constraintOps := []string{
		"",
		"=",
		"!=",
		">",
		"<",
		">=",
		"=>",
		"<=",
		"=<",
		"~",
		"~>",
		"^",
	}

	ops := make([]string, 0, len(constraintOps))
	for _, op := range constraintOps {
		ops = append(ops, regexp.QuoteMeta(op))
	}

	constraintRegex = regexp.MustCompile(fmt.Sprintf(
		`^\s*(%s)\s*(%s)\s*$`,
		strings.Join(ops, "|"),
		cvRegex))

	constraintRangeRegex = regexp.MustCompile(fmt.Sprintf(
		`\s*(%s)\s*-\s*(%s)\s*`,
		cvRegex, cvRegex))
}

// Constraint is the interface that wraps checking a semantic version against
// one or more constraints to find a match.
type Constraint interface {
	// Constraints compose the fmt.Stringer interface. This method is the
	// bijective inverse of NewConstraint(): if a string yielded from this
	// method is passed to NewConstraint(), a byte-identical instance of the
	// original Constraint will be returend.
	fmt.Stringer

	// ImpliedCaretString converts the Constraint to a string in the same manner
	// as String(), but treats the empty operator as equivalent to ^, rather
	// than =.
	//
	// In the same way that String() is the inverse of NewConstraint(), this
	// method is the inverse of to NewConstraintIC().
	ImpliedCaretString() string

	// Matches checks that a version satisfies the constraint. If it does not,
	// an error is returned indcating the problem; if it does, the error is nil.
	Matches(v Version) error

	// Intersect computes the intersection between the receiving Constraint and
	// passed Constraint, and returns a new Constraint representing the result.
	Intersect(Constraint) Constraint

	// Union computes the union between the receiving Constraint and the passed
	// Constraint, and returns a new Constraint representing the result.
	Union(Constraint) Constraint

	// MatchesAny returns a bool indicating whether there exists any version that
	// satisfies both the receiver constraint, and the passed Constraint.
	//
	// In other words, this reports whether an intersection would be non-empty.
	MatchesAny(Constraint) bool

	// Restrict implementation of this interface to this package. We need the
	// flexibility of an interface, but we cover all possibilities here; closing
	// off the interface to external implementation lets us safely do tricks
	// with types for magic types (none and any)
	_private()
}

// realConstraint is used internally to differentiate between any, none, and
// unionConstraints, vs. Version and rangeConstraints.
type realConstraint interface {
	Constraint
	_real()
}

// CacheConstraints controls whether or not parsed constraints are cached
var CacheConstraints = true
var constraintCache = make(map[string]ccache)
var constraintCacheIC = make(map[string]ccache)
var constraintCacheLock sync.RWMutex

type ccache struct {
	c   Constraint
	err error
}

// NewConstraint takes a string representing a set of semver constraints, and
// returns a corresponding Constraint object. Constraints are suitable
// for checking Versions for admissibility, or combining with other Constraint
// objects.
//
// If an invalid constraint string is passed, more information is provided in
// the returned error string.
func NewConstraint(in string) (Constraint, error) {
	return newConstraint(in, false, constraintCache)
}

// NewConstraintIC ("Implied Caret") is the same as NewConstraint, except that
// it treats an absent operator as being equivalent to ^ instead of =.
func NewConstraintIC(in string) (Constraint, error) {
	return newConstraint(in, true, constraintCacheIC)
}

func newConstraint(in string, ic bool, cache map[string]ccache) (Constraint, error) {
	if CacheConstraints {
		constraintCacheLock.RLock()
		if final, exists := cache[in]; exists {
			constraintCacheLock.RUnlock()
			return final.c, final.err
		}
		constraintCacheLock.RUnlock()
	}

	// Rewrite - ranges into a comparison operation.
	c := rewriteRange(in)

	ors := strings.Split(c, "||")
	or := make([]Constraint, len(ors))
	for k, v := range ors {
		cs := strings.Split(v, ",")
		result := make([]Constraint, len(cs))
		for i, s := range cs {
			pc, err := parseConstraint(s, ic)
			if err != nil {
				if CacheConstraints {
					constraintCacheLock.Lock()
					cache[in] = ccache{err: err}
					constraintCacheLock.Unlock()
				}
				return nil, err
			}

			result[i] = pc
		}
		or[k] = Intersection(result...)
	}

	final := Union(or...)

	if CacheConstraints {
		constraintCacheLock.Lock()
		cache[in] = ccache{c: final}
		constraintCacheLock.Unlock()
	}

	return final, nil
}

// Intersection computes the intersection between N Constraints, returning as
// compact a representation of the intersection as possible.
//
// No error is indicated if all the sets are collectively disjoint; you must inspect the
// return value to see if the result is the empty set (by calling IsNone() on
// it).
func Intersection(cg ...Constraint) Constraint {
	// If there's zero or one constraints in the group, we can quit fast
	switch len(cg) {
	case 0:
		// Zero members, only sane thing to do is return none
		return None()
	case 1:
		// Just one member means that's our final constraint
		return cg[0]
	}

	car, cdr := cg[0], cg[1:]
	for _, c := range cdr {
		if IsNone(car) {
			return None()
		}
		car = car.Intersect(c)
	}

	return car
}

// Union takes a variable number of constraints, and returns the most compact
// possible representation of those constraints.
//
// This effectively ORs together all the provided constraints. If any of the
// included constraints are the set of all versions (any), that supercedes
// everything else.
func Union(cg ...Constraint) Constraint {
	// If there's zero or one constraints in the group, we can quit fast
	switch len(cg) {
	case 0:
		// Zero members, only sane thing to do is return none
		return None()
	case 1:
		// One member, so the result will just be that
		return cg[0]
	}

	// Preliminary pass to look for 'any' in the current set (and bail out early
	// if found), but also construct a []realConstraint for everything else
	var real constraintList

	for _, c := range cg {
		switch tc := c.(type) {
		case any:
			return c
		case none:
			continue
		case Version:
			//heap.Push(&real, tc)
			real = append(real, tc)
		case rangeConstraint:
			//heap.Push(&real, tc)
			real = append(real, tc)
		case unionConstraint:
			real = append(real, tc...)
			//for _, c2 := range tc {
			//heap.Push(&real, c2)
			//}
		default:
			panic("unknown constraint type")
		}
	}
	// TODO wtf why isn't heap working...so, ugh, have to do this

	// Sort both the versions and ranges into ascending order
	sort.Sort(real)

	// Iteratively merge the constraintList elements
	var nuc unionConstraint
	for _, c := range real {
		if len(nuc) == 0 {
			nuc = append(nuc, c)
			continue
		}

		last := nuc[len(nuc)-1]
		switch lt := last.(type) {
		case Version:
			switch ct := c.(type) {
			case Version:
				// Two versions in a row; only append if they're not equal
				if !lt.Equal(ct) {
					nuc = append(nuc, ct)
				}
			case rangeConstraint:
				// Last was version, current is range. constraintList sorts by
				// min version, so it's guaranteed that the version will be less
				// than the range's min, guaranteeing that these are disjoint.
				//
				// ...almost. If the min of the range is the same as the
				// version, then a union should merge the two by making the
				// range inclusive at the bottom.
				if lt.Equal(ct.min) {
					ct.includeMin = true
					nuc[len(nuc)-1] = ct
				} else {
					nuc = append(nuc, c)
				}
			}
		case rangeConstraint:
			switch ct := c.(type) {
			case Version:
				// Last was range, current is version. constraintList sort invariants guarantee
				// that the version will be greater than the min, so we have to
				// determine if the version is less than the max. If it is, we
				// subsume it into the range with a Union call.
				//
				// Lazy version: just union them and let rangeConstraint figure
				// it out, then switch on the result type.
				c2 := lt.Union(ct)
				if crc, ok := c2.(realConstraint); ok {
					nuc[len(nuc)-1] = crc
				} else {
					// Otherwise, all it can be is a union constraint. First
					// item in the union will be the same range, second will be the
					// version, so append onto nuc from one back from the end
					nuc = append(nuc[:len(nuc)-1], c2.(unionConstraint)...)
				}
			case rangeConstraint:
				if lt.MatchesAny(ct) || areAdjacent(lt, ct) {
					// If the previous range overlaps or is adjacent to the
					// current range, we know they'll be able to merge together,
					// so overwrite the last item in nuc with the result of that
					// merge (which is what Union will produce)
					nuc[len(nuc)-1] = lt.Union(ct).(realConstraint)
				} else {
					nuc = append(nuc, c)
				}
			}
		}
	}

	if len(nuc) == 1 {
		return nuc[0]
	}
	return nuc
}
