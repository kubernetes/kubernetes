package mesos

import (
	"sort"
)

// Ranges represents a list of Ranges.
type Ranges []Value_Range

// NewRanges returns squashed Ranges from the given numbers.
func NewRanges(ns ...uint64) Ranges {
	xs := append(uint64s{}, ns...)
	sort.Sort(xs)
	rs := make(Ranges, len(xs))
	for i := range xs {
		rs[i].Begin, rs[i].End = xs[i], xs[i]
	}
	return rs.Squash()
}

// NewPortRanges returns Ranges from the "ports" resource in the
// given *Offer. If that resource isn't provided, nil will be returned.
//
// The returned Ranges are sorted and have all overlapping ranges merged from
// left to right. e.g. [[0, 5], [4, 3], [10, 7]] -> [[0, 5], [7, 10]]
func NewPortRanges(o *Offer) Ranges {
	if o == nil {
		return Ranges{}
	}

	var (
		r     Resource
		found bool
	)
	for i := range o.Resources {
		if o.Resources[i].GetName() == "ports" {
			r = o.Resources[i]
			found = true
			break
		}
	}

	if !found {
		return Ranges{}
	}

	offered := r.GetRanges().GetRange()
	rs := make(Ranges, len(offered))
	for i, r := range offered {
		if lo, hi := r.GetBegin(), r.GetEnd(); lo <= hi {
			rs[i].Begin, rs[i].End = lo, hi
		} else {
			rs[i].Begin, rs[i].End = hi, lo
		}
	}
	return rs.Sort().Squash()
}

// These three methods implement sort.Interface
func (rs Ranges) Len() int      { return len(rs) }
func (rs Ranges) Swap(i, j int) { rs[i], rs[j] = rs[j], rs[i] }
func (rs Ranges) Less(i, j int) bool {
	return rs[i].Begin < rs[j].Begin || (rs[i].Begin == rs[j].Begin && rs[i].End < rs[j].End)
}

// Size returns the sum of the Size of all Ranges.
func (rs Ranges) Size() uint64 {
	var sz uint64
	for i := range rs {
		sz += 1 + (rs[i].End - rs[i].Begin)
	}
	return sz
}

// Sort sorts the receiving Ranges and returns the result; convenience
func (rs Ranges) Sort() Ranges {
	sort.Sort(rs)
	return rs
}

// Squash merges overlapping and continuous Ranges. It assumes they're pre-sorted.
func (rs Ranges) Squash() Ranges {
	if len(rs) < 2 {
		return rs
	}
	squashed := Ranges{rs[0]}
	for i := 1; i < len(rs); i++ {
		switch max := squashed[len(squashed)-1].End; {
		case 1+max < rs[i].Begin: // no overlap nor continuity: push
			squashed = append(squashed, rs[i])
		case max <= rs[i].End: // overlap or continuity: squash
			squashed[len(squashed)-1].End = rs[i].End
		}
	}
	return squashed
}

// Search performs a binary search for n returning the index of the Range it was
// found at or -1 if not found.
func (rs Ranges) Search(n uint64) int {
	for lo, hi := 0, len(rs)-1; lo <= hi; {
		switch m := lo + (hi-lo)/2; {
		case n < rs[m].Begin:
			hi = m - 1
		case n > rs[m].End:
			lo = m + 1
		default:
			return m
		}
	}
	return -1
}

// Partition partitions Ranges around n. It returns the partitioned Ranges
// and a boolean indicating if n was found.
func (rs Ranges) Partition(n uint64) (Ranges, bool) {
	i := rs.Search(n)
	if i < 0 {
		return rs, false
	}

	pn := make(Ranges, 0, len(rs)+1)
	switch pn = append(pn, rs[:i]...); {
	case rs[i].Begin == rs[i].End: // delete
	case rs[i].Begin == n: // increment lower bound
		pn = append(pn, Value_Range{rs[i].Begin + 1, rs[i].End})
	case rs[i].End == n: // decrement upper bound
		pn = append(pn, Value_Range{rs[i].Begin, rs[i].End - 1})
	default: // split
		pn = append(pn, Value_Range{rs[i].Begin, n - 1}, Value_Range{n + 1, rs[i].End})
	}
	return append(pn, rs[i+1:]...), true
}

// Remove removes a range from already coalesced ranges.
// The algorithms constructs a new vector of ranges which is then
// Squash'ed into a Ranges instance.
func (rs Ranges) Remove(removal Value_Range) Ranges {
	ranges := make([]Value_Range, 0, len(rs))
	for _, r := range rs {
		// skip if the entire range is subsumed by removal
		if r.Begin >= removal.Begin && r.End <= removal.End {
			continue
		}
		// divide if the range subsumes the removal
		if r.Begin < removal.Begin && r.End > removal.End {
			ranges = append(ranges,
				Value_Range{r.Begin, removal.Begin - 1},
				Value_Range{removal.End + 1, r.End},
			)
			continue
		}
		// add the full range if there's no intersection
		if r.End < removal.Begin || r.Begin > removal.End {
			ranges = append(ranges, r)
			continue
		}
		// trim if the range does intersect
		if r.End > removal.End {
			ranges = append(ranges, Value_Range{removal.End + 1, r.End})
		} else {
			if r.Begin >= removal.Begin {
				// should never happen
				panic("r.Begin >= removal.Begin")
			}
			ranges = append(ranges, Value_Range{r.Begin, removal.Begin - 1})
		}
	}
	return Ranges(ranges).Squash()
}

// Compare assumes that both Ranges are already in sort-order.
// Returns 0 if rs and right are equivalent, -1 if rs is a subset of right, or else 1
func (rs Ranges) Compare(right Ranges) int {
	x, y, result := rs.equiv(right)
	if result {
		return 0
	}
	for _, a := range x {
		// make sure that this range is a subset of a range in y
		matched := false
		for _, b := range y {
			if a.Begin >= b.Begin && a.End <= b.End {
				matched = true
				break
			}
		}
		if !matched {
			return 1
		}
	}
	return -1
}

// Equivalent assumes that both Ranges are already in sort-order.
func (rs Ranges) Equivalent(right Ranges) (result bool) {
	_, _, result = rs.equiv(right)
	return
}

// Equivalent assumes that both Ranges are already in sort-order.
func (rs Ranges) equiv(right Ranges) (_, _ Ranges, _ bool) {
	// we need to squash rs and right but don't want to change the originals
	switch len(rs) {
	case 0:
	case 1:
		rs = Ranges{rs[0]}
	default:
		rs = Ranges(append([]Value_Range{rs[0], rs[1]}, rs[2:]...)).Sort().Squash()
	}
	switch len(right) {
	case 0:
	case 1:
		right = Ranges{right[0]}
	default:
		right = Ranges(append([]Value_Range{right[0], right[1]}, right[2:]...)).Sort().Squash()
	}
	return rs, right, (&Value_Ranges{Range: rs}).Equal(&Value_Ranges{Range: right})
}

func (rs Ranges) Clone() Ranges {
	if len(rs) == 0 {
		return nil
	}
	x := make(Ranges, len(rs))
	copy(x, rs)
	return x
}

// Min returns the minimum number in Ranges. It will panic on empty Ranges.
func (rs Ranges) Min() uint64 { return rs[0].Begin }

// Max returns the maximum number in Ranges. It will panic on empty Ranges.
func (rs Ranges) Max() uint64 { return rs[len(rs)-1].End }

// resource returns a *Resource with the given name and Ranges.
func (rs Ranges) resource(name string) Resource {
	vr := make([]Value_Range, len(rs))
	copy(vr, rs)
	return Resource{
		Name:   name,
		Type:   RANGES.Enum(),
		Ranges: &Value_Ranges{Range: vr},
	}
}

// uint64s is an utility used to sort a slice of uint64s
type uint64s []uint64

// These three methods implement sort.Interface
func (ns uint64s) Len() int           { return len(ns) }
func (ns uint64s) Less(i, j int) bool { return ns[i] < ns[j] }
func (ns uint64s) Swap(i, j int)      { ns[i], ns[j] = ns[j], ns[i] }
