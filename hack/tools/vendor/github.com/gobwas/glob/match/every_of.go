package match

import (
	"fmt"
)

type EveryOf struct {
	Matchers Matchers
}

func NewEveryOf(m ...Matcher) EveryOf {
	return EveryOf{Matchers(m)}
}

func (self *EveryOf) Add(m Matcher) error {
	self.Matchers = append(self.Matchers, m)
	return nil
}

func (self EveryOf) Len() (l int) {
	for _, m := range self.Matchers {
		if ml := m.Len(); l > 0 {
			l += ml
		} else {
			return -1
		}
	}

	return
}

func (self EveryOf) Index(s string) (int, []int) {
	var index int
	var offset int

	// make `in` with cap as len(s),
	// cause it is the maximum size of output segments values
	next := acquireSegments(len(s))
	current := acquireSegments(len(s))

	sub := s
	for i, m := range self.Matchers {
		idx, seg := m.Index(sub)
		if idx == -1 {
			releaseSegments(next)
			releaseSegments(current)
			return -1, nil
		}

		if i == 0 {
			// we use copy here instead of `current = seg`
			// cause seg is a slice from reusable buffer `in`
			// and it could be overwritten in next iteration
			current = append(current, seg...)
		} else {
			// clear the next
			next = next[:0]

			delta := index - (idx + offset)
			for _, ex := range current {
				for _, n := range seg {
					if ex+delta == n {
						next = append(next, n)
					}
				}
			}

			if len(next) == 0 {
				releaseSegments(next)
				releaseSegments(current)
				return -1, nil
			}

			current = append(current[:0], next...)
		}

		index = idx + offset
		sub = s[index:]
		offset += idx
	}

	releaseSegments(next)

	return index, current
}

func (self EveryOf) Match(s string) bool {
	for _, m := range self.Matchers {
		if !m.Match(s) {
			return false
		}
	}

	return true
}

func (self EveryOf) String() string {
	return fmt.Sprintf("<every_of:[%s]>", self.Matchers)
}
