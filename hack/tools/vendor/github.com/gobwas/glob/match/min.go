package match

import (
	"fmt"
	"unicode/utf8"
)

type Min struct {
	Limit int
}

func NewMin(l int) Min {
	return Min{l}
}

func (self Min) Match(s string) bool {
	var l int
	for range s {
		l += 1
		if l >= self.Limit {
			return true
		}
	}

	return false
}

func (self Min) Index(s string) (int, []int) {
	var count int

	c := len(s) - self.Limit + 1
	if c <= 0 {
		return -1, nil
	}

	segments := acquireSegments(c)
	for i, r := range s {
		count++
		if count >= self.Limit {
			segments = append(segments, i+utf8.RuneLen(r))
		}
	}

	if len(segments) == 0 {
		return -1, nil
	}

	return 0, segments
}

func (self Min) Len() int {
	return lenNo
}

func (self Min) String() string {
	return fmt.Sprintf("<min:%d>", self.Limit)
}
