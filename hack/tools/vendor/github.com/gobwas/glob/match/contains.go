package match

import (
	"fmt"
	"strings"
)

type Contains struct {
	Needle string
	Not    bool
}

func NewContains(needle string, not bool) Contains {
	return Contains{needle, not}
}

func (self Contains) Match(s string) bool {
	return strings.Contains(s, self.Needle) != self.Not
}

func (self Contains) Index(s string) (int, []int) {
	var offset int

	idx := strings.Index(s, self.Needle)

	if !self.Not {
		if idx == -1 {
			return -1, nil
		}

		offset = idx + len(self.Needle)
		if len(s) <= offset {
			return 0, []int{offset}
		}
		s = s[offset:]
	} else if idx != -1 {
		s = s[:idx]
	}

	segments := acquireSegments(len(s) + 1)
	for i := range s {
		segments = append(segments, offset+i)
	}

	return 0, append(segments, offset+len(s))
}

func (self Contains) Len() int {
	return lenNo
}

func (self Contains) String() string {
	var not string
	if self.Not {
		not = "!"
	}
	return fmt.Sprintf("<contains:%s[%s]>", not, self.Needle)
}
