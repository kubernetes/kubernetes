package match

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type Prefix struct {
	Prefix string
}

func NewPrefix(p string) Prefix {
	return Prefix{p}
}

func (self Prefix) Index(s string) (int, []int) {
	idx := strings.Index(s, self.Prefix)
	if idx == -1 {
		return -1, nil
	}

	length := len(self.Prefix)
	var sub string
	if len(s) > idx+length {
		sub = s[idx+length:]
	} else {
		sub = ""
	}

	segments := acquireSegments(len(sub) + 1)
	segments = append(segments, length)
	for i, r := range sub {
		segments = append(segments, length+i+utf8.RuneLen(r))
	}

	return idx, segments
}

func (self Prefix) Len() int {
	return lenNo
}

func (self Prefix) Match(s string) bool {
	return strings.HasPrefix(s, self.Prefix)
}

func (self Prefix) String() string {
	return fmt.Sprintf("<prefix:%s>", self.Prefix)
}
