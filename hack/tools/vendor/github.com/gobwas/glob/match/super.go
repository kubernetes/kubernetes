package match

import (
	"fmt"
)

type Super struct{}

func NewSuper() Super {
	return Super{}
}

func (self Super) Match(s string) bool {
	return true
}

func (self Super) Len() int {
	return lenNo
}

func (self Super) Index(s string) (int, []int) {
	segments := acquireSegments(len(s) + 1)
	for i := range s {
		segments = append(segments, i)
	}
	segments = append(segments, len(s))

	return 0, segments
}

func (self Super) String() string {
	return fmt.Sprintf("<super>")
}
