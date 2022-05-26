package match

import (
	"fmt"
	"github.com/gobwas/glob/util/strings"
)

type Any struct {
	Separators []rune
}

func NewAny(s []rune) Any {
	return Any{s}
}

func (self Any) Match(s string) bool {
	return strings.IndexAnyRunes(s, self.Separators) == -1
}

func (self Any) Index(s string) (int, []int) {
	found := strings.IndexAnyRunes(s, self.Separators)
	switch found {
	case -1:
	case 0:
		return 0, segments0
	default:
		s = s[:found]
	}

	segments := acquireSegments(len(s))
	for i := range s {
		segments = append(segments, i)
	}
	segments = append(segments, len(s))

	return 0, segments
}

func (self Any) Len() int {
	return lenNo
}

func (self Any) String() string {
	return fmt.Sprintf("<any:![%s]>", string(self.Separators))
}
