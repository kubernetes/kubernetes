package match

import (
	"fmt"
	"github.com/gobwas/glob/util/runes"
	"unicode/utf8"
)

// single represents ?
type Single struct {
	Separators []rune
}

func NewSingle(s []rune) Single {
	return Single{s}
}

func (self Single) Match(s string) bool {
	r, w := utf8.DecodeRuneInString(s)
	if len(s) > w {
		return false
	}

	return runes.IndexRune(self.Separators, r) == -1
}

func (self Single) Len() int {
	return lenOne
}

func (self Single) Index(s string) (int, []int) {
	for i, r := range s {
		if runes.IndexRune(self.Separators, r) == -1 {
			return i, segmentsByRuneLength[utf8.RuneLen(r)]
		}
	}

	return -1, nil
}

func (self Single) String() string {
	return fmt.Sprintf("<single:![%s]>", string(self.Separators))
}
