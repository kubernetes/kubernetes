package strings

import (
	"strings"
	"unicode/utf8"
)

func IndexAnyRunes(s string, rs []rune) int {
	for _, r := range rs {
		if i := strings.IndexRune(s, r); i != -1 {
			return i
		}
	}

	return -1
}

func LastIndexAnyRunes(s string, rs []rune) int {
	for _, r := range rs {
		i := -1
		if 0 <= r && r < utf8.RuneSelf {
			i = strings.LastIndexByte(s, byte(r))
		} else {
			sub := s
			for len(sub) > 0 {
				j := strings.IndexRune(s, r)
				if j == -1 {
					break
				}
				i = j
				sub = sub[i+1:]
			}
		}
		if i != -1 {
			return i
		}
	}
	return -1
}
