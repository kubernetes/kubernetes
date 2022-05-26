package asciicheck

import (
	"unicode"
	"unicode/utf8"
)

func isASCII(s string) (rune, bool) {
	if len(s) == 1 {
		r, size := utf8.DecodeRuneInString(s)
		return r, size < 2
	}

	for _, r := range s {
		if r > unicode.MaxASCII {
			return r, false
		}
	}

	return 0, true
}
