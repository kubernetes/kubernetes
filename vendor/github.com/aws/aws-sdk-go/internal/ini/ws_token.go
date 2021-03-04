package ini

import (
	"unicode"
)

// isWhitespace will return whether or not the character is
// a whitespace character.
//
// Whitespace is defined as a space or tab.
func isWhitespace(c rune) bool {
	return unicode.IsSpace(c) && c != '\n' && c != '\r'
}

func newWSToken(b []rune) (Token, int, error) {
	i := 0
	for ; i < len(b); i++ {
		if !isWhitespace(b[i]) {
			break
		}
	}

	return newToken(TokenWS, b[:i], NoneType), i, nil
}
