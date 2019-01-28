package ini

import (
	"fmt"
)

var (
	emptyRunes = []rune{}
)

func isSep(b []rune) bool {
	if len(b) == 0 {
		return false
	}

	switch b[0] {
	case '[', ']':
		return true
	default:
		return false
	}
}

var (
	openBrace  = []rune("[")
	closeBrace = []rune("]")
)

func newSepToken(b []rune) (Token, int, error) {
	tok := Token{}

	switch b[0] {
	case '[':
		tok = newToken(TokenSep, openBrace, NoneType)
	case ']':
		tok = newToken(TokenSep, closeBrace, NoneType)
	default:
		return tok, 0, NewParseError(fmt.Sprintf("unexpected sep type, %v", b[0]))
	}
	return tok, 1, nil
}
