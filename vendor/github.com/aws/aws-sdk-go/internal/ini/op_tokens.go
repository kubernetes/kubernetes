package ini

import (
	"fmt"
)

var (
	equalOp      = []rune("=")
	equalColonOp = []rune(":")
)

func isOp(b []rune) bool {
	if len(b) == 0 {
		return false
	}

	switch b[0] {
	case '=':
		return true
	case ':':
		return true
	default:
		return false
	}
}

func newOpToken(b []rune) (Token, int, error) {
	tok := Token{}

	switch b[0] {
	case '=':
		tok = newToken(TokenOp, equalOp, NoneType)
	case ':':
		tok = newToken(TokenOp, equalColonOp, NoneType)
	default:
		return tok, 0, NewParseError(fmt.Sprintf("unexpected op type, %v", b[0]))
	}
	return tok, 1, nil
}
