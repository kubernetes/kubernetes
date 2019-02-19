package ini

func isNewline(b []rune) bool {
	if len(b) == 0 {
		return false
	}

	if b[0] == '\n' {
		return true
	}

	if len(b) < 2 {
		return false
	}

	return b[0] == '\r' && b[1] == '\n'
}

func newNewlineToken(b []rune) (Token, int, error) {
	i := 1
	if b[0] == '\r' && isNewline(b[1:]) {
		i++
	}

	if !isNewline([]rune(b[:i])) {
		return emptyToken, 0, NewParseError("invalid new line token")
	}

	return newToken(TokenNL, b[:i], NoneType), i, nil
}
