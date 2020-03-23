package ini

// isComment will return whether or not the next byte(s) is a
// comment.
func isComment(b []rune) bool {
	if len(b) == 0 {
		return false
	}

	switch b[0] {
	case ';':
		return true
	case '#':
		return true
	}

	return false
}

// newCommentToken will create a comment token and
// return how many bytes were read.
func newCommentToken(b []rune) (Token, int, error) {
	i := 0
	for ; i < len(b); i++ {
		if b[i] == '\n' {
			break
		}

		if len(b)-i > 2 && b[i] == '\r' && b[i+1] == '\n' {
			break
		}
	}

	return newToken(TokenComment, b[:i], NoneType), i, nil
}
