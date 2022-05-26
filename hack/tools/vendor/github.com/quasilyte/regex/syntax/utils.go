package syntax

func isSpace(ch byte) bool {
	switch ch {
	case '\r', '\n', '\t', '\f', '\v':
		return true
	default:
		return false
	}
}

func isAlphanumeric(ch byte) bool {
	return (ch >= 'a' && ch <= 'z') ||
		(ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9')
}

func isDigit(ch byte) bool {
	return ch >= '0' && ch <= '9'
}

func isOctalDigit(ch byte) bool {
	return ch >= '0' && ch <= '7'
}

func isHexDigit(ch byte) bool {
	return (ch >= '0' && ch <= '9') ||
		(ch >= 'a' && ch <= 'f') ||
		(ch >= 'A' && ch <= 'F')
}
