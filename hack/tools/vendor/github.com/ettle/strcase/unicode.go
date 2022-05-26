package strcase

import "unicode"

// Unicode functions, optimized for the common case of ascii
// No performance lost by wrapping since these functions get inlined by the compiler

func isUpper(r rune) bool {
	return unicode.IsUpper(r)
}

func isLower(r rune) bool {
	return unicode.IsLower(r)
}

func isNumber(r rune) bool {
	if r >= '0' && r <= '9' {
		return true
	}
	return unicode.IsNumber(r)
}

func isSpace(r rune) bool {
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return true
	} else if r < 128 {
		return false
	}
	return unicode.IsSpace(r)
}

func toUpper(r rune) rune {
	if r >= 'a' && r <= 'z' {
		return r - 32
	} else if r < 128 {
		return r
	}
	return unicode.ToUpper(r)
}

func toLower(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + 32
	} else if r < 128 {
		return r
	}
	return unicode.ToLower(r)
}
