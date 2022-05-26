package ssh_config

import "fmt"

type token struct {
	Position
	typ tokenType
	val string
}

func (t token) String() string {
	switch t.typ {
	case tokenEOF:
		return "EOF"
	}
	return fmt.Sprintf("%q", t.val)
}

type tokenType int

const (
	eof = -(iota + 1)
)

const (
	tokenError tokenType = iota
	tokenEOF
	tokenEmptyLine
	tokenComment
	tokenKey
	tokenEquals
	tokenString
)

func isSpace(r rune) bool {
	return r == ' ' || r == '\t'
}

func isKeyStartChar(r rune) bool {
	return !(isSpace(r) || r == '\r' || r == '\n' || r == eof)
}

// I'm not sure that this is correct
func isKeyChar(r rune) bool {
	// Keys start with the first character that isn't whitespace or [ and end
	// with the last non-whitespace character before the equals sign. Keys
	// cannot contain a # character."
	return !(r == '\r' || r == '\n' || r == eof || r == '=')
}
