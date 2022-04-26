package toml

import "fmt"

// Define tokens
type tokenType int

const (
	eof = -(iota + 1)
)

const (
	tokenError tokenType = iota
	tokenEOF
	tokenComment
	tokenKey
	tokenString
	tokenInteger
	tokenTrue
	tokenFalse
	tokenFloat
	tokenInf
	tokenNan
	tokenEqual
	tokenLeftBracket
	tokenRightBracket
	tokenLeftCurlyBrace
	tokenRightCurlyBrace
	tokenLeftParen
	tokenRightParen
	tokenDoubleLeftBracket
	tokenDoubleRightBracket
	tokenLocalDate
	tokenLocalTime
	tokenTimeOffset
	tokenKeyGroup
	tokenKeyGroupArray
	tokenComma
	tokenColon
	tokenDollar
	tokenStar
	tokenQuestion
	tokenDot
	tokenDotDot
	tokenEOL
)

var tokenTypeNames = []string{
	"Error",
	"EOF",
	"Comment",
	"Key",
	"String",
	"Integer",
	"True",
	"False",
	"Float",
	"Inf",
	"NaN",
	"=",
	"[",
	"]",
	"{",
	"}",
	"(",
	")",
	"]]",
	"[[",
	"LocalDate",
	"LocalTime",
	"TimeOffset",
	"KeyGroup",
	"KeyGroupArray",
	",",
	":",
	"$",
	"*",
	"?",
	".",
	"..",
	"EOL",
}

type token struct {
	Position
	typ tokenType
	val string
}

func (tt tokenType) String() string {
	idx := int(tt)
	if idx < len(tokenTypeNames) {
		return tokenTypeNames[idx]
	}
	return "Unknown"
}

func (t token) String() string {
	switch t.typ {
	case tokenEOF:
		return "EOF"
	case tokenError:
		return t.val
	}

	return fmt.Sprintf("%q", t.val)
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t'
}

func isAlphanumeric(r rune) bool {
	return 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || r == '_'
}

func isKeyChar(r rune) bool {
	// Keys start with the first character that isn't whitespace or [ and end
	// with the last non-whitespace character before the equals sign. Keys
	// cannot contain a # character."
	return !(r == '\r' || r == '\n' || r == eof || r == '=')
}

func isKeyStartChar(r rune) bool {
	return !(isSpace(r) || r == '\r' || r == '\n' || r == eof || r == '[')
}

func isDigit(r rune) bool {
	return '0' <= r && r <= '9'
}

func isHexDigit(r rune) bool {
	return isDigit(r) ||
		(r >= 'a' && r <= 'f') ||
		(r >= 'A' && r <= 'F')
}
