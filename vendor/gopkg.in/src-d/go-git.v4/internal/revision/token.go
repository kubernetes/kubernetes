package revision

// token represents a entity extracted from string parsing
type token int

const (
	eof token = iota

	aslash
	asterisk
	at
	caret
	cbrace
	colon
	control
	dot
	emark
	minus
	number
	obrace
	obracket
	qmark
	slash
	space
	tilde
	tokenError
	word
)
