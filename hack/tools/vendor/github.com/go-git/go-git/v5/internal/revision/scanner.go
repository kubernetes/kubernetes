package revision

import (
	"bufio"
	"io"
	"unicode"
)

// runeCategoryValidator takes a rune as input and
// validates it belongs to a rune category
type runeCategoryValidator func(r rune) bool

// tokenizeExpression aggregates a series of runes matching check predicate into a single
// string and provides given tokenType as token type
func tokenizeExpression(ch rune, tokenType token, check runeCategoryValidator, r *bufio.Reader) (token, string, error) {
	var data []rune
	data = append(data, ch)

	for {
		c, _, err := r.ReadRune()

		if c == zeroRune {
			break
		}

		if err != nil {
			return tokenError, "", err
		}

		if check(c) {
			data = append(data, c)
		} else {
			err := r.UnreadRune()

			if err != nil {
				return tokenError, "", err
			}

			return tokenType, string(data), nil
		}
	}

	return tokenType, string(data), nil
}

var zeroRune = rune(0)

// scanner represents a lexical scanner.
type scanner struct {
	r *bufio.Reader
}

// newScanner returns a new instance of scanner.
func newScanner(r io.Reader) *scanner {
	return &scanner{r: bufio.NewReader(r)}
}

// Scan extracts tokens and their strings counterpart
// from the reader
func (s *scanner) scan() (token, string, error) {
	ch, _, err := s.r.ReadRune()

	if err != nil && err != io.EOF {
		return tokenError, "", err
	}

	switch ch {
	case zeroRune:
		return eof, "", nil
	case ':':
		return colon, string(ch), nil
	case '~':
		return tilde, string(ch), nil
	case '^':
		return caret, string(ch), nil
	case '.':
		return dot, string(ch), nil
	case '/':
		return slash, string(ch), nil
	case '{':
		return obrace, string(ch), nil
	case '}':
		return cbrace, string(ch), nil
	case '-':
		return minus, string(ch), nil
	case '@':
		return at, string(ch), nil
	case '\\':
		return aslash, string(ch), nil
	case '?':
		return qmark, string(ch), nil
	case '*':
		return asterisk, string(ch), nil
	case '[':
		return obracket, string(ch), nil
	case '!':
		return emark, string(ch), nil
	}

	if unicode.IsSpace(ch) {
		return space, string(ch), nil
	}

	if unicode.IsControl(ch) {
		return control, string(ch), nil
	}

	if unicode.IsLetter(ch) {
		return tokenizeExpression(ch, word, unicode.IsLetter, s.r)
	}

	if unicode.IsNumber(ch) {
		return tokenizeExpression(ch, number, unicode.IsNumber, s.r)
	}

	return tokenError, string(ch), nil
}
