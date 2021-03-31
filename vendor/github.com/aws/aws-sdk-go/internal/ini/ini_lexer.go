package ini

import (
	"bytes"
	"io"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

const (
	// ErrCodeUnableToReadFile is used when a file is failed to be
	// opened or read from.
	ErrCodeUnableToReadFile = "FailedRead"
)

// TokenType represents the various different tokens types
type TokenType int

func (t TokenType) String() string {
	switch t {
	case TokenNone:
		return "none"
	case TokenLit:
		return "literal"
	case TokenSep:
		return "sep"
	case TokenOp:
		return "op"
	case TokenWS:
		return "ws"
	case TokenNL:
		return "newline"
	case TokenComment:
		return "comment"
	case TokenComma:
		return "comma"
	default:
		return ""
	}
}

// TokenType enums
const (
	TokenNone = TokenType(iota)
	TokenLit
	TokenSep
	TokenComma
	TokenOp
	TokenWS
	TokenNL
	TokenComment
)

type iniLexer struct{}

// Tokenize will return a list of tokens during lexical analysis of the
// io.Reader.
func (l *iniLexer) Tokenize(r io.Reader) ([]Token, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, awserr.New(ErrCodeUnableToReadFile, "unable to read file", err)
	}

	return l.tokenize(b)
}

func (l *iniLexer) tokenize(b []byte) ([]Token, error) {
	runes := bytes.Runes(b)
	var err error
	n := 0
	tokenAmount := countTokens(runes)
	tokens := make([]Token, tokenAmount)
	count := 0

	for len(runes) > 0 && count < tokenAmount {
		switch {
		case isWhitespace(runes[0]):
			tokens[count], n, err = newWSToken(runes)
		case isComma(runes[0]):
			tokens[count], n = newCommaToken(), 1
		case isComment(runes):
			tokens[count], n, err = newCommentToken(runes)
		case isNewline(runes):
			tokens[count], n, err = newNewlineToken(runes)
		case isSep(runes):
			tokens[count], n, err = newSepToken(runes)
		case isOp(runes):
			tokens[count], n, err = newOpToken(runes)
		default:
			tokens[count], n, err = newLitToken(runes)
		}

		if err != nil {
			return nil, err
		}

		count++

		runes = runes[n:]
	}

	return tokens[:count], nil
}

func countTokens(runes []rune) int {
	count, n := 0, 0
	var err error

	for len(runes) > 0 {
		switch {
		case isWhitespace(runes[0]):
			_, n, err = newWSToken(runes)
		case isComma(runes[0]):
			_, n = newCommaToken(), 1
		case isComment(runes):
			_, n, err = newCommentToken(runes)
		case isNewline(runes):
			_, n, err = newNewlineToken(runes)
		case isSep(runes):
			_, n, err = newSepToken(runes)
		case isOp(runes):
			_, n, err = newOpToken(runes)
		default:
			_, n, err = newLitToken(runes)
		}

		if err != nil {
			return 0
		}

		count++
		runes = runes[n:]
	}

	return count + 1
}

// Token indicates a metadata about a given value.
type Token struct {
	t         TokenType
	ValueType ValueType
	base      int
	raw       []rune
}

var emptyValue = Value{}

func newToken(t TokenType, raw []rune, v ValueType) Token {
	return Token{
		t:         t,
		raw:       raw,
		ValueType: v,
	}
}

// Raw return the raw runes that were consumed
func (tok Token) Raw() []rune {
	return tok.raw
}

// Type returns the token type
func (tok Token) Type() TokenType {
	return tok.t
}
