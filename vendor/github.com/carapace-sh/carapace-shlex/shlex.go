package shlex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// TokenType is a top-level token classification: A word, space, comment, unknown.
type TokenType int

func (t TokenType) MarshalJSON() ([]byte, error) {
	return json.Marshal(tokenTypes[t])
}

// runeTokenClass is the type of a UTF-8 character classification: A quote, space, escape.
type runeTokenClass int

// the internal state used by the lexer state machine
type LexerState int

func (l LexerState) MarshalJSON() ([]byte, error) {
	return json.Marshal(lexerStates[l])
}

// Token is a (type, value) pair representing a lexographical token.
type Token struct {
	Type           TokenType
	Value          string
	RawValue       string
	Index          int
	State          LexerState
	WordbreakType  WordbreakType `json:",omitempty"`
	WordbreakIndex int           // index of last opening quote in Value (only correct when in quoting state)
}

func (t *Token) add(r rune) {
	t.Value += string(r)
}

func (t *Token) removeLastRaw() {
	runes := []rune(t.RawValue)
	t.RawValue = string(runes[:len(runes)-1])
}

func (t Token) adjoins(other Token) bool {
	return t.Index+len(t.RawValue) == other.Index || t.Index == other.Index+len(other.RawValue)
}

// Equal reports whether tokens a, and b, are equal.
// Two tokens are equal if both their types and values are equal. A nil token can
// never be equal to another token.
func (t *Token) Equal(other *Token) bool {
	switch {
	case t == nil,
		other == nil,
		t.Type != other.Type,
		t.Value != other.Value,
		t.RawValue != other.RawValue,
		t.Index != other.Index,
		t.State != other.State,
		t.WordbreakType != other.WordbreakType,
		t.WordbreakIndex != other.WordbreakIndex:
		return false
	default:
		return true
	}
}

// Named classes of UTF-8 runes
const (
	spaceRunes            = " \t\r\n"
	escapingQuoteRunes    = `"`
	nonEscapingQuoteRunes = "'"
	escapeRunes           = `\`
	commentRunes          = "#"
)

// Classes of rune token
const (
	unknownRuneClass runeTokenClass = iota
	spaceRuneClass
	escapingQuoteRuneClass
	nonEscapingQuoteRuneClass
	escapeRuneClass
	commentRuneClass
	wordbreakRuneClass
	eofRuneClass
)

// Classes of lexographic token
const (
	UNKNOWN_TOKEN TokenType = iota
	WORD_TOKEN
	SPACE_TOKEN
	COMMENT_TOKEN
	WORDBREAK_TOKEN
)

var tokenTypes = map[TokenType]string{
	UNKNOWN_TOKEN:   "UNKNOWN_TOKEN",
	WORD_TOKEN:      "WORD_TOKEN",
	SPACE_TOKEN:     "SPACE_TOKEN",
	COMMENT_TOKEN:   "COMMENT_TOKEN",
	WORDBREAK_TOKEN: "WORDBREAK_TOKEN",
}

// Lexer state machine states
const (
	START_STATE            LexerState = iota // no runes have been seen
	IN_WORD_STATE                            // processing regular runes in a word
	ESCAPING_STATE                           // we have just consumed an escape rune; the next rune is literal
	ESCAPING_QUOTED_STATE                    // we have just consumed an escape rune within a quoted string
	QUOTING_ESCAPING_STATE                   // we are within a quoted string that supports escaping ("...")
	QUOTING_STATE                            // we are within a string that does not support escaping ('...')
	COMMENT_STATE                            // we are within a comment (everything following an unquoted or unescaped #
	WORDBREAK_STATE                          // we have just consumed a wordbreak rune
)

var lexerStates = map[LexerState]string{
	START_STATE:            "START_STATE",
	IN_WORD_STATE:          "IN_WORD_STATE",
	ESCAPING_STATE:         "ESCAPING_STATE",
	ESCAPING_QUOTED_STATE:  "ESCAPING_QUOTED_STATE",
	QUOTING_ESCAPING_STATE: "QUOTING_ESCAPING_STATE",
	QUOTING_STATE:          "QUOTING_STATE",
	COMMENT_STATE:          "COMMENT_STATE",
	WORDBREAK_STATE:        "WORDBREAK_STATE",
}

// tokenClassifier is used for classifying rune characters.
type tokenClassifier map[rune]runeTokenClass

func (typeMap tokenClassifier) addRuneClass(runes string, tokenType runeTokenClass) {
	for _, runeChar := range runes {
		typeMap[runeChar] = tokenType
	}
}

// newDefaultClassifier creates a new classifier for ASCII characters.
func newDefaultClassifier() tokenClassifier {
	t := tokenClassifier{}
	t.addRuneClass(spaceRunes, spaceRuneClass)
	t.addRuneClass(escapingQuoteRunes, escapingQuoteRuneClass)
	t.addRuneClass(nonEscapingQuoteRunes, nonEscapingQuoteRuneClass)
	t.addRuneClass(escapeRunes, escapeRuneClass)
	t.addRuneClass(commentRunes, commentRuneClass)

	wordbreakRunes := BASH_WORDBREAKS
	if wordbreaks := os.Getenv("COMP_WORDBREAKS"); wordbreaks != "" {
		wordbreakRunes = wordbreaks
	}
	filtered := make([]rune, 0)
	for _, r := range wordbreakRunes {
		if t.ClassifyRune(r) == unknownRuneClass {
			filtered = append(filtered, r)
		}
	}
	t.addRuneClass(string(filtered), wordbreakRuneClass)

	return t
}

// ClassifyRune classifiees a rune
func (t tokenClassifier) ClassifyRune(runeVal rune) runeTokenClass {
	return t[runeVal]
}

// lexer turns an input stream into a sequence of tokens. Whitespace and comments are skipped.
type lexer tokenizer

// newLexer creates a new lexer from an input stream.
func newLexer(r io.Reader) *lexer {
	return (*lexer)(newTokenizer(r))
}

// Next returns the next token, or an error. If there are no more tokens,
// the error will be io.EOF.
func (l *lexer) Next() (*Token, error) {
	for {
		token, err := (*tokenizer)(l).Next()
		if err != nil {
			return token, err
		}
		switch token.Type {
		case WORD_TOKEN, WORDBREAK_TOKEN:
			return token, nil
		case COMMENT_TOKEN:
			// skip comments
		default:
			return nil, fmt.Errorf("unknown token type: %v", token.Type)
		}
	}
}

// tokenizer turns an input stream into a sequence of typed tokens
type tokenizer struct {
	input      bufio.Reader
	classifier tokenClassifier
	index      int
	state      LexerState
}

func (t *tokenizer) ReadRune() (r rune, size int, err error) {
	if r, size, err = t.input.ReadRune(); err == nil {
		t.index += 1
	}
	return
}

func (t *tokenizer) UnreadRune() (err error) {
	if err = t.input.UnreadRune(); err == nil {
		t.index -= 1
	}
	return
}

// newTokenizer creates a new tokenizer from an input stream.
func newTokenizer(r io.Reader) *tokenizer {
	input := bufio.NewReader(r)
	classifier := newDefaultClassifier()
	return &tokenizer{
		input:      *input,
		classifier: classifier}
}

// scanStream scans the stream for the next token using the internal state machine.
// It will panic if it encounters a rune which it does not know how to handle.
func (t *tokenizer) scanStream() (*Token, error) {
	previousState := t.state
	t.state = START_STATE
	token := &Token{}
	var nextRune rune
	var nextRuneType runeTokenClass
	var err error
	consumed := 0

	for {
		nextRune, _, err = t.ReadRune()
		nextRuneType = t.classifier.ClassifyRune(nextRune)
		token.RawValue += string(nextRune)
		consumed += 1 // TODO find a nicer solution for this

		switch {
		case err == io.EOF:
			nextRuneType = eofRuneClass
			err = nil
		case err != nil:
			return nil, err
		}

		switch t.state {
		case START_STATE: // no runes read yet
			{
				if nextRuneType != spaceRuneClass {
					token.Index = t.index - 1
				}
				switch nextRuneType {
				case eofRuneClass:
					switch {
					case t.index == 0: // tonkenizer contains an empty string
						token.removeLastRaw()
						token.Type = WORD_TOKEN
						token.Index = t.index
						t.index += 1
						return token, nil // return an additional empty token for current cursor position
					case previousState == WORDBREAK_STATE, consumed > 1: // consumed is greater than 1 when when there were spaceRunes before
						token.removeLastRaw()
						token.Type = WORD_TOKEN
						token.Index = t.index
						return token, nil // return an additional empty token for current cursor position
					default:
						return nil, io.EOF
					}
				case spaceRuneClass:
					token.removeLastRaw()
				case escapingQuoteRuneClass:
					token.Type = WORD_TOKEN
					t.state = QUOTING_ESCAPING_STATE
					token.WordbreakIndex = len(token.Value)
				case nonEscapingQuoteRuneClass:
					token.Type = WORD_TOKEN
					t.state = QUOTING_STATE
					token.WordbreakIndex = len(token.Value)
				case escapeRuneClass:
					token.Type = WORD_TOKEN
					t.state = ESCAPING_STATE
				case commentRuneClass:
					token.Type = COMMENT_TOKEN
					t.state = COMMENT_STATE
				case wordbreakRuneClass:
					token.Type = WORDBREAK_TOKEN
					token.add(nextRune)
					t.state = WORDBREAK_STATE
				default:
					token.Type = WORD_TOKEN
					token.add(nextRune)
					t.state = IN_WORD_STATE
				}
			}
		case WORDBREAK_STATE:
			switch nextRuneType {
			case wordbreakRuneClass:
				token.add(nextRune)
			default:
				token.removeLastRaw()
				t.UnreadRune()
				return token, err
			}
		case IN_WORD_STATE: // in a regular word
			switch nextRuneType {
			case wordbreakRuneClass:
				token.removeLastRaw()
				t.UnreadRune()
				return token, err
			case eofRuneClass, spaceRuneClass:
				token.removeLastRaw()
				t.UnreadRune()
				return token, err
			case escapingQuoteRuneClass:
				t.state = QUOTING_ESCAPING_STATE
				token.WordbreakIndex = len(token.Value)
			case nonEscapingQuoteRuneClass:
				t.state = QUOTING_STATE
				token.WordbreakIndex = len(token.Value)
			case escapeRuneClass:
				t.state = ESCAPING_STATE
			default:
				token.add(nextRune)
			}
		case ESCAPING_STATE: // the rune after an escape character
			switch nextRuneType {
			case eofRuneClass: // EOF found after escape character
				token.removeLastRaw()
				return token, err
			default:
				t.state = IN_WORD_STATE
				token.add(nextRune)
			}
		case ESCAPING_QUOTED_STATE: // the next rune after an escape character, in double quotes
			switch nextRuneType {
			case eofRuneClass: // EOF found after escape character
				token.removeLastRaw()
				return token, err
			default:
				t.state = QUOTING_ESCAPING_STATE
				token.add(nextRune)
			}
		case QUOTING_ESCAPING_STATE: // in escaping double quotes
			switch nextRuneType {
			case eofRuneClass: // EOF found when expecting closing quote
				token.removeLastRaw()
				return token, err
			case escapingQuoteRuneClass:
				t.state = IN_WORD_STATE
			case escapeRuneClass:
				t.state = ESCAPING_QUOTED_STATE
			default:
				token.add(nextRune)
			}
		case QUOTING_STATE: // in non-escaping single quotes
			switch nextRuneType {
			case eofRuneClass: // EOF found when expecting closing quote
				token.removeLastRaw()
				return token, err
			case nonEscapingQuoteRuneClass:
				t.state = IN_WORD_STATE
			default:
				token.add(nextRune)
			}
		case COMMENT_STATE: // in a comment
			switch nextRuneType {
			case eofRuneClass:
				return token, err
			case spaceRuneClass:
				if nextRune == '\n' {
					token.removeLastRaw()
					t.state = START_STATE
					return token, err
				} else {
					token.add(nextRune)
				}
			default:
				token.add(nextRune)
			}
		default:
			return nil, fmt.Errorf("unexpected state: %v", t.state)
		}
	}
}

// Next returns the next token in the stream.
func (t *tokenizer) Next() (*Token, error) {
	token, err := t.scanStream()
	if err == nil {
		token.State = t.state // TODO should be done in scanStream
		token.WordbreakType = wordbreakType(*token)
	}
	return token, err
}

// Split partitions of a string into tokens.
func Split(s string) (TokenSlice, error) {
	l := newLexer(strings.NewReader(s))
	tokens := make(TokenSlice, 0)
	for {
		token, err := l.Next()
		if err != nil {
			if err == io.EOF {
				return tokens, nil
			}
			return nil, err
		}
		tokens = append(tokens, *token)
	}
}

// Join concatenates words to create a single string.
// It quotes and escapes where appropriate.
// TODO experimental
func Join(s []string) string {
	replacer := strings.NewReplacer(
		"$", "\\$",
		"`", "\\`",
	)

	formatted := make([]string, 0, len(s))
	for _, arg := range s {
		switch {
		case arg == "",
			strings.ContainsAny(arg, `"' `+"\n\r\t"):
			formatted = append(formatted, replacer.Replace(fmt.Sprintf("%#v", arg)))
		default:
			formatted = append(formatted, arg)
		}
	}
	return strings.Join(formatted, " ")
}
