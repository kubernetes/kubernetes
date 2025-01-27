// labelselector is trim down version of k8s/pkg/labels/selector.go
// It only accepts exact label matches
// Example: "k1=v1, k2 = v2"
package labelselector

import (
	"fmt"

	kvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// constants definition for lexer token
type Token int

const (
	ErrorToken Token = iota
	EndOfStringToken
	CommaToken
	EqualsToken
	IdentifierToken // to represent keys and values
)

// string2token contains the mapping between lexer Token and token literal
// (except IdentifierToken, EndOfStringToken and ErrorToken since it makes no sense)
var string2token = map[string]Token{
	",": CommaToken,
	"=": EqualsToken,
}

// ScannedItem are the item produced by the lexer. It contains the Token and the literal.
type ScannedItem struct {
	tok     Token
	literal string
}

// isWhitespace returns true if the rune is a space, tab, or newline.
func isWhitespace(ch byte) bool {
	return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'
}

// isSpecialSymbol detect if the character ch can be an operator
func isSpecialSymbol(ch byte) bool {
	switch ch {
	case '=', ',':
		return true
	}
	return false
}

// Lexer represents the Lexer struct for label selector.
// It contains necessary informationt to tokenize the input string
type Lexer struct {
	// s stores the string to be tokenized
	s string
	// pos is the position currently tokenized
	pos int
}

// read return the character currently lexed
// increment the position and check the buffer overflow
func (l *Lexer) read() (b byte) {
	b = 0
	if l.pos < len(l.s) {
		b = l.s[l.pos]
		l.pos++
	}
	return b
}

// unread 'undoes' the last read character
func (l *Lexer) unread() {
	l.pos--
}

// scanIdOrKeyword scans string to recognize literal token or an identifier.
func (l *Lexer) scanIdOrKeyword() (tok Token, lit string) {
	var buffer []byte
IdentifierLoop:
	for {
		switch ch := l.read(); {
		case ch == 0:
			break IdentifierLoop
		case isSpecialSymbol(ch) || isWhitespace(ch):
			l.unread()
			break IdentifierLoop
		default:
			buffer = append(buffer, ch)
		}
	}
	s := string(buffer)
	if val, ok := string2token[s]; ok { // is a literal token
		return val, s
	}
	return IdentifierToken, s // otherwise is an identifier
}

// scanSpecialSymbol scans string starting with special symbol.
// special symbol identify non literal operators: "="
func (l *Lexer) scanSpecialSymbol() (Token, string) {
	lastScannedItem := ScannedItem{}
	var buffer []byte
SpecialSymbolLoop:
	for {
		switch ch := l.read(); {
		case ch == 0:
			break SpecialSymbolLoop
		case isSpecialSymbol(ch):
			buffer = append(buffer, ch)
			if token, ok := string2token[string(buffer)]; ok {
				lastScannedItem = ScannedItem{tok: token, literal: string(buffer)}
			} else if lastScannedItem.tok != 0 {
				l.unread()
				break SpecialSymbolLoop
			}
		default:
			l.unread()
			break SpecialSymbolLoop
		}
	}
	if lastScannedItem.tok == 0 {
		return ErrorToken, fmt.Sprintf("error expected: keyword found '%s'", buffer)
	}
	return lastScannedItem.tok, lastScannedItem.literal
}

// skipWhiteSpaces consumes all blank characters
// returning the first non blank character
func (l *Lexer) skipWhiteSpaces(ch byte) byte {
	for {
		if !isWhitespace(ch) {
			return ch
		}
		ch = l.read()
	}
}

// Lex returns a pair of Token and the literal
// literal is meaningfull only for IdentifierToken token
func (l *Lexer) Lex() (tok Token, lit string) {
	switch ch := l.skipWhiteSpaces(l.read()); {
	case ch == 0:
		return EndOfStringToken, ""
	case isSpecialSymbol(ch):
		l.unread()
		return l.scanSpecialSymbol()
	default:
		l.unread()
		return l.scanIdOrKeyword()
	}
}

// Parser data structure contains the label selector parser data structure
type Parser struct {
	l            *Lexer
	scannedItems []ScannedItem
	position     int
}

// lookahead func returns the current token and string. No increment of current position
func (p *Parser) lookahead() (Token, string) {
	tok, lit := p.scannedItems[p.position].tok, p.scannedItems[p.position].literal
	return tok, lit
}

// consume returns current token and string. Increments the the position
func (p *Parser) consume() (Token, string) {
	p.position++
	if p.position > len(p.scannedItems) {
		return EndOfStringToken, ""
	}
	tok, lit := p.scannedItems[p.position-1].tok, p.scannedItems[p.position-1].literal
	return tok, lit
}

// scan runs through the input string and stores the ScannedItem in an array
// Parser can now lookahead and consume the tokens
func (p *Parser) scan() {
	for {
		token, literal := p.l.Lex()
		p.scannedItems = append(p.scannedItems, ScannedItem{token, literal})
		if token == EndOfStringToken {
			break
		}
	}
}

// parse runs the left recursive descending algorithm
// on input string. It returns a list of map[key]value.
func (p *Parser) parse() (map[string]string, error) {
	p.scan() // init scannedItems

	labelsMap := map[string]string{}
	for {
		tok, lit := p.lookahead()
		switch tok {
		case IdentifierToken:
			key, value, err := p.parseLabel()
			if err != nil {
				return nil, fmt.Errorf("unable to parse requirement: %v", err)
			}
			labelsMap[key] = value
			t, l := p.consume()
			switch t {
			case EndOfStringToken:
				return labelsMap, nil
			case CommaToken:
				t2, l2 := p.lookahead()
				if t2 != IdentifierToken {
					return nil, fmt.Errorf("found '%s', expected: identifier after ','", l2)
				}
			default:
				return nil, fmt.Errorf("found '%s', expected: ',' or 'end of string'", l)
			}
		case EndOfStringToken:
			return labelsMap, nil
		default:
			return nil, fmt.Errorf("found '%s', expected: identifier or 'end of string'", lit)
		}
	}
}

func (p *Parser) parseLabel() (string, string, error) {
	key, err := p.parseKey()
	if err != nil {
		return "", "", err
	}
	op, err := p.parseOperator()
	if err != nil {
		return "", "", err
	}
	if op != "=" {
		return "", "", fmt.Errorf("invalid operator: %s, expected: '='", op)
	}
	value, err := p.parseExactValue()
	if err != nil {
		return "", "", err
	}
	return key, value, nil
}

// parseKey parse literals.
func (p *Parser) parseKey() (string, error) {
	tok, literal := p.consume()
	if tok != IdentifierToken {
		err := fmt.Errorf("found '%s', expected: identifier", literal)
		return "", err
	}
	if err := validateLabelKey(literal); err != nil {
		return "", err
	}
	return literal, nil
}

// parseOperator returns operator
func (p *Parser) parseOperator() (op string, err error) {
	tok, lit := p.consume()
	switch tok {
	case EqualsToken:
		op = "="
	default:
		return "", fmt.Errorf("found '%s', expected: '='", lit)
	}
	return op, nil
}

// parseExactValue parses the only value for exact match style
func (p *Parser) parseExactValue() (string, error) {
	if tok, _ := p.lookahead(); tok == EndOfStringToken || tok == CommaToken {
		return "", nil
	}
	tok, lit := p.consume()
	if tok != IdentifierToken {
		return "", fmt.Errorf("found '%s', expected: identifier", lit)
	}
	if err := validateLabelValue(lit); err != nil {
		return "", err
	}
	return lit, nil
}

// Parse takes a string representing a selector and returns
// map[key]value, or an error.
// The input will cause an error if it does not follow this form:
//
// <selector-syntax> ::= [ <requirement> | <requirement> "," <selector-syntax> ]
// <requirement> ::= KEY "=" VALUE
// KEY is a sequence of one or more characters following [ DNS_SUBDOMAIN "/" ] DNS_LABEL
// VALUE is a sequence of zero or more characters "([A-Za-z0-9_-\.])". Max length is 64 character.
// Delimiter is white space: (' ', '\t')
func Parse(selector string) (map[string]string, error) {
	p := &Parser{l: &Lexer{s: selector, pos: 0}}
	labels, error := p.parse()
	if error != nil {
		return map[string]string{}, error
	}
	return labels, nil
}

// Conflicts takes 2 maps
// returns true if there a key match between the maps but the value doesn't match
// returns false in other cases
func Conflicts(labels1, labels2 map[string]string) bool {
	for k, v := range labels1 {
		if val, match := labels2[k]; match {
			if val != v {
				return true
			}
		}
	}
	return false
}

// Merge combines given maps
// Note: It doesn't not check for any conflicts between the maps
func Merge(labels1, labels2 map[string]string) map[string]string {
	mergedMap := map[string]string{}

	for k, v := range labels1 {
		mergedMap[k] = v
	}
	for k, v := range labels2 {
		mergedMap[k] = v
	}
	return mergedMap
}

// Equals returns true if the given maps are equal
func Equals(labels1, labels2 map[string]string) bool {
	if len(labels1) != len(labels2) {
		return false
	}

	for k, v := range labels1 {
		value, ok := labels2[k]
		if !ok {
			return false
		}
		if value != v {
			return false
		}
	}
	return true
}

const qualifiedNameErrorMsg string = "must match format [ DNS 1123 subdomain / ] DNS 1123 label"

func validateLabelKey(k string) error {
	if len(kvalidation.IsQualifiedName(k)) != 0 {
		return field.Invalid(field.NewPath("label key"), k, qualifiedNameErrorMsg)
	}
	return nil
}

func validateLabelValue(v string) error {
	if len(kvalidation.IsValidLabelValue(v)) != 0 {
		return field.Invalid(field.NewPath("label value"), v, qualifiedNameErrorMsg)
	}
	return nil
}
