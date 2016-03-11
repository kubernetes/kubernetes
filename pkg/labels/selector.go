/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package labels

import (
	"bytes"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation"
)

// Selector represents a label selector.
type Selector interface {
	// Matches returns true if this selector matches the given set of labels.
	Matches(Labels) bool

	// Empty returns true if this selector does not restrict the selection space.
	Empty() bool

	// String returns a human readable string that represents this selector.
	String() string

	// Add adds requirements to the Selector
	Add(r ...Requirement) Selector
}

// Everything returns a selector that matches all labels.
func Everything() Selector {
	return internalSelector{}
}

type nothingSelector struct{}

func (n nothingSelector) Matches(_ Labels) bool         { return false }
func (n nothingSelector) Empty() bool                   { return false }
func (n nothingSelector) String() string                { return "<null>" }
func (n nothingSelector) Add(_ ...Requirement) Selector { return n }

// Nothing returns a selector that matches no labels
func Nothing() Selector {
	return nothingSelector{}
}

// Operator represents a key's relationship
// to a set of values in a Requirement.
type Operator string

const (
	DoesNotExistOperator Operator = "!"
	EqualsOperator       Operator = "="
	DoubleEqualsOperator Operator = "=="
	InOperator           Operator = "in"
	NotEqualsOperator    Operator = "!="
	NotInOperator        Operator = "notin"
	ExistsOperator       Operator = "exists"
	GreaterThanOperator  Operator = "gt"
	LessThanOperator     Operator = "lt"
)

func NewSelector() Selector {
	return internalSelector(nil)
}

type internalSelector []Requirement

// Sort by key to obtain determisitic parser
type ByKey []Requirement

func (a ByKey) Len() int { return len(a) }

func (a ByKey) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func (a ByKey) Less(i, j int) bool { return a[i].key < a[j].key }

// Requirement is a selector that contains values, a key
// and an operator that relates the key and values. The zero
// value of Requirement is invalid.
// Requirement implements both set based match and exact match
// Requirement is initialized via NewRequirement constructor for creating a valid Requirement.
type Requirement struct {
	key       string
	operator  Operator
	strValues sets.String
}

// NewRequirement is the constructor for a Requirement.
// If any of these rules is violated, an error is returned:
// (1) The operator can only be In, NotIn, Equals, DoubleEquals, NotEquals, Exists, or DoesNotExist.
// (2) If the operator is In or NotIn, the values set must be non-empty.
// (3) If the operator is Equals, DoubleEquals, or NotEquals, the values set must contain one value.
// (4) If the operator is Exists or DoesNotExist, the value set must be empty.
// (5) If the operator is Gt or Lt, the values set must contain only one value.
// (6) The key is invalid due to its length, or sequence
//     of characters. See validateLabelKey for more details.
//
// The empty string is a valid value in the input values set.
func NewRequirement(key string, op Operator, vals sets.String) (*Requirement, error) {
	if err := validateLabelKey(key); err != nil {
		return nil, err
	}
	switch op {
	case InOperator, NotInOperator:
		if len(vals) == 0 {
			return nil, fmt.Errorf("for 'in', 'notin' operators, values set can't be empty")
		}
	case EqualsOperator, DoubleEqualsOperator, NotEqualsOperator:
		if len(vals) != 1 {
			return nil, fmt.Errorf("exact-match compatibility requires one single value")
		}
	case ExistsOperator, DoesNotExistOperator:
		if len(vals) != 0 {
			return nil, fmt.Errorf("values set must be empty for exists and does not exist")
		}
	case GreaterThanOperator, LessThanOperator:
		if len(vals) != 1 {
			return nil, fmt.Errorf("for 'Gt', 'Lt' operators, exactly one value is required")
		}
		for val := range vals {
			if _, err := strconv.ParseFloat(val, 64); err != nil {
				return nil, fmt.Errorf("for 'Gt', 'Lt' operators, the value must be a number")
			}
		}
	default:
		return nil, fmt.Errorf("operator '%v' is not recognized", op)
	}

	for v := range vals {
		if err := validateLabelValue(v); err != nil {
			return nil, err
		}
	}
	return &Requirement{key: key, operator: op, strValues: vals}, nil
}

// Matches returns true if the Requirement matches the input Labels.
// There is a match in the following cases:
// (1) The operator is Exists and Labels has the Requirement's key.
// (2) The operator is In, Labels has the Requirement's key and Labels'
//     value for that key is in Requirement's value set.
// (3) The operator is NotIn, Labels has the Requirement's key and
//     Labels' value for that key is not in Requirement's value set.
// (4) The operator is DoesNotExist or NotIn and Labels does not have the
//     Requirement's key.
func (r *Requirement) Matches(ls Labels) bool {
	switch r.operator {
	case InOperator, EqualsOperator, DoubleEqualsOperator:
		if !ls.Has(r.key) {
			return false
		}
		return r.strValues.Has(ls.Get(r.key))
	case NotInOperator, NotEqualsOperator:
		if !ls.Has(r.key) {
			return true
		}
		return !r.strValues.Has(ls.Get(r.key))
	case ExistsOperator:
		return ls.Has(r.key)
	case DoesNotExistOperator:
		return !ls.Has(r.key)
	case GreaterThanOperator, LessThanOperator:
		if !ls.Has(r.key) {
			return false
		}
		lsValue, err := strconv.ParseFloat(ls.Get(r.key), 64)
		if err != nil {
			glog.V(10).Infof("Parse float failed for value %+v in label %+v, %+v", ls.Get(r.key), ls, err)
			return false
		}

		// There should be only one strValue in r.strValues, and can be converted to a float number.
		if len(r.strValues) != 1 {
			glog.V(10).Infof("Invalid values count %+v of requirement %+v, for 'Gt', 'Lt' operators, exactly one value is required", len(r.strValues), r)
			return false
		}

		var rValue float64
		for strValue := range r.strValues {
			rValue, err = strconv.ParseFloat(strValue, 64)
			if err != nil {
				glog.V(10).Infof("Parse float failed for value %+v in requirement %+v, for 'Gt', 'Lt' operators, the value must be a number", strValue, r)
				return false
			}
		}
		return (r.operator == GreaterThanOperator && lsValue > rValue) || (r.operator == LessThanOperator && lsValue < rValue)
	default:
		return false
	}
}

func (r *Requirement) Key() string {
	return r.key
}
func (r *Requirement) Operator() Operator {
	return r.operator
}
func (r *Requirement) Values() sets.String {
	ret := sets.String{}
	for k := range r.strValues {
		ret.Insert(k)
	}
	return ret
}

// Return true if the internalSelector doesn't restrict selection space
func (lsel internalSelector) Empty() bool {
	if lsel == nil {
		return true
	}
	return len(lsel) == 0
}

// String returns a human-readable string that represents this
// Requirement. If called on an invalid Requirement, an error is
// returned. See NewRequirement for creating a valid Requirement.
func (r *Requirement) String() string {
	var buffer bytes.Buffer
	if r.operator == DoesNotExistOperator {
		buffer.WriteString("!")
	}
	buffer.WriteString(r.key)

	switch r.operator {
	case EqualsOperator:
		buffer.WriteString("=")
	case DoubleEqualsOperator:
		buffer.WriteString("==")
	case NotEqualsOperator:
		buffer.WriteString("!=")
	case InOperator:
		buffer.WriteString(" in ")
	case NotInOperator:
		buffer.WriteString(" notin ")
	case GreaterThanOperator:
		buffer.WriteString(">")
	case LessThanOperator:
		buffer.WriteString("<")
	case ExistsOperator, DoesNotExistOperator:
		return buffer.String()
	}

	switch r.operator {
	case InOperator, NotInOperator:
		buffer.WriteString("(")
	}
	if len(r.strValues) == 1 {
		buffer.WriteString(r.strValues.List()[0])
	} else { // only > 1 since == 0 prohibited by NewRequirement
		buffer.WriteString(strings.Join(r.strValues.List(), ","))
	}

	switch r.operator {
	case InOperator, NotInOperator:
		buffer.WriteString(")")
	}
	return buffer.String()
}

// Add adds requirements to the selector. It copies the current selector returning a new one
func (lsel internalSelector) Add(reqs ...Requirement) Selector {
	var sel internalSelector
	for ix := range lsel {
		sel = append(sel, lsel[ix])
	}
	for _, r := range reqs {
		sel = append(sel, r)
	}
	sort.Sort(ByKey(sel))
	return sel
}

// Matches for a internalSelector returns true if all
// its Requirements match the input Labels. If any
// Requirement does not match, false is returned.
func (lsel internalSelector) Matches(l Labels) bool {
	for ix := range lsel {
		if matches := lsel[ix].Matches(l); !matches {
			return false
		}
	}
	return true
}

// String returns a comma-separated string of all
// the internalSelector Requirements' human-readable strings.
func (lsel internalSelector) String() string {
	var reqs []string
	for ix := range lsel {
		reqs = append(reqs, lsel[ix].String())
	}
	return strings.Join(reqs, ",")
}

// constants definition for lexer token
type Token int

const (
	ErrorToken Token = iota
	EndOfStringToken
	ClosedParToken
	CommaToken
	DoesNotExistToken
	DoubleEqualsToken
	EqualsToken
	GreaterThanToken
	IdentifierToken // to represent keys and values
	InToken
	LessThanToken
	NotEqualsToken
	NotInToken
	OpenParToken
)

// string2token contains the mapping between lexer Token and token literal
// (except IdentifierToken, EndOfStringToken and ErrorToken since it makes no sense)
var string2token = map[string]Token{
	")":     ClosedParToken,
	",":     CommaToken,
	"!":     DoesNotExistToken,
	"==":    DoubleEqualsToken,
	"=":     EqualsToken,
	">":     GreaterThanToken,
	"in":    InToken,
	"<":     LessThanToken,
	"!=":    NotEqualsToken,
	"notin": NotInToken,
	"(":     OpenParToken,
}

// The item produced by the lexer. It contains the Token and the literal.
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
	case '=', '!', '(', ')', ',', '>', '<':
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

// scanIdOrKeyword scans string to recognize literal token (for example 'in') or an identifier.
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
	if val, ok := string2token[s]; ok { // is a literal token?
		return val, s
	}
	return IdentifierToken, s // otherwise is an identifier
}

// scanSpecialSymbol scans string starting with special symbol.
// special symbol identify non literal operators. "!=", "==", "="
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

// Parser data structure contains the label selector parser data strucutre
type Parser struct {
	l            *Lexer
	scannedItems []ScannedItem
	position     int
}

// Parser context represents context during parsing:
// some literal for example 'in' and 'notin' can be
// recognized as operator for example 'x in (a)' but
// it can be recognized as value for example 'value in (in)'
type ParserContext int

const (
	KeyAndOperator ParserContext = iota
	Values
)

// lookahead func returns the current token and string. No increment of current position
func (p *Parser) lookahead(context ParserContext) (Token, string) {
	tok, lit := p.scannedItems[p.position].tok, p.scannedItems[p.position].literal
	if context == Values {
		switch tok {
		case InToken, NotInToken:
			tok = IdentifierToken
		}
	}
	return tok, lit
}

// consume returns current token and string. Increments the the position
func (p *Parser) consume(context ParserContext) (Token, string) {
	p.position++
	tok, lit := p.scannedItems[p.position-1].tok, p.scannedItems[p.position-1].literal
	if context == Values {
		switch tok {
		case InToken, NotInToken:
			tok = IdentifierToken
		}
	}
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
// on input string. It returns a list of Requirement objects.
func (p *Parser) parse() (internalSelector, error) {
	p.scan() // init scannedItems

	var requirements internalSelector
	for {
		tok, lit := p.lookahead(Values)
		switch tok {
		case IdentifierToken, DoesNotExistToken:
			r, err := p.parseRequirement()
			if err != nil {
				return nil, fmt.Errorf("unable to parse requirement: %v", err)
			}
			requirements = append(requirements, *r)
			t, l := p.consume(Values)
			switch t {
			case EndOfStringToken:
				return requirements, nil
			case CommaToken:
				t2, l2 := p.lookahead(Values)
				if t2 != IdentifierToken && t2 != DoesNotExistToken {
					return nil, fmt.Errorf("found '%s', expected: identifier after ','", l2)
				}
			default:
				return nil, fmt.Errorf("found '%s', expected: ',' or 'end of string'", l)
			}
		case EndOfStringToken:
			return requirements, nil
		default:
			return nil, fmt.Errorf("found '%s', expected: !, identifier, or 'end of string'", lit)
		}
	}
}

func (p *Parser) parseRequirement() (*Requirement, error) {
	key, operator, err := p.parseKeyAndInferOperator()
	if err != nil {
		return nil, err
	}
	if operator == ExistsOperator || operator == DoesNotExistOperator { // operator found lookahead set checked
		return NewRequirement(key, operator, nil)
	}
	operator, err = p.parseOperator()
	if err != nil {
		return nil, err
	}
	var values sets.String
	switch operator {
	case InOperator, NotInOperator:
		values, err = p.parseValues()
	case EqualsOperator, DoubleEqualsOperator, NotEqualsOperator, GreaterThanOperator, LessThanOperator:
		values, err = p.parseExactValue()
	}
	if err != nil {
		return nil, err
	}
	return NewRequirement(key, operator, values)

}

// parseKeyAndInferOperator parse literals.
// in case of no operator '!, in, notin, ==, =, !=' are found
// the 'exists' operator is inferred
func (p *Parser) parseKeyAndInferOperator() (string, Operator, error) {
	var operator Operator
	tok, literal := p.consume(Values)
	if tok == DoesNotExistToken {
		operator = DoesNotExistOperator
		tok, literal = p.consume(Values)
	}
	if tok != IdentifierToken {
		err := fmt.Errorf("found '%s', expected: identifier", literal)
		return "", "", err
	}
	if err := validateLabelKey(literal); err != nil {
		return "", "", err
	}
	if t, _ := p.lookahead(Values); t == EndOfStringToken || t == CommaToken {
		if operator != DoesNotExistOperator {
			operator = ExistsOperator
		}
	}
	return literal, operator, nil
}

// parseOperator return operator and eventually matchType
// matchType can be exact
func (p *Parser) parseOperator() (op Operator, err error) {
	tok, lit := p.consume(KeyAndOperator)
	switch tok {
	// DoesNotExistToken shouldn't be here because it's a unary operator, not a binary operator
	case InToken:
		op = InOperator
	case EqualsToken:
		op = EqualsOperator
	case DoubleEqualsToken:
		op = DoubleEqualsOperator
	case GreaterThanToken:
		op = GreaterThanOperator
	case LessThanToken:
		op = LessThanOperator
	case NotInToken:
		op = NotInOperator
	case NotEqualsToken:
		op = NotEqualsOperator
	default:
		return "", fmt.Errorf("found '%s', expected: '=', '!=', '==', 'in', notin'", lit)
	}
	return op, nil
}

// parseValues parses the values for set based matching (x,y,z)
func (p *Parser) parseValues() (sets.String, error) {
	tok, lit := p.consume(Values)
	if tok != OpenParToken {
		return nil, fmt.Errorf("found '%s' expected: '('", lit)
	}
	tok, lit = p.lookahead(Values)
	switch tok {
	case IdentifierToken, CommaToken:
		s, err := p.parseIdentifiersList() // handles general cases
		if err != nil {
			return s, err
		}
		if tok, _ = p.consume(Values); tok != ClosedParToken {
			return nil, fmt.Errorf("found '%s', expected: ')'", lit)
		}
		return s, nil
	case ClosedParToken: // handles "()"
		p.consume(Values)
		return sets.NewString(""), nil
	default:
		return nil, fmt.Errorf("found '%s', expected: ',', ')' or identifier", lit)
	}
}

// parseIdentifiersList parses a (possibly empty) list of
// of comma separated (possibly empty) identifiers
func (p *Parser) parseIdentifiersList() (sets.String, error) {
	s := sets.NewString()
	for {
		tok, lit := p.consume(Values)
		switch tok {
		case IdentifierToken:
			s.Insert(lit)
			tok2, lit2 := p.lookahead(Values)
			switch tok2 {
			case CommaToken:
				continue
			case ClosedParToken:
				return s, nil
			default:
				return nil, fmt.Errorf("found '%s', expected: ',' or ')'", lit2)
			}
		case CommaToken: // handled here since we can have "(,"
			if s.Len() == 0 {
				s.Insert("") // to handle (,
			}
			tok2, _ := p.lookahead(Values)
			if tok2 == ClosedParToken {
				s.Insert("") // to handle ,)  Double "" removed by StringSet
				return s, nil
			}
			if tok2 == CommaToken {
				p.consume(Values)
				s.Insert("") // to handle ,, Double "" removed by StringSet
			}
		default: // it can be operator
			return s, fmt.Errorf("found '%s', expected: ',', or identifier", lit)
		}
	}
}

// parseExactValue parses the only value for exact match style
func (p *Parser) parseExactValue() (sets.String, error) {
	s := sets.NewString()
	tok, lit := p.lookahead(Values)
	if tok == EndOfStringToken || tok == CommaToken {
		s.Insert("")
		return s, nil
	}
	tok, lit = p.consume(Values)
	if tok == IdentifierToken {
		s.Insert(lit)
		return s, nil
	}
	return nil, fmt.Errorf("found '%s', expected: identifier", lit)
}

// Parse takes a string representing a selector and returns a selector
// object, or an error. This parsing function differs from ParseSelector
// as they parse different selectors with different syntaxes.
// The input will cause an error if it does not follow this form:
//
// <selector-syntax> ::= <requirement> | <requirement> "," <selector-syntax> ]
// <requirement> ::= [!] KEY [ <set-based-restriction> | <exact-match-restriction> ]
// <set-based-restriction> ::= "" | <inclusion-exclusion> <value-set>
// <inclusion-exclusion> ::= <inclusion> | <exclusion>
//           <exclusion> ::= "notin"
//           <inclusion> ::= "in"
//           <value-set> ::= "(" <values> ")"
//              <values> ::= VALUE | VALUE "," <values>
// <exact-match-restriction> ::= ["="|"=="|"!="] VALUE
// KEY is a sequence of one or more characters following [ DNS_SUBDOMAIN "/" ] DNS_LABEL. Max length is 63 characters.
// VALUE is a sequence of zero or more characters "([A-Za-z0-9_-\.])". Max length is 63 characters.
// Delimiter is white space: (' ', '\t')
// Example of valid syntax:
//  "x in (foo,,baz),y,z notin ()"
//
// Note:
//  (1) Inclusion - " in " - denotes that the KEY exists and is equal to any of the
//      VALUEs in its requirement
//  (2) Exclusion - " notin " - denotes that the KEY is not equal to any
//      of the VALUEs in its requirement or does not exist
//  (3) The empty string is a valid VALUE
//  (4) A requirement with just a KEY - as in "y" above - denotes that
//      the KEY exists and can be any VALUE.
//  (5) A requirement with just !KEY requires that the KEY not exist.
//
func Parse(selector string) (Selector, error) {
	parsedSelector, err := parse(selector)
	if err == nil {
		return parsedSelector, nil
	}
	return nil, err
}

// parse parses the string representation of the selector and returns the internalSelector struct.
// The callers of this method can then decide how to return the internalSelector struct to their
// callers. This function has two callers now, one returns a Selector interface and the other
// returns a list of requirements.
func parse(selector string) (internalSelector, error) {
	p := &Parser{l: &Lexer{s: selector, pos: 0}}
	items, err := p.parse()
	if err != nil {
		return nil, err
	}
	sort.Sort(ByKey(items)) // sort to grant determistic parsing
	return internalSelector(items), err
}

var qualifiedNameErrorMsg string = fmt.Sprintf(`must be a qualified name (at most %d characters, matching regex %s), with an optional DNS subdomain prefix (at most %d characters, matching regex %s) and slash (/): e.g. "MyName" or "example.com/MyName"`, validation.QualifiedNameMaxLength, validation.QualifiedNameFmt, validation.DNS1123SubdomainMaxLength, validation.DNS1123SubdomainFmt)
var labelValueErrorMsg string = fmt.Sprintf(`must have at most %d characters, matching regex %s: e.g. "MyValue" or ""`, validation.LabelValueMaxLength, validation.LabelValueFmt)

func validateLabelKey(k string) error {
	if !validation.IsQualifiedName(k) {
		return fmt.Errorf("invalid label key: %s", qualifiedNameErrorMsg)
	}
	return nil
}

func validateLabelValue(v string) error {
	if !validation.IsValidLabelValue(v) {
		return fmt.Errorf("invalid label value: %s", labelValueErrorMsg)
	}
	return nil
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil and empty Sets are considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil {
		return internalSelector{}
	}
	var requirements internalSelector
	for label, value := range ls {
		if r, err := NewRequirement(label, EqualsOperator, sets.NewString(value)); err != nil {
			//TODO: double check errors when input comes from serialization?
			return internalSelector{}
		} else {
			requirements = append(requirements, *r)
		}
	}
	// sort to have deterministic string representation
	sort.Sort(ByKey(requirements))
	return internalSelector(requirements)
}

// ParseToRequirements takes a string representing a selector and returns a list of
// requirements. This function is suitable for those callers that perform additional
// processing on selector requirements.
// See the documentation for Parse() function for more details.
// TODO: Consider exporting the internalSelector type instead.
func ParseToRequirements(selector string) ([]Requirement, error) {
	return parse(selector)
}
