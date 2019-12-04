/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/selection"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog"
)

// Requirements is AND of all requirements.
type Requirements []Requirement

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

	// Requirements converts this interface into Requirements to expose
	// more detailed selection information.
	// If there are querying parameters, it will return converted requirements and selectable=true.
	// If this selector doesn't want to select anything, it will return selectable=false.
	Requirements() (requirements Requirements, selectable bool)

	// Make a deep copy of the selector.
	DeepCopySelector() Selector

	// RequiresExactMatch allows a caller to introspect whether a given selector
	// requires a single specific label to be set, and if so returns the value it
	// requires.
	RequiresExactMatch(label string) (value string, found bool)
}

// Everything returns a selector that matches all labels.
func Everything() Selector {
	return internalSelector{}
}

type nothingSelector struct{}

func (n nothingSelector) Matches(_ Labels) bool                                      { return false }
func (n nothingSelector) Empty() bool                                                { return false }
func (n nothingSelector) String() string                                             { return "" }
func (n nothingSelector) Add(_ ...Requirement) Selector                              { return n }
func (n nothingSelector) Requirements() (Requirements, bool)                         { return nil, false }
func (n nothingSelector) DeepCopySelector() Selector                                 { return n }
func (n nothingSelector) RequiresExactMatch(label string) (value string, found bool) { return "", false }

// Nothing returns a selector that matches no labels
func Nothing() Selector {
	return nothingSelector{}
}

// NewSelector returns a nil selector
func NewSelector() Selector {
	return internalSelector(nil)
}

type internalSelector []Requirement

func (s internalSelector) DeepCopy() internalSelector {
	if s == nil {
		return nil
	}
	result := make([]Requirement, len(s))
	for i := range s {
		s[i].DeepCopyInto(&result[i])
	}
	return result
}

func (s internalSelector) DeepCopySelector() Selector {
	return s.DeepCopy()
}

// ByKey sorts requirements by key to obtain deterministic parser
type ByKey []Requirement

func (a ByKey) Len() int { return len(a) }

func (a ByKey) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func (a ByKey) Less(i, j int) bool { return a[i].key < a[j].key }

// Requirement contains values, a key, and an operator that relates the key and values.
// The zero value of Requirement is invalid.
// Requirement implements both set based match and exact match
// Requirement should be initialized via NewRequirement constructor for creating a valid Requirement.
// +k8s:deepcopy-gen=true
type Requirement struct {
	key      string
	operator selection.Operator
	// In huge majority of cases we have at most one value here.
	// It is generally faster to operate on a single-element slice
	// than on a single-element map, so we have a slice here.
	strValues []string
}

// NewRequirement is the constructor for a Requirement.
// If any of these rules is violated, an error is returned:
// (1) The operator can only be In, NotIn, Equals, DoubleEquals, NotEquals, Exists, or DoesNotExist.
// (2) If the operator is In or NotIn, the values set must be non-empty.
// (3) If the operator is Equals, DoubleEquals, or NotEquals, the values set must contain one value.
// (4) If the operator is Exists or DoesNotExist, the value set must be empty.
// (5) If the operator is Gt or Lt, the values set must contain only one value, which will be interpreted as an integer.
// (6) The key is invalid due to its length, or sequence
//     of characters. See validateLabelKey for more details.
//
// The empty string is a valid value in the input values set.
func NewRequirement(key string, op selection.Operator, vals []string) (*Requirement, error) {
	if err := validateLabelKey(key); err != nil {
		return nil, err
	}
	switch op {
	case selection.In, selection.NotIn:
		if len(vals) == 0 {
			return nil, fmt.Errorf("for 'in', 'notin' operators, values set can't be empty")
		}
	case selection.Equals, selection.DoubleEquals, selection.NotEquals:
		if len(vals) != 1 {
			return nil, fmt.Errorf("exact-match compatibility requires one single value")
		}
	case selection.Exists, selection.DoesNotExist:
		if len(vals) != 0 {
			return nil, fmt.Errorf("values set must be empty for exists and does not exist")
		}
	case selection.GreaterThan, selection.LessThan:
		if len(vals) != 1 {
			return nil, fmt.Errorf("for 'Gt', 'Lt' operators, exactly one value is required")
		}
		for i := range vals {
			if _, err := strconv.ParseInt(vals[i], 10, 64); err != nil {
				return nil, fmt.Errorf("for 'Gt', 'Lt' operators, the value must be an integer")
			}
		}
	default:
		return nil, fmt.Errorf("operator '%v' is not recognized", op)
	}

	for i := range vals {
		if err := validateLabelValue(key, vals[i]); err != nil {
			return nil, err
		}
	}
	return &Requirement{key: key, operator: op, strValues: vals}, nil
}

func (r *Requirement) hasValue(value string) bool {
	for i := range r.strValues {
		if r.strValues[i] == value {
			return true
		}
	}
	return false
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
// (5) The operator is GreaterThanOperator or LessThanOperator, and Labels has
//     the Requirement's key and the corresponding value satisfies mathematical inequality.
func (r *Requirement) Matches(ls Labels) bool {
	switch r.operator {
	case selection.In, selection.Equals, selection.DoubleEquals:
		if !ls.Has(r.key) {
			return false
		}
		return r.hasValue(ls.Get(r.key))
	case selection.NotIn, selection.NotEquals:
		if !ls.Has(r.key) {
			return true
		}
		return !r.hasValue(ls.Get(r.key))
	case selection.Exists:
		return ls.Has(r.key)
	case selection.DoesNotExist:
		return !ls.Has(r.key)
	case selection.GreaterThan, selection.LessThan:
		if !ls.Has(r.key) {
			return false
		}
		lsValue, err := strconv.ParseInt(ls.Get(r.key), 10, 64)
		if err != nil {
			klog.V(10).Infof("ParseInt failed for value %+v in label %+v, %+v", ls.Get(r.key), ls, err)
			return false
		}

		// There should be only one strValue in r.strValues, and can be converted to a integer.
		if len(r.strValues) != 1 {
			klog.V(10).Infof("Invalid values count %+v of requirement %#v, for 'Gt', 'Lt' operators, exactly one value is required", len(r.strValues), r)
			return false
		}

		var rValue int64
		for i := range r.strValues {
			rValue, err = strconv.ParseInt(r.strValues[i], 10, 64)
			if err != nil {
				klog.V(10).Infof("ParseInt failed for value %+v in requirement %#v, for 'Gt', 'Lt' operators, the value must be an integer", r.strValues[i], r)
				return false
			}
		}
		return (r.operator == selection.GreaterThan && lsValue > rValue) || (r.operator == selection.LessThan && lsValue < rValue)
	default:
		return false
	}
}

// Key returns requirement key
func (r *Requirement) Key() string {
	return r.key
}

// Operator returns requirement operator
func (r *Requirement) Operator() selection.Operator {
	return r.operator
}

// Values returns requirement values
func (r *Requirement) Values() sets.String {
	ret := sets.String{}
	for i := range r.strValues {
		ret.Insert(r.strValues[i])
	}
	return ret
}

// Empty returns true if the internalSelector doesn't restrict selection space
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
	if r.operator == selection.DoesNotExist {
		buffer.WriteString("!")
	}
	buffer.WriteString(r.key)

	switch r.operator {
	case selection.Equals:
		buffer.WriteString("=")
	case selection.DoubleEquals:
		buffer.WriteString("==")
	case selection.NotEquals:
		buffer.WriteString("!=")
	case selection.In:
		buffer.WriteString(" in ")
	case selection.NotIn:
		buffer.WriteString(" notin ")
	case selection.GreaterThan:
		buffer.WriteString(">")
	case selection.LessThan:
		buffer.WriteString("<")
	case selection.Exists, selection.DoesNotExist:
		return buffer.String()
	}

	switch r.operator {
	case selection.In, selection.NotIn:
		buffer.WriteString("(")
	}
	if len(r.strValues) == 1 {
		buffer.WriteString(r.strValues[0])
	} else { // only > 1 since == 0 prohibited by NewRequirement
		// normalizes value order on output, without mutating the in-memory selector representation
		// also avoids normalization when it is not required, and ensures we do not mutate shared data
		buffer.WriteString(strings.Join(safeSort(r.strValues), ","))
	}

	switch r.operator {
	case selection.In, selection.NotIn:
		buffer.WriteString(")")
	}
	return buffer.String()
}

// safeSort sort input strings without modification
func safeSort(in []string) []string {
	if sort.StringsAreSorted(in) {
		return in
	}
	out := make([]string, len(in))
	copy(out, in)
	sort.Strings(out)
	return out
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

func (lsel internalSelector) Requirements() (Requirements, bool) { return Requirements(lsel), true }

// String returns a comma-separated string of all
// the internalSelector Requirements' human-readable strings.
func (lsel internalSelector) String() string {
	var reqs []string
	for ix := range lsel {
		reqs = append(reqs, lsel[ix].String())
	}
	return strings.Join(reqs, ",")
}

// RequiresExactMatch introspect whether a given selector requires a single specific field
// to be set, and if so returns the value it requires.
func (lsel internalSelector) RequiresExactMatch(label string) (value string, found bool) {
	for ix := range lsel {
		if lsel[ix].key == label {
			switch lsel[ix].operator {
			case selection.Equals, selection.DoubleEquals, selection.In:
				if len(lsel[ix].strValues) == 1 {
					return lsel[ix].strValues[0], true
				}
			}
			return "", false
		}
	}
	return "", false
}

// Token represents constant definition for lexer token
type Token int

const (
	// ErrorToken represents scan error
	ErrorToken Token = iota
	// EndOfStringToken represents end of string
	EndOfStringToken
	// ClosedParToken represents close parenthesis
	ClosedParToken
	// CommaToken represents the comma
	CommaToken
	// DoesNotExistToken represents logic not
	DoesNotExistToken
	// DoubleEqualsToken represents double equals
	DoubleEqualsToken
	// EqualsToken represents equal
	EqualsToken
	// GreaterThanToken represents greater than
	GreaterThanToken
	// IdentifierToken represents identifier, e.g. keys and values
	IdentifierToken
	// InToken represents in
	InToken
	// LessThanToken represents less than
	LessThanToken
	// NotEqualsToken represents not equal
	NotEqualsToken
	// NotInToken represents not in
	NotInToken
	// OpenParToken represents open parenthesis
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

// ScannedItem contains the Token and the literal produced by the lexer.
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

// scanIDOrKeyword scans string to recognize literal token (for example 'in') or an identifier.
func (l *Lexer) scanIDOrKeyword() (tok Token, lit string) {
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
		return l.scanIDOrKeyword()
	}
}

// Parser data structure contains the label selector parser data structure
type Parser struct {
	l            *Lexer
	scannedItems []ScannedItem
	position     int
}

// ParserContext represents context during parsing:
// some literal for example 'in' and 'notin' can be
// recognized as operator for example 'x in (a)' but
// it can be recognized as value for example 'value in (in)'
type ParserContext int

const (
	// KeyAndOperator represents key and operator
	KeyAndOperator ParserContext = iota
	// Values represents values
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

// consume returns current token and string. Increments the position
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
	if operator == selection.Exists || operator == selection.DoesNotExist { // operator found lookahead set checked
		return NewRequirement(key, operator, []string{})
	}
	operator, err = p.parseOperator()
	if err != nil {
		return nil, err
	}
	var values sets.String
	switch operator {
	case selection.In, selection.NotIn:
		values, err = p.parseValues()
	case selection.Equals, selection.DoubleEquals, selection.NotEquals, selection.GreaterThan, selection.LessThan:
		values, err = p.parseExactValue()
	}
	if err != nil {
		return nil, err
	}
	return NewRequirement(key, operator, values.List())

}

// parseKeyAndInferOperator parse literals.
// in case of no operator '!, in, notin, ==, =, !=' are found
// the 'exists' operator is inferred
func (p *Parser) parseKeyAndInferOperator() (string, selection.Operator, error) {
	var operator selection.Operator
	tok, literal := p.consume(Values)
	if tok == DoesNotExistToken {
		operator = selection.DoesNotExist
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
		if operator != selection.DoesNotExist {
			operator = selection.Exists
		}
	}
	return literal, operator, nil
}

// parseOperator return operator and eventually matchType
// matchType can be exact
func (p *Parser) parseOperator() (op selection.Operator, err error) {
	tok, lit := p.consume(KeyAndOperator)
	switch tok {
	// DoesNotExistToken shouldn't be here because it's a unary operator, not a binary operator
	case InToken:
		op = selection.In
	case EqualsToken:
		op = selection.Equals
	case DoubleEqualsToken:
		op = selection.DoubleEquals
	case GreaterThanToken:
		op = selection.GreaterThan
	case LessThanToken:
		op = selection.LessThan
	case NotInToken:
		op = selection.NotIn
	case NotEqualsToken:
		op = selection.NotEquals
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
//  <selector-syntax>         ::= <requirement> | <requirement> "," <selector-syntax>
//  <requirement>             ::= [!] KEY [ <set-based-restriction> | <exact-match-restriction> ]
//  <set-based-restriction>   ::= "" | <inclusion-exclusion> <value-set>
//  <inclusion-exclusion>     ::= <inclusion> | <exclusion>
//  <exclusion>               ::= "notin"
//  <inclusion>               ::= "in"
//  <value-set>               ::= "(" <values> ")"
//  <values>                  ::= VALUE | VALUE "," <values>
//  <exact-match-restriction> ::= ["="|"=="|"!="] VALUE
//
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

func validateLabelKey(k string) error {
	if errs := validation.IsQualifiedName(k); len(errs) != 0 {
		return fmt.Errorf("invalid label key %q: %s", k, strings.Join(errs, "; "))
	}
	return nil
}

func validateLabelValue(k, v string) error {
	if errs := validation.IsValidLabelValue(v); len(errs) != 0 {
		return fmt.Errorf("invalid label value: %q: at key: %q: %s", v, k, strings.Join(errs, "; "))
	}
	return nil
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil and empty Sets are considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil || len(ls) == 0 {
		return internalSelector{}
	}
	requirements := make([]Requirement, 0, len(ls))
	for label, value := range ls {
		r, err := NewRequirement(label, selection.Equals, []string{value})
		if err == nil {
			requirements = append(requirements, *r)
		} else {
			//TODO: double check errors when input comes from serialization?
			return internalSelector{}
		}
	}
	// sort to have deterministic string representation
	sort.Sort(ByKey(requirements))
	return internalSelector(requirements)
}

// SelectorFromValidatedSet returns a Selector which will match exactly the given Set.
// A nil and empty Sets are considered equivalent to Everything().
// It assumes that Set is already validated and doesn't do any validation.
func SelectorFromValidatedSet(ls Set) Selector {
	if ls == nil || len(ls) == 0 {
		return internalSelector{}
	}
	requirements := make([]Requirement, 0, len(ls))
	for label, value := range ls {
		requirements = append(requirements, Requirement{key: label, operator: selection.Equals, strValues: []string{value}})
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
