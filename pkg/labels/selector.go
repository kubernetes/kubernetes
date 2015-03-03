/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Selector represents a label selector.
type Selector interface {
	// Matches returns true if this selector matches the given set of labels.
	Matches(Labels) bool

	// Empty returns true if this selector does not restrict the selection space.
	Empty() bool

	// RequiresExactMatch allows a caller to introspect whether a given selector
	// requires a single specific label to be set, and if so returns the value it
	// requires.
	RequiresExactMatch(label string) (value string, found bool)

	// String returns a human readable string that represents this selector.
	String() string
}

// Everything returns a selector that matches all labels.
func Everything() Selector {
	return andTerm{}
}

type hasTerm struct {
	label, value string
}

func (t *hasTerm) Matches(ls Labels) bool {
	return ls.Get(t.label) == t.value
}

func (t *hasTerm) Empty() bool {
	return false
}

func (t *hasTerm) RequiresExactMatch(label string) (value string, found bool) {
	if t.label == label {
		return t.value, true
	}
	return "", false
}

func (t *hasTerm) String() string {
	return fmt.Sprintf("%v=%v", t.label, t.value)
}

type notHasTerm struct {
	label, value string
}

func (t *notHasTerm) Matches(ls Labels) bool {
	return ls.Get(t.label) != t.value
}

func (t *notHasTerm) Empty() bool {
	return false
}

func (t *notHasTerm) RequiresExactMatch(label string) (value string, found bool) {
	return "", false
}

func (t *notHasTerm) String() string {
	return fmt.Sprintf("%v!=%v", t.label, t.value)
}

type andTerm []Selector

func (t andTerm) Matches(ls Labels) bool {
	for _, q := range t {
		if !q.Matches(ls) {
			return false
		}
	}
	return true
}

func (t andTerm) Empty() bool {
	if t == nil {
		return true
	}
	if len([]Selector(t)) == 0 {
		return true
	}
	for i := range t {
		if !t[i].Empty() {
			return false
		}
	}
	return true
}

func (t andTerm) RequiresExactMatch(label string) (string, bool) {
	if t == nil || len([]Selector(t)) == 0 {
		return "", false
	}
	for i := range t {
		if value, found := t[i].RequiresExactMatch(label); found {
			return value, found
		}
	}
	return "", false
}

func (t andTerm) String() string {
	var terms []string
	for _, q := range t {
		terms = append(terms, q.String())
	}
	return strings.Join(terms, ",")
}

// Operator represents a key's relationship
// to a set of values in a Requirement.
type Operator string

const (
	EqualsOperator       Operator = "="
	DoubleEqualsOperator Operator = "=="
	InOperator           Operator = "in"
	NotEqualsOperator    Operator = "!="
	NotInOperator        Operator = "notin"
	ExistsOperator       Operator = "exists"
)

// LabelSelector contains a list of Requirements.
// LabelSelector is set-based and is distinguished from exact
// match-based selectors composed of key=value matching conjunctions.
type LabelSelector struct {
	Requirements []Requirement
}

// Sort by  obtain determisitic parser (minimic previous andTerm based stuff)
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
	strValues util.StringSet
}

// NewRequirement is the constructor for a Requirement.
// If any of these rules is violated, an error is returned:
// (1) The operator can only be In, NotIn or Exists.
// (2) If the operator is In or NotIn, the values set must
//     be non-empty.
// (3) The key is invalid due to its length, or sequence
//     of characters. See validateLabelKey for more details.
//
// The empty string is a valid value in the input values set.
func NewRequirement(key string, op Operator, vals util.StringSet) (*Requirement, error) {
	if err := validateLabelKey(key); err != nil {
		return nil, err
	}
	switch op {
	case InOperator, NotInOperator:
		if len(vals) == 0 {
			return nil, fmt.Errorf("for In,NotIn operators, values set can't be empty")
		}
	case EqualsOperator, DoubleEqualsOperator, NotEqualsOperator:
		if len(vals) != 1 {
			return nil, fmt.Errorf("exact match compatibility requires one single value")
		}
	case ExistsOperator:
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
// (4) The operator is NotIn and Labels does not have the
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
	default:
		return false
	}
}

// Return true if the LabelSelector doesn't restrict selection space
func (lsel LabelSelector) Empty() bool {
	if len(lsel.Requirements) == 0 {
		return true
	}

	return false
}

// RequiresExactMatch allows a caller to introspect whether a given selector
// requires a single specific label to be set, and if so returns the value it
// requires.
func (r *Requirement) RequiresExactMatch(label string) (string, bool) {
	if len(r.strValues) == 1 && r.operator == InOperator && r.key == label {
		return r.strValues.List()[0], true
	}
	return "", false
}

// String returns a human-readable string that represents this
// Requirement. If called on an invalid Requirement, an error is
// returned. See NewRequirement for creating a valid Requirement.
func (r *Requirement) String() string {
	var buffer bytes.Buffer
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
	case ExistsOperator:
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

// Matches for a LabelSelector returns true if all
// its Requirements match the input Labels. If any
// Requirement does not match, false is returned.
func (lsel LabelSelector) Matches(l Labels) bool {
	for _, req := range lsel.Requirements {
		if matches := req.Matches(l); !matches {
			return false
		}
	}
	return true
}

// RequiresExactMatch allows a caller to introspect whether a given selector
// requires a single specific label to be set, and if so returns the value it
// requires.
func (lsel LabelSelector) RequiresExactMatch(label string) (value string, found bool) {
	for _, req := range lsel.Requirements {
		if value, found = req.RequiresExactMatch(label); found {
			return value, found
		}
	}
	return "", false
}

// String returns a comma-separated string of all
// the LabelSelector Requirements' human-readable strings.
func (lsel LabelSelector) String() string {
	var reqs []string
	for _, req := range lsel.Requirements {
		reqs = append(reqs, req.String())
	}
	return strings.Join(reqs, ",")
}

// constants definition for lexer token
type Token int

const (
	ERROR Token = iota
	EOS         // end of string
	CPAR
	COMMA
	EEQUAL
	EQUAL
	IDENTIFIER
	IN
	NEQUAL
	NOTIN
	OPAR
	OR
)

// string2token contains the mapping between lexer Token and token literal
// (except IDENTIFIER, EOS and ERROR since it makes no sense)
var string2token = map[string]Token{
	")":     CPAR,
	",":     COMMA,
	"==":    EEQUAL,
	"=":     EQUAL,
	"in":    IN,
	"!=":    NEQUAL,
	"notin": NOTIN,
	"(":     OPAR,
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
	case '=', '!', '(', ')', ',':
		return true
	}
	return false
}

// Lexer struct
type Lexer struct {
	// s stores the string to be lexed
	s string
	// pos is the position currently lexed
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

// no return simply unread
func (l *Lexer) unread() {
	l.pos--
}

// func return a literal token (for example IN) and or an identifier.
func (l *Lexer) scanIdOrKeyword() (tok Token, lit string) {
	var buffer []byte
	for {
		if ch := l.read(); ch == 0 { // end of string found
			break
		} else if isSpecialSymbol(ch) || isWhitespace(ch) {
			l.unread() // stop scanning and unread
			break
		} else {
			buffer = append(buffer, ch)
		}
	}
	s := string(buffer)
	if val, ok := string2token[s]; ok { // is a literal token?
		return val, s
	}
	return IDENTIFIER, s // otherwise is an identifier
}

// scan string starting with specail symbol. At the moment this special symbols
// identify not literal operators
func (l *Lexer) scanSpecialSymbol() (Token, string) {
	lastScannedItem := ScannedItem{}
	var buffer []byte
	for {
		if ch := l.read(); ch == 0 {
			break
		} else if isSpecialSymbol(ch) {
			buffer = append(buffer, ch)
			if token, ok := string2token[string(buffer)]; ok {
				lastScannedItem = ScannedItem{tok: token, literal: string(buffer)}
			} else if lastScannedItem.tok != 0 {
				l.unread()
				break
			}
		} else { // in any other cases (identifer or whitespace) stop
			l.unread()
			break
		}
	}
	if lastScannedItem.tok == 0 {
		return ERROR, fmt.Sprintf("error expected keyword found '%s'", buffer)
	} else {
		return lastScannedItem.tok, lastScannedItem.literal
	}
}

// func Lex return Token and the literal (meaningfull only in case of IDENTIFIER)
func (l *Lexer) Lex() (tok Token, lit string) {
	ch := l.read()
	for { // consume spaces until no more spaces
		if !isWhitespace(ch) {
			break
		}
		ch = l.read()
	}
	if ch == 0 { // end of the string?
		return EOS, ""
	} else if isSpecialSymbol(ch) {
		l.unread()
		return l.scanSpecialSymbol() // can be an operator
	} else {
		l.unread()
		return l.scanIdOrKeyword()
	}
}

// Parser data structure contains the label selector parser data and algos
type Parser struct {
	l            *Lexer
	scannedItems []ScannedItem
	position     int
}

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
		case IN, NOTIN:
			tok = IDENTIFIER
		}
	}
	return tok, lit
}

// return current token and string. Increments the the position
func (p *Parser) consume(context ParserContext) (Token, string) {
	p.position++
	tok, lit := p.scannedItems[p.position-1].tok, p.scannedItems[p.position-1].literal
	if context == Values {
		switch tok {
		case IN, NOTIN:
			tok = IDENTIFIER
		}
	}
	return tok, lit
}

// scan method scan all the input string and storin <token, literal> pairs in
// scanned items slice.
// The Parser can now lookahead and consume the tokens
func (p *Parser) scan() {
	for {
		token, literal := p.l.Lex()
		p.scannedItems = append(p.scannedItems, ScannedItem{token, literal})
		if token == EOS {
			break
		}
	}
}

// the entry function to parse list of requirements
func (p *Parser) parse() ([]Requirement, error) {
	p.scan() // init scannedItems

	var requirements []Requirement
	for {
		tok, lit := p.lookahead(Values)
		switch tok {
		case IDENTIFIER:
			r, err := p.parseRequirement()
			if err != nil {
				return nil, fmt.Errorf("Error: ", err)
			}
			requirements = append(requirements, *r)
			t, l := p.consume(Values)
			switch t {
			case EOS:
				return requirements, nil
			case COMMA:
				t2, l2 := p.lookahead(Values)
				if t2 != IDENTIFIER {
					return nil, fmt.Errorf("Expected identifier after comma, found '%s'", l2)
				}
			default:
				return nil, fmt.Errorf("Bad value '%s', expetected comma or 'end of string'", l)
			}
		case EOS:
			return requirements, nil
		default:
			return nil, fmt.Errorf("Bad value %s. Expected identifier or 'end of string'", lit)
		}
	}
	return requirements, nil
}

// parse a Requirement data structure
func (p *Parser) parseRequirement() (*Requirement, error) {
	key, operator, err := p.parseKeyAndInferOperator()
	if err != nil {
		return nil, err
	}
	if operator == ExistsOperator { // operator Exists found lookahead set checked
		return NewRequirement(key, operator, nil)
	}
	operator, err = p.parseOperator()
	if err != nil {
		return nil, err
	}
	var values util.StringSet
	switch operator {
	case InOperator, NotInOperator:
		values, err = p.parseValues()
	case EqualsOperator, DoubleEqualsOperator, NotEqualsOperator:
		values, err = p.parseExactValue()
	}
	if err != nil {
		return nil, err
	}
	return NewRequirement(key, operator, values)

}

// parseKeyAndInferOperator parse literals.
func (p *Parser) parseKeyAndInferOperator() (string, Operator, error) {
	tok, literal := p.consume(Values)
	if tok != IDENTIFIER {
		err := fmt.Errorf("Found '%s' instead of expected IDENTIFIER", literal)
		return "", "", err
	}
	if err := validateLabelKey(literal); err != nil {
		return "", "", err
	}
	var operator Operator
	if t, _ := p.lookahead(Values); t == EOS || t == COMMA {
		operator = ExistsOperator
	}
	return literal, operator, nil
}

// parseOperator return operator and eventually matchType
// matchType can be exact
func (p *Parser) parseOperator() (op Operator, err error) {
	tok, lit := p.consume(KeyAndOperator)
	switch tok {
	case IN:
		op = InOperator
	case EQUAL:
		op = EqualsOperator
	case EEQUAL:
		op = DoubleEqualsOperator
	case NOTIN:
		op = NotInOperator
	case NEQUAL:
		op = NotEqualsOperator
	default:
		return "", fmt.Errorf("Expected '=', '!=', '==', 'in', notin', found %s", lit)
	}
	return op, nil
}

// parse values parse the values for set based matching (x,y,z)
func (p *Parser) parseValues() (util.StringSet, error) {
	tok, lit := p.consume(Values)
	if tok != OPAR {
		return nil, fmt.Errorf("Found '%s' expected '('", lit)
	}
	tok, lit = p.lookahead(Values)
	switch tok {
	case IDENTIFIER, COMMA:
		s, err := p.parseIdentifiersList() // handles general cases
		if err != nil {
			return s, err
		}
		if tok, _ = p.consume(Values); tok != CPAR {
			return nil, fmt.Errorf("Expected a ')', found '%s'", lit)
		}
		return s, nil
	case CPAR: // handles "()"
		p.consume(Values)
		return util.NewStringSet(""), nil
	default:
		return nil, fmt.Errorf("Expected ')' or ',' or identifier. Found '%s'", lit)
	}
	return util.NewStringSet(), nil
}

// parseIdentifiersList parse a (possibly empty) list of
// of comma separated (possibly empty) identifiers
func (p *Parser) parseIdentifiersList() (util.StringSet, error) {
	s := util.NewStringSet()
	for {
		tok, lit := p.consume(Values)
		switch tok {
		case IDENTIFIER:
			s.Insert(lit)
			tok2, lit2 := p.lookahead(Values)
			switch tok2 {
			case COMMA:
				continue
			case CPAR:
				return s, nil
			default:
				return nil, fmt.Errorf("Found '%s', expected ',' or ')'", lit2)
			}
		case COMMA: // handled here since we can have "(,"
			if s.Len() == 0 {
				s.Insert("") // to handle (,
			}
			tok2, _ := p.lookahead(Values)
			if tok2 == CPAR {
				s.Insert("") // to handle ,)  Double "" removed by StringSet
				return s, nil
			}
			if tok2 == COMMA {
				p.consume(Values)
				s.Insert("") // to handle ,, Double "" removed by StringSet
			}
		default: // it can be operator
			return s, fmt.Errorf("Found '%s', expected ',', or identifier", lit)
		}
	}
}

// parse the only value for exact match style
func (p *Parser) parseExactValue() (util.StringSet, error) {
	s := util.NewStringSet()
	if tok, lit := p.consume(Values); tok == IDENTIFIER {
		s.Insert(lit)
	} else {
		return nil, fmt.Errorf("Found '%s', expected identifier", lit)
	}
	return s, nil
}

// Parse takes a string representing a selector and returns a selector
// object, or an error. This parsing function differs from ParseSelector
// as they parse different selectors with different syntaxes.
// The input will cause an error if it does not follow this form:
//
// <selector-syntax> ::= <requirement> | <requirement> "," <selector-syntax> ]
// <requirement> ::= KEY [ <set-based-restriction> | <exact-match-restriction>
// <set-based-restriction> ::= "" | <inclusion-exclusion> <value-set>
// <inclusion-exclusion> ::= <inclusion> | <exclusion>
//           <exclusion> ::= "not" <inclusion>
//           <inclusion> ::= "in"
//           <value-set> ::= "(" <values> ")"
//              <values> ::= VALUE | VALUE "," <values>
// <exact-match-restriction> ::= ["="|"=="|"!="] VALUE
// KEY is a sequence of one or more characters following [ DNS_SUBDOMAIN "/" ] DNS_LABEL
// VALUE is a sequence of zero or more characters "([A-Za-z0-9_-\.])". Max length is 64 character.
// Delimiter is white space: (' ', '\t')
// Example of valid syntax:
//  "x in (foo,,baz),y,z not in ()"
//
// Note:
//  (1) Inclusion - " in " - denotes that the KEY is equal to any of the
//      VALUEs in its requirement
//  (2) Exclusion - " not in " - denotes that the KEY is not equal to any
//      of the VALUEs in its requirement
//  (3) The empty string is a valid VALUE
//  (4) A requirement with just a KEY - as in "y" above - denotes that
//      the KEY exists and can be any VALUE.
//
func Parse(selector string) (Selector, error) {
	p := &Parser{l: &Lexer{s: selector, pos: 0}}
	items, error := p.parse()
	if error == nil {
		sort.Sort(ByKey(items)) // sort to grant determistic parsing
		return &LabelSelector{Requirements: items}, error
	}
	return nil, error
}

const qualifiedNameErrorMsg string = "must match regex [" + util.DNS1123SubdomainFmt + " / ] " + util.DNS1123LabelFmt

func validateLabelKey(k string) error {
	if !util.IsQualifiedName(k) {
		return errors.NewFieldInvalid("label key", k, qualifiedNameErrorMsg)
	}
	return nil
}

func validateLabelValue(v string) error {
	if !util.IsValidLabelValue(v) {
		return errors.NewFieldInvalid("label value", v, qualifiedNameErrorMsg)
	}
	return nil
}

func try(selectorPiece, op string) (lhs, rhs string, ok bool) {
	pieces := strings.Split(selectorPiece, op)
	if len(pieces) == 2 {
		return pieces[0], pieces[1], true
	}
	return "", "", false
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil Set is considered equivalent to Everything().
func SelectorFromSet(ls Set) Selector {
	if ls == nil {
		return Everything()
	}
	items := make([]Selector, 0, len(ls))
	for label, value := range ls {
		items = append(items, &hasTerm{label: label, value: value})
	}
	if len(items) == 1 {
		return items[0]
	}
	return andTerm(items)
}

// SelectorFromSet returns a Selector which will match exactly the given Set. A
// nil Set is considered equivalent to Everything().
func SelectorFromSetParse(ls Set) (Selector, error) {
	if ls == nil {
		return LabelSelector{}, nil
	}
	var requirements []Requirement
	for label, value := range ls {
		if r, err := NewRequirement(label, InOperator, util.NewStringSet(value)); err != nil {
			return LabelSelector{}, err
		} else {
			requirements = append(requirements, *r)
		}
	}
	return LabelSelector{Requirements: requirements}, nil
}

// ParseSelector takes a string representing a selector and returns an
// object suitable for matching, or an error.
func ParseSelector(selector string) (Selector, error) {
	return parseSelector(selector,
		func(lhs, rhs string) (newLhs, newRhs string, err error) {
			return lhs, rhs, nil
		})
}

// Parses the selector and runs them through the given TransformFunc.
func ParseAndTransformSelector(selector string, fn TransformFunc) (Selector, error) {
	return parseSelector(selector, fn)
}

// Function to transform selectors.
type TransformFunc func(label, value string) (newLabel, newValue string, err error)

func parseSelector(selector string, fn TransformFunc) (Selector, error) {
	parts := strings.Split(selector, ",")
	sort.StringSlice(parts).Sort()
	var items []Selector
	for _, part := range parts {
		if part == "" {
			continue
		}
		if lhs, rhs, ok := try(part, "!="); ok {
			lhs, rhs, err := fn(lhs, rhs)
			if err != nil {
				return nil, err
			}
			items = append(items, &notHasTerm{label: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "=="); ok {
			lhs, rhs, err := fn(lhs, rhs)
			if err != nil {
				return nil, err
			}
			items = append(items, &hasTerm{label: lhs, value: rhs})
		} else if lhs, rhs, ok := try(part, "="); ok {
			lhs, rhs, err := fn(lhs, rhs)
			if err != nil {
				return nil, err
			}
			items = append(items, &hasTerm{label: lhs, value: rhs})
		} else {
			return nil, fmt.Errorf("invalid selector: '%s'; can't understand '%s'", selector, part)
		}
	}
	if len(items) == 1 {
		return items[0], nil
	}
	return andTerm(items), nil
}

// OneTermEqualSelector returns an object that matches objects where one label/field equals one value.
// Cannot return an error.
func OneTermEqualSelector(k, v string) Selector {
	return &hasTerm{label: k, value: v}
}

// OneTermEqualSelectorParse: implement OneTermEqualSelector using of LabelSelector and Requirement
// TODO: remove the original OneTermSelector  and rename OneTermEqualSelectorParse to OneTermEqualSelector
// Since OneTermEqualSelector cannot return an error. the Requirement based version ignore error.
// it's up to the caller being sure that k and v are not empty
func OneTermEqualSelectorParse(k, v string) Selector {
	r, _ := NewRequirement(k, InOperator, util.NewStringSet(v))
	return &LabelSelector{Requirements: []Requirement{*r}}
}
