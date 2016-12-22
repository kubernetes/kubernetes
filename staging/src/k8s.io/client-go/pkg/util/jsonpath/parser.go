/*
Copyright 2015 The Kubernetes Authors.

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

package jsonpath

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const eof = -1

const (
	leftDelim  = "{"
	rightDelim = "}"
)

type Parser struct {
	Name  string
	Root  *ListNode
	input string
	cur   *ListNode
	pos   int
	start int
	width int
}

// Parse parsed the given text and return a node Parser.
// If an error is encountered, parsing stops and an empty
// Parser is returned with the error
func Parse(name, text string) (*Parser, error) {
	p := NewParser(name)
	err := p.Parse(text)
	if err != nil {
		p = nil
	}
	return p, err
}

func NewParser(name string) *Parser {
	return &Parser{
		Name: name,
	}
}

// parseAction parsed the expression inside delimiter
func parseAction(name, text string) (*Parser, error) {
	p, err := Parse(name, fmt.Sprintf("%s%s%s", leftDelim, text, rightDelim))
	// when error happens, p will be nil, so we need to return here
	if err != nil {
		return p, err
	}
	p.Root = p.Root.Nodes[0].(*ListNode)
	return p, nil
}

func (p *Parser) Parse(text string) error {
	p.input = text
	p.Root = newList()
	p.pos = 0
	return p.parseText(p.Root)
}

// consumeText return the parsed text since last cosumeText
func (p *Parser) consumeText() string {
	value := p.input[p.start:p.pos]
	p.start = p.pos
	return value
}

// next returns the next rune in the input.
func (p *Parser) next() rune {
	if int(p.pos) >= len(p.input) {
		p.width = 0
		return eof
	}
	r, w := utf8.DecodeRuneInString(p.input[p.pos:])
	p.width = w
	p.pos += p.width
	return r
}

// peek returns but does not consume the next rune in the input.
func (p *Parser) peek() rune {
	r := p.next()
	p.backup()
	return r
}

// backup steps back one rune. Can only be called once per call of next.
func (p *Parser) backup() {
	p.pos -= p.width
}

func (p *Parser) parseText(cur *ListNode) error {
	for {
		if strings.HasPrefix(p.input[p.pos:], leftDelim) {
			if p.pos > p.start {
				cur.append(newText(p.consumeText()))
			}
			return p.parseLeftDelim(cur)
		}
		if p.next() == eof {
			break
		}
	}
	// Correctly reached EOF.
	if p.pos > p.start {
		cur.append(newText(p.consumeText()))
	}
	return nil
}

// parseLeftDelim scans the left delimiter, which is known to be present.
func (p *Parser) parseLeftDelim(cur *ListNode) error {
	p.pos += len(leftDelim)
	p.consumeText()
	newNode := newList()
	cur.append(newNode)
	cur = newNode
	return p.parseInsideAction(cur)
}

func (p *Parser) parseInsideAction(cur *ListNode) error {
	prefixMap := map[string]func(*ListNode) error{
		rightDelim: p.parseRightDelim,
		"[?(":      p.parseFilter,
		"..":       p.parseRecursive,
	}
	for prefix, parseFunc := range prefixMap {
		if strings.HasPrefix(p.input[p.pos:], prefix) {
			return parseFunc(cur)
		}
	}

	switch r := p.next(); {
	case r == eof || isEndOfLine(r):
		return fmt.Errorf("unclosed action")
	case r == ' ':
		p.consumeText()
	case r == '@' || r == '$': //the current object, just pass it
		p.consumeText()
	case r == '[':
		return p.parseArray(cur)
	case r == '"':
		return p.parseQuote(cur)
	case r == '.':
		return p.parseField(cur)
	case r == '+' || r == '-' || unicode.IsDigit(r):
		p.backup()
		return p.parseNumber(cur)
	case isAlphaNumeric(r):
		p.backup()
		return p.parseIdentifier(cur)
	default:
		return fmt.Errorf("unrecognized character in action: %#U", r)
	}
	return p.parseInsideAction(cur)
}

// parseRightDelim scans the right delimiter, which is known to be present.
func (p *Parser) parseRightDelim(cur *ListNode) error {
	p.pos += len(rightDelim)
	p.consumeText()
	cur = p.Root
	return p.parseText(cur)
}

// parseIdentifier scans build-in keywords, like "range" "end"
func (p *Parser) parseIdentifier(cur *ListNode) error {
	var r rune
	for {
		r = p.next()
		if isTerminator(r) {
			p.backup()
			break
		}
	}
	value := p.consumeText()
	cur.append(newIdentifier(value))
	return p.parseInsideAction(cur)
}

// parseRecursive scans the recursive desent operator ..
func (p *Parser) parseRecursive(cur *ListNode) error {
	p.pos += len("..")
	p.consumeText()
	cur.append(newRecursive())
	if r := p.peek(); isAlphaNumeric(r) {
		return p.parseField(cur)
	}
	return p.parseInsideAction(cur)
}

// parseNumber scans number
func (p *Parser) parseNumber(cur *ListNode) error {
	r := p.peek()
	if r == '+' || r == '-' {
		r = p.next()
	}
	for {
		r = p.next()
		if r != '.' && !unicode.IsDigit(r) {
			p.backup()
			break
		}
	}
	value := p.consumeText()
	i, err := strconv.Atoi(value)
	if err == nil {
		cur.append(newInt(i))
		return p.parseInsideAction(cur)
	}
	d, err := strconv.ParseFloat(value, 64)
	if err == nil {
		cur.append(newFloat(d))
		return p.parseInsideAction(cur)
	}
	return fmt.Errorf("cannot parse number %s", value)
}

// parseArray scans array index selection
func (p *Parser) parseArray(cur *ListNode) error {
Loop:
	for {
		switch p.next() {
		case eof, '\n':
			return fmt.Errorf("unterminated array")
		case ']':
			break Loop
		}
	}
	text := p.consumeText()
	text = string(text[1 : len(text)-1])
	if text == "*" {
		text = ":"
	}

	//union operator
	strs := strings.Split(text, ",")
	if len(strs) > 1 {
		union := []*ListNode{}
		for _, str := range strs {
			parser, err := parseAction("union", fmt.Sprintf("[%s]", strings.Trim(str, " ")))
			if err != nil {
				return err
			}
			union = append(union, parser.Root)
		}
		cur.append(newUnion(union))
		return p.parseInsideAction(cur)
	}

	// dict key
	reg := regexp.MustCompile(`^'([^']*)'$`)
	value := reg.FindStringSubmatch(text)
	if value != nil {
		parser, err := parseAction("arraydict", fmt.Sprintf(".%s", value[1]))
		if err != nil {
			return err
		}
		for _, node := range parser.Root.Nodes {
			cur.append(node)
		}
		return p.parseInsideAction(cur)
	}

	//slice operator
	reg = regexp.MustCompile(`^(-?[\d]*)(:-?[\d]*)?(:[\d]*)?$`)
	value = reg.FindStringSubmatch(text)
	if value == nil {
		return fmt.Errorf("invalid array index %s", text)
	}
	value = value[1:]
	params := [3]ParamsEntry{}
	for i := 0; i < 3; i++ {
		if value[i] != "" {
			if i > 0 {
				value[i] = value[i][1:]
			}
			if i > 0 && value[i] == "" {
				params[i].Known = false
			} else {
				var err error
				params[i].Known = true
				params[i].Value, err = strconv.Atoi(value[i])
				if err != nil {
					return fmt.Errorf("array index %s is not a number", value[i])
				}
			}
		} else {
			if i == 1 {
				params[i].Known = true
				params[i].Value = params[0].Value + 1
			} else {
				params[i].Known = false
				params[i].Value = 0
			}
		}
	}
	cur.append(newArray(params))
	return p.parseInsideAction(cur)
}

// parseFilter scans filter inside array selection
func (p *Parser) parseFilter(cur *ListNode) error {
	p.pos += len("[?(")
	p.consumeText()
Loop:
	for {
		switch p.next() {
		case eof, '\n':
			return fmt.Errorf("unterminated filter")
		case ')':
			break Loop
		}
	}
	if p.next() != ']' {
		return fmt.Errorf("unclosed array expect ]")
	}
	reg := regexp.MustCompile(`^([^!<>=]+)([!<>=]+)(.+?)$`)
	text := p.consumeText()
	text = string(text[:len(text)-2])
	value := reg.FindStringSubmatch(text)
	if value == nil {
		parser, err := parseAction("text", text)
		if err != nil {
			return err
		}
		cur.append(newFilter(parser.Root, newList(), "exists"))
	} else {
		leftParser, err := parseAction("left", value[1])
		if err != nil {
			return err
		}
		rightParser, err := parseAction("right", value[3])
		if err != nil {
			return err
		}
		cur.append(newFilter(leftParser.Root, rightParser.Root, value[2]))
	}
	return p.parseInsideAction(cur)
}

// parseQuote unquotes string inside double quote
func (p *Parser) parseQuote(cur *ListNode) error {
Loop:
	for {
		switch p.next() {
		case eof, '\n':
			return fmt.Errorf("unterminated quoted string")
		case '"':
			break Loop
		}
	}
	value := p.consumeText()
	s, err := strconv.Unquote(value)
	if err != nil {
		return fmt.Errorf("unquote string %s error %v", value, err)
	}
	cur.append(newText(s))
	return p.parseInsideAction(cur)
}

// parseField scans a field until a terminator
func (p *Parser) parseField(cur *ListNode) error {
	p.consumeText()
	for p.advance() {
	}
	value := p.consumeText()
	if value == "*" {
		cur.append(newWildcard())
	} else {
		cur.append(newField(strings.Replace(value, "\\", "", -1)))
	}
	return p.parseInsideAction(cur)
}

// advance scans until next non-escaped terminator
func (p *Parser) advance() bool {
	r := p.next()
	if r == '\\' {
		p.next()
	} else if isTerminator(r) {
		p.backup()
		return false
	}
	return true
}

// isTerminator reports whether the input is at valid termination character to appear after an identifier.
func isTerminator(r rune) bool {
	if isSpace(r) || isEndOfLine(r) {
		return true
	}
	switch r {
	case eof, '.', ',', '[', ']', '$', '@', '{', '}':
		return true
	}
	return false
}

// isSpace reports whether r is a space character.
func isSpace(r rune) bool {
	return r == ' ' || r == '\t'
}

// isEndOfLine reports whether r is an end-of-line character.
func isEndOfLine(r rune) bool {
	return r == '\r' || r == '\n'
}

// isAlphaNumeric reports whether r is an alphabetic, digit, or underscore.
func isAlphaNumeric(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}
