/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	leftDelim  = "${"
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
	return p, p.Parse(text)
}

func NewParser(name string) *Parser {
	return &Parser{
		Name: name,
	}
}

func (p *Parser) Parse(text string) error {
	p.input = text
	p.Root = newList()
	p.pos = 0
	return p.parseText(p.Root)
}

func (p *Parser) consumeText() string {
	value := p.input[p.start:p.pos]
	p.start = p.pos
	return value
}

// next returns the next rune in the input.
func (p *Parser) next() rune {
	if int(p.pos) >= len(p.input) {
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
	if strings.HasPrefix(p.input[p.pos:], rightDelim) {
		return p.parseRightDelim(cur)
	}
	if strings.HasPrefix(p.input[p.pos:], "[?(") {
		p.pos += len("[?(")
		p.consumeText()
		return p.parseFilter(cur)
	}

	switch r := p.next(); {
	case r == eof || isEndOfLine(r):
		return fmt.Errorf("unclosed action")
	case r == '[':
		return p.parseArray(cur)
	case r == '"':
		return p.parseQuote(cur)
	case r == '.':
		return p.parseField(cur)
	default:
		return fmt.Errorf("unrecognized charactor in action: %#U", r)
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

// parseArray scans array index selection
func (p *Parser) parseArray(cur *ListNode) error {
Loop:
	for {
		switch p.next() {
		case eof, '\n':
			return fmt.Errorf("unterminated array string")
		case ']':
			break Loop
		}
	}
	text := p.consumeText()
	text = string(text[1 : len(text)-1])
	reg := regexp.MustCompile(`^(-?[\d]*)(:-?[\d]*)?(:[\d]*)?$`)
	value := reg.FindStringSubmatch(text)
	if value == nil {
		return fmt.Errorf("incorrect array index")
	}
	params := [3]int{}
	exist := [3]bool{true, true, true}
	value = value[1:]
	params[0], _ = strconv.Atoi(value[0])
	for i := 1; i < 3; i++ {
		if value[i] != "" {
			value[i] = value[i][1:]
			params[i], _ = strconv.Atoi(value[i])
		} else {
			exist[i] = false
			params[i] = 0
		}
	}
	cur.append(newArray(params, exist))
	return p.parseInsideAction(cur)
}

// parseFilter scans filter inside array selection
func (p *Parser) parseFilter(cur *ListNode) error {
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
		cur.append(newFilter(text, "exist", ""))
	} else {
		cur.append(newFilter(value[1], value[2], value[3]))
	}
	return p.parseInsideAction(cur)
}

// parseQuote scans array index selection
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
	cur.append(newText(value[1 : len(value)-1]))
	return p.parseInsideAction(cur)
}

// parseField scans a field: .Alphanumeric.
func (p *Parser) parseField(cur *ListNode) error {
	if r := p.peek(); isTerminator(r) {
		newNode := newText(p.consumeText())
		cur.append(newNode)
		return p.parseInsideAction(cur)
	}
	var r rune
	for {
		r = p.next()
		if !isAlphaNumeric(r) {
			p.backup()
			break
		}
	}
	if r := p.peek(); !isTerminator(r) {
		return fmt.Errorf("bad character %#U", r)
	}
	value := p.consumeText()
	newNode := newField(value[1:])
	cur.append(newNode)
	return p.parseInsideAction(cur)
}

// isTerminator reports whether the input is at valid termination character to appear after an identifier.
func isTerminator(r rune) bool {
	if isSpace(r) || isEndOfLine(r) {
		return true
	}
	switch r {
	case eof, '.', ',', '|', ':', ')', '(':
		return true
	}
	// Does r start the delimiter? This can be ambiguous (with delim=="//", $x/2 will
	// succeed but should fail) but only in extremely rare cases caused by willfully
	// bad choice of delimiter.
	if rd, _ := utf8.DecodeRuneInString(rightDelim); rd == r {
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
