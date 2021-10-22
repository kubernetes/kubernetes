// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webdav

// The If header is covered by Section 10.4.
// http://www.webdav.org/specs/rfc4918.html#HEADER_If

import (
	"strings"
)

// ifHeader is a disjunction (OR) of ifLists.
type ifHeader struct {
	lists []ifList
}

// ifList is a conjunction (AND) of Conditions, and an optional resource tag.
type ifList struct {
	resourceTag string
	conditions  []Condition
}

// parseIfHeader parses the "If: foo bar" HTTP header. The httpHeader string
// should omit the "If:" prefix and have any "\r\n"s collapsed to a " ", as is
// returned by req.Header.Get("If") for a http.Request req.
func parseIfHeader(httpHeader string) (h ifHeader, ok bool) {
	s := strings.TrimSpace(httpHeader)
	switch tokenType, _, _ := lex(s); tokenType {
	case '(':
		return parseNoTagLists(s)
	case angleTokenType:
		return parseTaggedLists(s)
	default:
		return ifHeader{}, false
	}
}

func parseNoTagLists(s string) (h ifHeader, ok bool) {
	for {
		l, remaining, ok := parseList(s)
		if !ok {
			return ifHeader{}, false
		}
		h.lists = append(h.lists, l)
		if remaining == "" {
			return h, true
		}
		s = remaining
	}
}

func parseTaggedLists(s string) (h ifHeader, ok bool) {
	resourceTag, n := "", 0
	for first := true; ; first = false {
		tokenType, tokenStr, remaining := lex(s)
		switch tokenType {
		case angleTokenType:
			if !first && n == 0 {
				return ifHeader{}, false
			}
			resourceTag, n = tokenStr, 0
			s = remaining
		case '(':
			n++
			l, remaining, ok := parseList(s)
			if !ok {
				return ifHeader{}, false
			}
			l.resourceTag = resourceTag
			h.lists = append(h.lists, l)
			if remaining == "" {
				return h, true
			}
			s = remaining
		default:
			return ifHeader{}, false
		}
	}
}

func parseList(s string) (l ifList, remaining string, ok bool) {
	tokenType, _, s := lex(s)
	if tokenType != '(' {
		return ifList{}, "", false
	}
	for {
		tokenType, _, remaining = lex(s)
		if tokenType == ')' {
			if len(l.conditions) == 0 {
				return ifList{}, "", false
			}
			return l, remaining, true
		}
		c, remaining, ok := parseCondition(s)
		if !ok {
			return ifList{}, "", false
		}
		l.conditions = append(l.conditions, c)
		s = remaining
	}
}

func parseCondition(s string) (c Condition, remaining string, ok bool) {
	tokenType, tokenStr, s := lex(s)
	if tokenType == notTokenType {
		c.Not = true
		tokenType, tokenStr, s = lex(s)
	}
	switch tokenType {
	case strTokenType, angleTokenType:
		c.Token = tokenStr
	case squareTokenType:
		c.ETag = tokenStr
	default:
		return Condition{}, "", false
	}
	return c, s, true
}

// Single-rune tokens like '(' or ')' have a token type equal to their rune.
// All other tokens have a negative token type.
const (
	errTokenType    = rune(-1)
	eofTokenType    = rune(-2)
	strTokenType    = rune(-3)
	notTokenType    = rune(-4)
	angleTokenType  = rune(-5)
	squareTokenType = rune(-6)
)

func lex(s string) (tokenType rune, tokenStr string, remaining string) {
	// The net/textproto Reader that parses the HTTP header will collapse
	// Linear White Space that spans multiple "\r\n" lines to a single " ",
	// so we don't need to look for '\r' or '\n'.
	for len(s) > 0 && (s[0] == '\t' || s[0] == ' ') {
		s = s[1:]
	}
	if len(s) == 0 {
		return eofTokenType, "", ""
	}
	i := 0
loop:
	for ; i < len(s); i++ {
		switch s[i] {
		case '\t', ' ', '(', ')', '<', '>', '[', ']':
			break loop
		}
	}

	if i != 0 {
		tokenStr, remaining = s[:i], s[i:]
		if tokenStr == "Not" {
			return notTokenType, "", remaining
		}
		return strTokenType, tokenStr, remaining
	}

	j := 0
	switch s[0] {
	case '<':
		j, tokenType = strings.IndexByte(s, '>'), angleTokenType
	case '[':
		j, tokenType = strings.IndexByte(s, ']'), squareTokenType
	default:
		return rune(s[0]), "", s[1:]
	}
	if j < 0 {
		return errTokenType, "", ""
	}
	return tokenType, s[1:j], s[j+1:]
}
