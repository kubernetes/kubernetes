// Copyright 2013 Joshua Tacoma. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package uritemplates is a level 3 implementation of RFC 6570 (URI
// Template, http://tools.ietf.org/html/rfc6570).
// uritemplates does not support composite values (in Go: slices or maps)
// and so does not qualify as a level 4 implementation.
package uritemplates

import (
	"bytes"
	"errors"
	"regexp"
	"strconv"
	"strings"
)

var (
	unreserved = regexp.MustCompile("[^A-Za-z0-9\\-._~]")
	reserved   = regexp.MustCompile("[^A-Za-z0-9\\-._~:/?#[\\]@!$&'()*+,;=]")
	validname  = regexp.MustCompile("^([A-Za-z0-9_\\.]|%[0-9A-Fa-f][0-9A-Fa-f])+$")
	hex        = []byte("0123456789ABCDEF")
)

func pctEncode(src []byte) []byte {
	dst := make([]byte, len(src)*3)
	for i, b := range src {
		buf := dst[i*3 : i*3+3]
		buf[0] = 0x25
		buf[1] = hex[b/16]
		buf[2] = hex[b%16]
	}
	return dst
}

func escape(s string, allowReserved bool) string {
	if allowReserved {
		return string(reserved.ReplaceAllFunc([]byte(s), pctEncode))
	}
	return string(unreserved.ReplaceAllFunc([]byte(s), pctEncode))
}

// A uriTemplate is a parsed representation of a URI template.
type uriTemplate struct {
	raw   string
	parts []templatePart
}

// parse parses a URI template string into a uriTemplate object.
func parse(rawTemplate string) (*uriTemplate, error) {
	split := strings.Split(rawTemplate, "{")
	parts := make([]templatePart, len(split)*2-1)
	for i, s := range split {
		if i == 0 {
			if strings.Contains(s, "}") {
				return nil, errors.New("unexpected }")
			}
			parts[i].raw = s
			continue
		}
		subsplit := strings.Split(s, "}")
		if len(subsplit) != 2 {
			return nil, errors.New("malformed template")
		}
		expression := subsplit[0]
		var err error
		parts[i*2-1], err = parseExpression(expression)
		if err != nil {
			return nil, err
		}
		parts[i*2].raw = subsplit[1]
	}
	return &uriTemplate{
		raw:   rawTemplate,
		parts: parts,
	}, nil
}

type templatePart struct {
	raw           string
	terms         []templateTerm
	first         string
	sep           string
	named         bool
	ifemp         string
	allowReserved bool
}

type templateTerm struct {
	name     string
	explode  bool
	truncate int
}

func parseExpression(expression string) (result templatePart, err error) {
	switch expression[0] {
	case '+':
		result.sep = ","
		result.allowReserved = true
		expression = expression[1:]
	case '.':
		result.first = "."
		result.sep = "."
		expression = expression[1:]
	case '/':
		result.first = "/"
		result.sep = "/"
		expression = expression[1:]
	case ';':
		result.first = ";"
		result.sep = ";"
		result.named = true
		expression = expression[1:]
	case '?':
		result.first = "?"
		result.sep = "&"
		result.named = true
		result.ifemp = "="
		expression = expression[1:]
	case '&':
		result.first = "&"
		result.sep = "&"
		result.named = true
		result.ifemp = "="
		expression = expression[1:]
	case '#':
		result.first = "#"
		result.sep = ","
		result.allowReserved = true
		expression = expression[1:]
	default:
		result.sep = ","
	}
	rawterms := strings.Split(expression, ",")
	result.terms = make([]templateTerm, len(rawterms))
	for i, raw := range rawterms {
		result.terms[i], err = parseTerm(raw)
		if err != nil {
			break
		}
	}
	return result, err
}

func parseTerm(term string) (result templateTerm, err error) {
	// TODO(djd): Remove "*" suffix parsing once we check that no APIs have
	// mistakenly used that attribute.
	if strings.HasSuffix(term, "*") {
		result.explode = true
		term = term[:len(term)-1]
	}
	split := strings.Split(term, ":")
	if len(split) == 1 {
		result.name = term
	} else if len(split) == 2 {
		result.name = split[0]
		var parsed int64
		parsed, err = strconv.ParseInt(split[1], 10, 0)
		result.truncate = int(parsed)
	} else {
		err = errors.New("multiple colons in same term")
	}
	if !validname.MatchString(result.name) {
		err = errors.New("not a valid name: " + result.name)
	}
	if result.explode && result.truncate > 0 {
		err = errors.New("both explode and prefix modifers on same term")
	}
	return result, err
}

// Expand expands a URI template with a set of values to produce a string.
func (t *uriTemplate) Expand(values map[string]string) string {
	var buf bytes.Buffer
	for _, p := range t.parts {
		p.expand(&buf, values)
	}
	return buf.String()
}

func (tp *templatePart) expand(buf *bytes.Buffer, values map[string]string) {
	if len(tp.raw) > 0 {
		buf.WriteString(tp.raw)
		return
	}
	var first = true
	for _, term := range tp.terms {
		value, exists := values[term.name]
		if !exists {
			continue
		}
		if first {
			buf.WriteString(tp.first)
			first = false
		} else {
			buf.WriteString(tp.sep)
		}
		tp.expandString(buf, term, value)
	}
}

func (tp *templatePart) expandName(buf *bytes.Buffer, name string, empty bool) {
	if tp.named {
		buf.WriteString(name)
		if empty {
			buf.WriteString(tp.ifemp)
		} else {
			buf.WriteString("=")
		}
	}
}

func (tp *templatePart) expandString(buf *bytes.Buffer, t templateTerm, s string) {
	if len(s) > t.truncate && t.truncate > 0 {
		s = s[:t.truncate]
	}
	tp.expandName(buf, t.name, len(s) == 0)
	buf.WriteString(escape(s, tp.allowReserved))
}
