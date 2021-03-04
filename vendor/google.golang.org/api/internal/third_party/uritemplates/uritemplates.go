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

// pairWriter is a convenience struct which allows escaped and unescaped
// versions of the template to be written in parallel.
type pairWriter struct {
	escaped, unescaped bytes.Buffer
}

// Write writes the provided string directly without any escaping.
func (w *pairWriter) Write(s string) {
	w.escaped.WriteString(s)
	w.unescaped.WriteString(s)
}

// Escape writes the provided string, escaping the string for the
// escaped output.
func (w *pairWriter) Escape(s string, allowReserved bool) {
	w.unescaped.WriteString(s)
	if allowReserved {
		w.escaped.Write(reserved.ReplaceAllFunc([]byte(s), pctEncode))
	} else {
		w.escaped.Write(unreserved.ReplaceAllFunc([]byte(s), pctEncode))
	}
}

// Escaped returns the escaped string.
func (w *pairWriter) Escaped() string {
	return w.escaped.String()
}

// Unescaped returns the unescaped string.
func (w *pairWriter) Unescaped() string {
	return w.unescaped.String()
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
		err = errors.New("both explode and prefix modifiers on same term")
	}
	return result, err
}

// Expand expands a URI template with a set of values to produce the
// resultant URI. Two forms of the result are returned: one with all the
// elements escaped, and one with the elements unescaped.
func (t *uriTemplate) Expand(values map[string]string) (escaped, unescaped string) {
	var w pairWriter
	for _, p := range t.parts {
		p.expand(&w, values)
	}
	return w.Escaped(), w.Unescaped()
}

func (tp *templatePart) expand(w *pairWriter, values map[string]string) {
	if len(tp.raw) > 0 {
		w.Write(tp.raw)
		return
	}
	var first = true
	for _, term := range tp.terms {
		value, exists := values[term.name]
		if !exists {
			continue
		}
		if first {
			w.Write(tp.first)
			first = false
		} else {
			w.Write(tp.sep)
		}
		tp.expandString(w, term, value)
	}
}

func (tp *templatePart) expandName(w *pairWriter, name string, empty bool) {
	if tp.named {
		w.Write(name)
		if empty {
			w.Write(tp.ifemp)
		} else {
			w.Write("=")
		}
	}
}

func (tp *templatePart) expandString(w *pairWriter, t templateTerm, s string) {
	if len(s) > t.truncate && t.truncate > 0 {
		s = s[:t.truncate]
	}
	tp.expandName(w, t.name, len(s) == 0)
	w.Escape(s, tp.allowReserved)
}
