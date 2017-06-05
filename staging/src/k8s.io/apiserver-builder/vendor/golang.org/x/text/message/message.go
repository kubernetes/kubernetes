// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package message implements formatted I/O for localized strings with functions
// analogous to the fmt's print functions.
//
// NOTE: Under construction. See https://golang.org/design/12750-localization
// and its corresponding proposal issue https://golang.org/issues/12750.
package message // import "golang.org/x/text/message"

import (
	"fmt"
	"io"
	"strings"

	"golang.org/x/text/internal/format"
	"golang.org/x/text/language"
)

// A Printer implements language-specific formatted I/O analogous to the fmt
// package. Only one goroutine may use a Printer at the same time.
type Printer struct {
	tag language.Tag

	cat *Catalog

	// NOTE: limiting one goroutine per Printer allows for many optimizations
	// and simplifications. We can consider removing this restriction down the
	// road if it the benefits do not seem to outweigh the disadvantages.
}

// NewPrinter returns a Printer that formats messages tailored to language t.
func NewPrinter(t language.Tag) *Printer {
	return DefaultCatalog.Printer(t)
}

// Sprint is like fmt.Sprint, but using language-specific formatting.
func (p *Printer) Sprint(a ...interface{}) string {
	return fmt.Sprint(p.bindArgs(a)...)
}

// Fprint is like fmt.Fprint, but using language-specific formatting.
func (p *Printer) Fprint(w io.Writer, a ...interface{}) (n int, err error) {
	return fmt.Fprint(w, p.bindArgs(a)...)
}

// Print is like fmt.Print, but using language-specific formatting.
func (p *Printer) Print(a ...interface{}) (n int, err error) {
	return fmt.Print(p.bindArgs(a)...)
}

// Sprintln is like fmt.Sprintln, but using language-specific formatting.
func (p *Printer) Sprintln(a ...interface{}) string {
	return fmt.Sprintln(p.bindArgs(a)...)
}

// Fprintln is like fmt.Fprintln, but using language-specific formatting.
func (p *Printer) Fprintln(w io.Writer, a ...interface{}) (n int, err error) {
	return fmt.Fprintln(w, p.bindArgs(a)...)
}

// Println is like fmt.Println, but using language-specific formatting.
func (p *Printer) Println(a ...interface{}) (n int, err error) {
	return fmt.Println(p.bindArgs(a)...)
}

// Sprintf is like fmt.Sprintf, but using language-specific formatting.
func (p *Printer) Sprintf(key Reference, a ...interface{}) string {
	msg, hasSub := p.lookup(key)
	if !hasSub {
		return fmt.Sprintf(msg) // work around limitation of fmt
	}
	return fmt.Sprintf(msg, p.bindArgs(a)...)
}

// Fprintf is like fmt.Fprintf, but using language-specific formatting.
func (p *Printer) Fprintf(w io.Writer, key Reference, a ...interface{}) (n int, err error) {
	msg, hasSub := p.lookup(key)
	if !hasSub {
		return fmt.Fprintf(w, msg) // work around limitation of fmt
	}
	return fmt.Fprintf(w, msg, p.bindArgs(a)...)
}

// Printf is like fmt.Printf, but using language-specific formatting.
func (p *Printer) Printf(key Reference, a ...interface{}) (n int, err error) {
	msg, hasSub := p.lookup(key)
	if !hasSub {
		return fmt.Printf(msg) // work around limitation of fmt
	}
	return fmt.Printf(msg, p.bindArgs(a)...)
}

func (p *Printer) lookup(r Reference) (msg string, hasSub bool) {
	var id string
	switch v := r.(type) {
	case string:
		id, msg = v, v
	case key:
		id, msg = v.id, v.fallback
	default:
		panic("key argument is not a Reference")
	}
	if s, ok := p.cat.get(p.tag, id); ok {
		msg = s
	}
	// fmt does not allow all arguments to be dropped in a format string. It
	// only allows arguments to be dropped if at least one of the substitutions
	// uses the positional marker (e.g. %[1]s). This hack works around this.
	// TODO: This is only an approximation of the parsing of substitution
	// patterns. Make more precise once we know if we can get by with fmt's
	// formatting, which may not be the case.
	for i := 0; i < len(msg)-1; i++ {
		if msg[i] == '%' {
			for i++; i < len(msg); i++ {
				if strings.IndexByte("[]#+- *01234567890.", msg[i]) < 0 {
					break
				}
			}
			if i < len(msg) && msg[i] != '%' {
				hasSub = true
				break
			}
		}
	}
	return msg, hasSub
}

// A Reference is a string or a message reference.
type Reference interface {
}

// Key creates a message Reference for a message where the given id is used for
// message lookup and the fallback is returned when no matches are found.
func Key(id string, fallback string) Reference {
	return key{id, fallback}
}

type key struct {
	id, fallback string
}

// bindArgs wraps arguments with implementation of fmt.Formatter, if needed.
func (p *Printer) bindArgs(a []interface{}) []interface{} {
	out := make([]interface{}, len(a))
	for i, x := range a {
		switch v := x.(type) {
		case fmt.Formatter:
			// Wrap the value with a Formatter that augments the State with
			// language-specific attributes.
			out[i] = &value{v, p}

			// NOTE: as we use fmt.Formatter, we can't distinguish between
			// regular and localized formatters, so we always need to wrap it.

			// TODO: handle
			// - numbers
			// - lists
			// - time?
		default:
			out[i] = x
		}
	}
	return out
}

// state implements "golang.org/x/text/internal/format".State.
type state struct {
	fmt.State
	p *Printer
}

func (s *state) Language() language.Tag { return s.p.tag }

var _ format.State = &state{}

type value struct {
	x fmt.Formatter
	p *Printer
}

func (v *value) Format(s fmt.State, verb rune) {
	v.x.Format(&state{s, v.p}, verb)
}
