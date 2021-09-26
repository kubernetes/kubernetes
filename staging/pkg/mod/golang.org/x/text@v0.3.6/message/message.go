// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message // import "golang.org/x/text/message"

import (
	"io"
	"os"

	// Include features to facilitate generated catalogs.
	_ "golang.org/x/text/feature/plural"

	"golang.org/x/text/internal/number"
	"golang.org/x/text/language"
	"golang.org/x/text/message/catalog"
)

// A Printer implements language-specific formatted I/O analogous to the fmt
// package.
type Printer struct {
	// the language
	tag language.Tag

	toDecimal    number.Formatter
	toScientific number.Formatter

	cat catalog.Catalog
}

type options struct {
	cat catalog.Catalog
	// TODO:
	// - allow %s to print integers in written form (tables are likely too large
	//   to enable this by default).
	// - list behavior
	//
}

// An Option defines an option of a Printer.
type Option func(o *options)

// Catalog defines the catalog to be used.
func Catalog(c catalog.Catalog) Option {
	return func(o *options) { o.cat = c }
}

// NewPrinter returns a Printer that formats messages tailored to language t.
func NewPrinter(t language.Tag, opts ...Option) *Printer {
	options := &options{
		cat: DefaultCatalog,
	}
	for _, o := range opts {
		o(options)
	}
	p := &Printer{
		tag: t,
		cat: options.cat,
	}
	p.toDecimal.InitDecimal(t)
	p.toScientific.InitScientific(t)
	return p
}

// Sprint is like fmt.Sprint, but using language-specific formatting.
func (p *Printer) Sprint(a ...interface{}) string {
	pp := newPrinter(p)
	pp.doPrint(a)
	s := pp.String()
	pp.free()
	return s
}

// Fprint is like fmt.Fprint, but using language-specific formatting.
func (p *Printer) Fprint(w io.Writer, a ...interface{}) (n int, err error) {
	pp := newPrinter(p)
	pp.doPrint(a)
	n64, err := io.Copy(w, &pp.Buffer)
	pp.free()
	return int(n64), err
}

// Print is like fmt.Print, but using language-specific formatting.
func (p *Printer) Print(a ...interface{}) (n int, err error) {
	return p.Fprint(os.Stdout, a...)
}

// Sprintln is like fmt.Sprintln, but using language-specific formatting.
func (p *Printer) Sprintln(a ...interface{}) string {
	pp := newPrinter(p)
	pp.doPrintln(a)
	s := pp.String()
	pp.free()
	return s
}

// Fprintln is like fmt.Fprintln, but using language-specific formatting.
func (p *Printer) Fprintln(w io.Writer, a ...interface{}) (n int, err error) {
	pp := newPrinter(p)
	pp.doPrintln(a)
	n64, err := io.Copy(w, &pp.Buffer)
	pp.free()
	return int(n64), err
}

// Println is like fmt.Println, but using language-specific formatting.
func (p *Printer) Println(a ...interface{}) (n int, err error) {
	return p.Fprintln(os.Stdout, a...)
}

// Sprintf is like fmt.Sprintf, but using language-specific formatting.
func (p *Printer) Sprintf(key Reference, a ...interface{}) string {
	pp := newPrinter(p)
	lookupAndFormat(pp, key, a)
	s := pp.String()
	pp.free()
	return s
}

// Fprintf is like fmt.Fprintf, but using language-specific formatting.
func (p *Printer) Fprintf(w io.Writer, key Reference, a ...interface{}) (n int, err error) {
	pp := newPrinter(p)
	lookupAndFormat(pp, key, a)
	n, err = w.Write(pp.Bytes())
	pp.free()
	return n, err

}

// Printf is like fmt.Printf, but using language-specific formatting.
func (p *Printer) Printf(key Reference, a ...interface{}) (n int, err error) {
	pp := newPrinter(p)
	lookupAndFormat(pp, key, a)
	n, err = os.Stdout.Write(pp.Bytes())
	pp.free()
	return n, err
}

func lookupAndFormat(p *printer, r Reference, a []interface{}) {
	p.fmt.Reset(a)
	var id, msg string
	switch v := r.(type) {
	case string:
		id, msg = v, v
	case key:
		id, msg = v.id, v.fallback
	default:
		panic("key argument is not a Reference")
	}

	if p.catContext.Execute(id) == catalog.ErrNotFound {
		if p.catContext.Execute(msg) == catalog.ErrNotFound {
			p.Render(msg)
			return
		}
	}
}

type rawPrinter struct {
	p *printer
}

func (p rawPrinter) Render(msg string)     { p.p.WriteString(msg) }
func (p rawPrinter) Arg(i int) interface{} { return nil }

// Arg implements catmsg.Renderer.
func (p *printer) Arg(i int) interface{} { // TODO, also return "ok" bool
	i--
	if uint(i) < uint(len(p.fmt.Args)) {
		return p.fmt.Args[i]
	}
	return nil
}

// Render implements catmsg.Renderer.
func (p *printer) Render(msg string) {
	p.doPrintf(msg)
}

// A Reference is a string or a message reference.
type Reference interface {
	// TODO: also allow []string
}

// Key creates a message Reference for a message where the given id is used for
// message lookup and the fallback is returned when no matches are found.
func Key(id string, fallback string) Reference {
	return key{id, fallback}
}

type key struct {
	id, fallback string
}
