// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run ../collate/maketables.go -cldr=23 -unicode=6.2.0 -types=search,searchjl -package=search

// Package search provides language-specific search and string matching.
//
// Natural language matching can be intricate. For example, Danish will insist
// "Århus" and "Aarhus" are the same name and Turkish will match I to ı (note
// the lack of a dot) in a case-insensitive match. This package handles such
// language-specific details.
//
// Text passed to any of the calls in this message does not need to be
// normalized.
package search // import "golang.org/x/text/search"

import (
	"strings"

	"golang.org/x/text/collate/colltab"
	newcolltab "golang.org/x/text/internal/colltab"
	"golang.org/x/text/language"
)

// An Option configures a Matcher.
type Option func(*Matcher)

var (
	// WholeWord restricts matches to complete words. The default is to match at
	// the character level.
	WholeWord Option = nil

	// Exact requires that two strings are their exact equivalent. For example
	// å would not match aa in Danish. It overrides any of the ignore options.
	Exact Option = nil

	// Loose causes case, diacritics and width to be ignored.
	Loose Option = loose

	// IgnoreCase enables case-insensitive search.
	IgnoreCase Option = ignoreCase

	// IgnoreDiacritics causes diacritics to be ignored ("ö" == "o").
	IgnoreDiacritics Option = ignoreDiacritics

	// IgnoreWidth equates narrow with wide variants.
	IgnoreWidth Option = ignoreWidth
)

func ignoreDiacritics(m *Matcher) { m.ignoreDiacritics = true }
func ignoreCase(m *Matcher)       { m.ignoreCase = true }
func ignoreWidth(m *Matcher)      { m.ignoreWidth = true }
func loose(m *Matcher) {
	ignoreDiacritics(m)
	ignoreCase(m)
	ignoreWidth(m)
}

var (
	// Supported lists the languages for which search differs from its parent.
	Supported language.Coverage

	tags []language.Tag
)

func init() {
	ids := strings.Split(availableLocales, ",")
	tags = make([]language.Tag, len(ids))
	for i, s := range ids {
		tags[i] = language.Raw.MustParse(s)
	}
	Supported = language.NewCoverage(tags)
}

// New returns a new Matcher for the given language and options.
func New(t language.Tag, opts ...Option) *Matcher {
	m := &Matcher{
		w: colltab.Init(locales[newcolltab.MatchLang(t, tags)]),
	}
	for _, f := range opts {
		f(m)
	}
	return m
}

// A Matcher implements language-specific string matching.
type Matcher struct {
	w                colltab.Weighter
	ignoreCase       bool
	ignoreWidth      bool
	ignoreDiacritics bool
}

// An IndexOption specifies how the Index methods of Pattern or Matcher should
// match the input.
type IndexOption byte

const (
	// Anchor restricts the search to the start (or end for Backwards) of the
	// text.
	Anchor IndexOption = 1 << iota

	// Backwards starts the search from the end of the text.
	Backwards

	anchorBackwards = Anchor | Backwards
)

// Index reports the start and end position of the first occurrence of pat in b
// or -1, -1 if pat is not present.
func (m *Matcher) Index(b, pat []byte, opts ...IndexOption) (start, end int) {
	// TODO: implement optimized version that does not use a pattern.
	return m.Compile(pat).Index(b, opts...)
}

// IndexString reports the start and end position of the first occurrence of pat
// in s or -1, -1 if pat is not present.
func (m *Matcher) IndexString(s, pat string, opts ...IndexOption) (start, end int) {
	// TODO: implement optimized version that does not use a pattern.
	return m.CompileString(pat).IndexString(s, opts...)
}

// Equal reports whether a and b are equivalent.
func (m *Matcher) Equal(a, b []byte) bool {
	_, end := m.Index(a, b, Anchor)
	return end == len(a)
}

// EqualString reports whether a and b are equivalent.
func (m *Matcher) EqualString(a, b string) bool {
	_, end := m.IndexString(a, b, Anchor)
	return end == len(a)
}

// Compile compiles and returns a pattern that can be used for faster searching.
func (m *Matcher) Compile(b []byte) *Pattern {
	p := &Pattern{m: m}
	iter := newcolltab.Iter{Weighter: m.w}
	for iter.SetInput(b); iter.Next(); {
	}
	p.ce = iter.Elems
	p.deleteEmptyElements()
	return p
}

// CompileString compiles and returns a pattern that can be used for faster
// searching.
func (m *Matcher) CompileString(s string) *Pattern {
	p := &Pattern{m: m}
	iter := newcolltab.Iter{Weighter: m.w}
	for iter.SetInputString(s); iter.Next(); {
	}
	p.ce = iter.Elems
	p.deleteEmptyElements()
	return p
}

// A Pattern is a compiled search string. It is safe for concurrent use.
type Pattern struct {
	m  *Matcher
	ce []colltab.Elem
}

// Design note (TODO remove):
// The cost of retrieving collation elements for each rune, which is used for
// search as well, is not trivial. Also, algorithms like Boyer-Moore and
// Sunday require some additional precomputing.

// Index reports the start and end position of the first occurrence of p in b
// or -1, -1 if p is not present.
func (p *Pattern) Index(b []byte, opts ...IndexOption) (start, end int) {
	// Pick a large enough buffer such that we likely do not need to allocate
	// and small enough to not cause too much overhead initializing.
	var buf [8]colltab.Elem

	it := &newcolltab.Iter{
		Weighter: p.m.w,
		Elems:    buf[:0],
	}
	it.SetInput(b)

	var optMask IndexOption
	for _, o := range opts {
		optMask |= o
	}

	switch optMask {
	case 0:
		return p.forwardSearch(it)
	case Anchor:
		return p.anchoredForwardSearch(it)
	case Backwards, anchorBackwards:
		panic("TODO: implement")
	default:
		panic("unrecognized option")
	}
}

// IndexString reports the start and end position of the first occurrence of p
// in s or -1, -1 if p is not present.
func (p *Pattern) IndexString(s string, opts ...IndexOption) (start, end int) {
	// Pick a large enough buffer such that we likely do not need to allocate
	// and small enough to not cause too much overhead initializing.
	var buf [8]colltab.Elem

	it := &newcolltab.Iter{
		Weighter: p.m.w,
		Elems:    buf[:0],
	}
	it.SetInputString(s)

	var optMask IndexOption
	for _, o := range opts {
		optMask |= o
	}

	switch optMask {
	case 0:
		return p.forwardSearch(it)
	case Anchor:
		return p.anchoredForwardSearch(it)
	case Backwards, anchorBackwards:
		panic("TODO: implement")
	default:
		panic("unrecognized option")
	}
}

// TODO:
// - Maybe IndexAll methods (probably not necessary).
// - Some way to match patterns in a Reader (a bit tricky).
// - Some fold transformer that folds text to comparable text, based on the
//   search options. This is a common technique, though very different from the
//   collation-based design of this package. It has a somewhat different use
//   case, so probably makes sense to support both. Should probably be in a
//   different package, though, as it uses completely different kind of tables
//   (based on norm, cases, width and range tables.)
