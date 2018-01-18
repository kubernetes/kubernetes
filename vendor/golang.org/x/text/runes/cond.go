// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runes

import (
	"unicode/utf8"

	"golang.org/x/text/transform"
)

// Note: below we pass invalid UTF-8 to the tIn and tNotIn transformers as is.
// This is done for various reasons:
// - To retain the semantics of the Nop transformer: if input is passed to a Nop
//   one would expect it to be unchanged.
// - It would be very expensive to pass a converted RuneError to a transformer:
//   a transformer might need more source bytes after RuneError, meaning that
//   the only way to pass it safely is to create a new buffer and manage the
//   intermingling of RuneErrors and normal input.
// - Many transformers leave ill-formed UTF-8 as is, so this is not
//   inconsistent. Generally ill-formed UTF-8 is only replaced if it is a
//   logical consequence of the operation (as for Map) or if it otherwise would
//   pose security concerns (as for Remove).
// - An alternative would be to return an error on ill-formed UTF-8, but this
//   would be inconsistent with other operations.

// If returns a transformer that applies tIn to consecutive runes for which
// s.Contains(r) and tNotIn to consecutive runes for which !s.Contains(r). Reset
// is called on tIn and tNotIn at the start of each run. A Nop transformer will
// substitute a nil value passed to tIn or tNotIn. Invalid UTF-8 is translated
// to RuneError to determine which transformer to apply, but is passed as is to
// the respective transformer.
func If(s Set, tIn, tNotIn transform.Transformer) Transformer {
	if tIn == nil && tNotIn == nil {
		return Transformer{transform.Nop}
	}
	if tIn == nil {
		tIn = transform.Nop
	}
	if tNotIn == nil {
		tNotIn = transform.Nop
	}
	sIn, ok := tIn.(transform.SpanningTransformer)
	if !ok {
		sIn = dummySpan{tIn}
	}
	sNotIn, ok := tNotIn.(transform.SpanningTransformer)
	if !ok {
		sNotIn = dummySpan{tNotIn}
	}

	a := &cond{
		tIn:    sIn,
		tNotIn: sNotIn,
		f:      s.Contains,
	}
	a.Reset()
	return Transformer{a}
}

type dummySpan struct{ transform.Transformer }

func (d dummySpan) Span(src []byte, atEOF bool) (n int, err error) {
	return 0, transform.ErrEndOfSpan
}

type cond struct {
	tIn, tNotIn transform.SpanningTransformer
	f           func(rune) bool
	check       func(rune) bool               // current check to perform
	t           transform.SpanningTransformer // current transformer to use
}

// Reset implements transform.Transformer.
func (t *cond) Reset() {
	t.check = t.is
	t.t = t.tIn
	t.t.Reset() // notIn will be reset on first usage.
}

func (t *cond) is(r rune) bool {
	if t.f(r) {
		return true
	}
	t.check = t.isNot
	t.t = t.tNotIn
	t.tNotIn.Reset()
	return false
}

func (t *cond) isNot(r rune) bool {
	if !t.f(r) {
		return true
	}
	t.check = t.is
	t.t = t.tIn
	t.tIn.Reset()
	return false
}

// This implementation of Span doesn't help all too much, but it needs to be
// there to satisfy this package's Transformer interface.
// TODO: there are certainly room for improvements, though. For example, if
// t.t == transform.Nop (which will a common occurrence) it will save a bundle
// to special-case that loop.
func (t *cond) Span(src []byte, atEOF bool) (n int, err error) {
	p := 0
	for n < len(src) && err == nil {
		// Don't process too much at a time as the Spanner that will be
		// called on this block may terminate early.
		const maxChunk = 4096
		max := len(src)
		if v := n + maxChunk; v < max {
			max = v
		}
		atEnd := false
		size := 0
		current := t.t
		for ; p < max; p += size {
			r := rune(src[p])
			if r < utf8.RuneSelf {
				size = 1
			} else if r, size = utf8.DecodeRune(src[p:]); size == 1 {
				if !atEOF && !utf8.FullRune(src[p:]) {
					err = transform.ErrShortSrc
					break
				}
			}
			if !t.check(r) {
				// The next rune will be the start of a new run.
				atEnd = true
				break
			}
		}
		n2, err2 := current.Span(src[n:p], atEnd || (atEOF && p == len(src)))
		n += n2
		if err2 != nil {
			return n, err2
		}
		// At this point either err != nil or t.check will pass for the rune at p.
		p = n + size
	}
	return n, err
}

func (t *cond) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	p := 0
	for nSrc < len(src) && err == nil {
		// Don't process too much at a time, as the work might be wasted if the
		// destination buffer isn't large enough to hold the result or a
		// transform returns an error early.
		const maxChunk = 4096
		max := len(src)
		if n := nSrc + maxChunk; n < len(src) {
			max = n
		}
		atEnd := false
		size := 0
		current := t.t
		for ; p < max; p += size {
			r := rune(src[p])
			if r < utf8.RuneSelf {
				size = 1
			} else if r, size = utf8.DecodeRune(src[p:]); size == 1 {
				if !atEOF && !utf8.FullRune(src[p:]) {
					err = transform.ErrShortSrc
					break
				}
			}
			if !t.check(r) {
				// The next rune will be the start of a new run.
				atEnd = true
				break
			}
		}
		nDst2, nSrc2, err2 := current.Transform(dst[nDst:], src[nSrc:p], atEnd || (atEOF && p == len(src)))
		nDst += nDst2
		nSrc += nSrc2
		if err2 != nil {
			return nDst, nSrc, err2
		}
		// At this point either err != nil or t.check will pass for the rune at p.
		p = nSrc + size
	}
	return nDst, nSrc, err
}
