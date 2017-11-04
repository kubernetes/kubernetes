// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"bytes"
	"errors"
	"unicode/utf8"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
	"golang.org/x/text/runes"
	"golang.org/x/text/secure/bidirule"
	"golang.org/x/text/transform"
	"golang.org/x/text/width"
)

var (
	errDisallowedRune = errors.New("precis: disallowed rune encountered")
)

var dpTrie = newDerivedPropertiesTrie(0)

// A Profile represents a set of rules for normalizing and validating strings in
// the PRECIS framework.
type Profile struct {
	options
	class *class
}

// NewIdentifier creates a new PRECIS profile based on the Identifier string
// class. Profiles created from this class are suitable for use where safety is
// prioritized over expressiveness like network identifiers, user accounts, chat
// rooms, and file names.
func NewIdentifier(opts ...Option) *Profile {
	return &Profile{
		options: getOpts(opts...),
		class:   identifier,
	}
}

// NewFreeform creates a new PRECIS profile based on the Freeform string class.
// Profiles created from this class are suitable for use where expressiveness is
// prioritized over safety like passwords, and display-elements such as
// nicknames in a chat room.
func NewFreeform(opts ...Option) *Profile {
	return &Profile{
		options: getOpts(opts...),
		class:   freeform,
	}
}

// NewTransformer creates a new transform.Transformer that performs the PRECIS
// preparation and enforcement steps on the given UTF-8 encoded bytes.
func (p *Profile) NewTransformer() *Transformer {
	var ts []transform.Transformer

	// These transforms are applied in the order defined in
	// https://tools.ietf.org/html/rfc7564#section-7

	if p.options.foldWidth {
		ts = append(ts, width.Fold)
	}

	for _, f := range p.options.additional {
		ts = append(ts, f())
	}

	if p.options.cases != nil {
		ts = append(ts, p.options.cases)
	}

	ts = append(ts, p.options.norm)

	if p.options.bidiRule {
		ts = append(ts, bidirule.New())
	}

	ts = append(ts, &checker{p: p, allowed: p.Allowed()})

	// TODO: Add the disallow empty rule with a dummy transformer?

	return &Transformer{transform.Chain(ts...)}
}

var errEmptyString = errors.New("precis: transformation resulted in empty string")

type buffers struct {
	src  []byte
	buf  [2][]byte
	next int
}

func (b *buffers) apply(t transform.SpanningTransformer) (err error) {
	n, err := t.Span(b.src, true)
	if err != transform.ErrEndOfSpan {
		return err
	}
	x := b.next & 1
	if b.buf[x] == nil {
		b.buf[x] = make([]byte, 0, 8+len(b.src)+len(b.src)>>2)
	}
	span := append(b.buf[x][:0], b.src[:n]...)
	b.src, _, err = transform.Append(t, span, b.src[n:])
	b.buf[x] = b.src
	b.next++
	return err
}

// Pre-allocate transformers when possible. In some cases this avoids allocation.
var (
	foldWidthT transform.SpanningTransformer = width.Fold
	lowerCaseT transform.SpanningTransformer = cases.Lower(language.Und, cases.HandleFinalSigma(false))
)

// TODO: make this a method on profile.

func (b *buffers) enforce(p *Profile, src []byte, comparing bool) (str []byte, err error) {
	b.src = src

	ascii := true
	for _, c := range src {
		if c >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	// ASCII fast path.
	if ascii {
		for _, f := range p.options.additional {
			if err = b.apply(f()); err != nil {
				return nil, err
			}
		}
		switch {
		case p.options.asciiLower || (comparing && p.options.ignorecase):
			for i, c := range b.src {
				if 'A' <= c && c <= 'Z' {
					b.src[i] = c ^ 1<<5
				}
			}
		case p.options.cases != nil:
			b.apply(p.options.cases)
		}
		c := checker{p: p}
		if _, err := c.span(b.src, true); err != nil {
			return nil, err
		}
		if p.disallow != nil {
			for _, c := range b.src {
				if p.disallow.Contains(rune(c)) {
					return nil, errDisallowedRune
				}
			}
		}
		if p.options.disallowEmpty && len(b.src) == 0 {
			return nil, errEmptyString
		}
		return b.src, nil
	}

	// These transforms are applied in the order defined in
	// https://tools.ietf.org/html/rfc7564#section-7

	// TODO: allow different width transforms options.
	if p.options.foldWidth || (p.options.ignorecase && comparing) {
		b.apply(foldWidthT)
	}
	for _, f := range p.options.additional {
		if err = b.apply(f()); err != nil {
			return nil, err
		}
	}
	if p.options.cases != nil {
		b.apply(p.options.cases)
	}
	if comparing && p.options.ignorecase {
		b.apply(lowerCaseT)
	}
	b.apply(p.norm)
	if p.options.bidiRule && !bidirule.Valid(b.src) {
		return nil, bidirule.ErrInvalid
	}
	c := checker{p: p}
	if _, err := c.span(b.src, true); err != nil {
		return nil, err
	}
	if p.disallow != nil {
		for i := 0; i < len(b.src); {
			r, size := utf8.DecodeRune(b.src[i:])
			if p.disallow.Contains(r) {
				return nil, errDisallowedRune
			}
			i += size
		}
	}
	if p.options.disallowEmpty && len(b.src) == 0 {
		return nil, errEmptyString
	}
	return b.src, nil
}

// Append appends the result of applying p to src writing the result to dst.
// It returns an error if the input string is invalid.
func (p *Profile) Append(dst, src []byte) ([]byte, error) {
	var buf buffers
	b, err := buf.enforce(p, src, false)
	if err != nil {
		return nil, err
	}
	return append(dst, b...), nil
}

func processBytes(p *Profile, b []byte, key bool) ([]byte, error) {
	var buf buffers
	b, err := buf.enforce(p, b, key)
	if err != nil {
		return nil, err
	}
	if buf.next == 0 {
		c := make([]byte, len(b))
		copy(c, b)
		return c, nil
	}
	return b, nil
}

// Bytes returns a new byte slice with the result of applying the profile to b.
func (p *Profile) Bytes(b []byte) ([]byte, error) {
	return processBytes(p, b, false)
}

// AppendCompareKey appends the result of applying p to src (including any
// optional rules to make strings comparable or useful in a map key such as
// applying lowercasing) writing the result to dst. It returns an error if the
// input string is invalid.
func (p *Profile) AppendCompareKey(dst, src []byte) ([]byte, error) {
	var buf buffers
	b, err := buf.enforce(p, src, true)
	if err != nil {
		return nil, err
	}
	return append(dst, b...), nil
}

func processString(p *Profile, s string, key bool) (string, error) {
	var buf buffers
	b, err := buf.enforce(p, []byte(s), key)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// String returns a string with the result of applying the profile to s.
func (p *Profile) String(s string) (string, error) {
	return processString(p, s, false)
}

// CompareKey returns a string that can be used for comparison, hashing, or
// collation.
func (p *Profile) CompareKey(s string) (string, error) {
	return processString(p, s, true)
}

// Compare enforces both strings, and then compares them for bit-string identity
// (byte-for-byte equality). If either string cannot be enforced, the comparison
// is false.
func (p *Profile) Compare(a, b string) bool {
	var buf buffers

	akey, err := buf.enforce(p, []byte(a), true)
	if err != nil {
		return false
	}

	buf = buffers{}
	bkey, err := buf.enforce(p, []byte(b), true)
	if err != nil {
		return false
	}

	return bytes.Compare(akey, bkey) == 0
}

// Allowed returns a runes.Set containing every rune that is a member of the
// underlying profile's string class and not disallowed by any profile specific
// rules.
func (p *Profile) Allowed() runes.Set {
	if p.options.disallow != nil {
		return runes.Predicate(func(r rune) bool {
			return p.class.Contains(r) && !p.options.disallow.Contains(r)
		})
	}
	return p.class
}

type checker struct {
	p       *Profile
	allowed runes.Set

	beforeBits catBitmap
	termBits   catBitmap
	acceptBits catBitmap
}

func (c *checker) Reset() {
	c.beforeBits = 0
	c.termBits = 0
	c.acceptBits = 0
}

func (c *checker) span(src []byte, atEOF bool) (n int, err error) {
	for n < len(src) {
		e, sz := dpTrie.lookup(src[n:])
		d := categoryTransitions[category(e&catMask)]
		if sz == 0 {
			if !atEOF {
				return n, transform.ErrShortSrc
			}
			return n, errDisallowedRune
		}
		doLookAhead := false
		if property(e) < c.p.class.validFrom {
			if d.rule == nil {
				return n, errDisallowedRune
			}
			doLookAhead, err = d.rule(c.beforeBits)
			if err != nil {
				return n, err
			}
		}
		c.beforeBits &= d.keep
		c.beforeBits |= d.set
		if c.termBits != 0 {
			// We are currently in an unterminated lookahead.
			if c.beforeBits&c.termBits != 0 {
				c.termBits = 0
				c.acceptBits = 0
			} else if c.beforeBits&c.acceptBits == 0 {
				// Invalid continuation of the unterminated lookahead sequence.
				return n, errContext
			}
		}
		if doLookAhead {
			if c.termBits != 0 {
				// A previous lookahead run has not been terminated yet.
				return n, errContext
			}
			c.termBits = d.term
			c.acceptBits = d.accept
		}
		n += sz
	}
	if m := c.beforeBits >> finalShift; c.beforeBits&m != m || c.termBits != 0 {
		err = errContext
	}
	return n, err
}

// TODO: we may get rid of this transform if transform.Chain understands
// something like a Spanner interface.
func (c checker) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	short := false
	if len(dst) < len(src) {
		src = src[:len(dst)]
		atEOF = false
		short = true
	}
	nSrc, err = c.span(src, atEOF)
	nDst = copy(dst, src[:nSrc])
	if short && (err == transform.ErrShortSrc || err == nil) {
		err = transform.ErrShortDst
	}
	return nDst, nSrc, err
}
