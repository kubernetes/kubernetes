// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"errors"
	"unicode/utf8"

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

func (b *buffers) init(n int) {
	b.buf[0] = make([]byte, 0, n)
	b.buf[1] = make([]byte, 0, n)
}

func (b *buffers) apply(t transform.Transformer) (err error) {
	// TODO: use Span, once available.
	x := b.next & 1
	b.src, _, err = transform.Append(t, b.buf[x][:0], b.src)
	b.buf[x] = b.src
	b.next++
	return err
}

func (b *buffers) enforce(p *Profile, src []byte) (str []byte, err error) {
	b.src = src

	// These transforms are applied in the order defined in
	// https://tools.ietf.org/html/rfc7564#section-7

	// TODO: allow different width transforms options.
	if p.options.foldWidth {
		// TODO: use Span, once available.
		if err = b.apply(width.Fold); err != nil {
			return nil, err
		}
	}
	for _, f := range p.options.additional {
		if err = b.apply(f()); err != nil {
			return nil, err
		}
	}
	if p.options.cases != nil {
		if err = b.apply(p.options.cases); err != nil {
			return nil, err
		}
	}
	if n := p.norm.QuickSpan(b.src); n < len(b.src) {
		x := b.next & 1
		n = copy(b.buf[x], b.src[:n])
		b.src, _, err = transform.Append(p.norm, b.buf[x][:n], b.src[n:])
		b.buf[x] = b.src
		b.next++
		if err != nil {
			return nil, err
		}
	}
	if p.options.bidiRule {
		if err := b.apply(bidirule.New()); err != nil {
			return nil, err
		}
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

	// TODO: Add the disallow empty rule with a dummy transformer?

	if p.options.disallowEmpty && len(b.src) == 0 {
		return nil, errEmptyString
	}
	return b.src, nil
}

// Append appends the result of applying p to src writing the result to dst.
// It returns an error if the input string is invalid.
func (p *Profile) Append(dst, src []byte) ([]byte, error) {
	var buf buffers
	buf.init(8 + len(src) + len(src)>>2)
	b, err := buf.enforce(p, src)
	if err != nil {
		return nil, err
	}
	return append(dst, b...), nil
}

// Bytes returns a new byte slice with the result of applying the profile to b.
func (p *Profile) Bytes(b []byte) ([]byte, error) {
	var buf buffers
	buf.init(8 + len(b) + len(b)>>2)
	b, err := buf.enforce(p, b)
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

// String returns a string with the result of applying the profile to s.
func (p *Profile) String(s string) (string, error) {
	var buf buffers
	buf.init(8 + len(s) + len(s)>>2)
	b, err := buf.enforce(p, []byte(s))
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// Compare enforces both strings, and then compares them for bit-string identity
// (byte-for-byte equality). If either string cannot be enforced, the comparison
// is false.
func (p *Profile) Compare(a, b string) bool {
	a, err := p.String(a)
	if err != nil {
		return false
	}
	b, err = p.String(b)
	if err != nil {
		return false
	}

	// TODO: This is out of order. Need to extract the transformation logic and
	// put this in where the normal case folding would go (but only for
	// comparison).
	if p.options.ignorecase {
		a = width.Fold.String(a)
		b = width.Fold.String(a)
	}

	return a == b
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
		if property(e) < c.p.class.validFrom {
			if d.rule == nil {
				return n, errDisallowedRune
			}
			doLookAhead, err := d.rule(c.beforeBits)
			if err != nil {
				return n, err
			}
			if doLookAhead {
				c.beforeBits &= d.keep
				c.beforeBits |= d.set
				// We may still have a lookahead rule which we will require to
				// complete (by checking termBits == 0) before setting the new
				// bits.
				if c.termBits != 0 && (!c.checkLookahead() || c.termBits == 0) {
					return n, err
				}
				c.termBits = d.term
				c.acceptBits = d.accept
				n += sz
				continue
			}
		}
		c.beforeBits &= d.keep
		c.beforeBits |= d.set
		if c.termBits != 0 && !c.checkLookahead() {
			return n, errContext
		}
		n += sz
	}
	if m := c.beforeBits >> finalShift; c.beforeBits&m != m || c.termBits != 0 {
		err = errContext
	}
	return n, err
}

func (c *checker) checkLookahead() bool {
	switch {
	case c.beforeBits&c.termBits != 0:
		c.termBits = 0
		c.acceptBits = 0
	case c.beforeBits&c.acceptBits != 0:
	default:
		return false
	}
	return true
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
