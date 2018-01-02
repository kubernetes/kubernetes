// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"bytes"
	"testing"

	"golang.org/x/text/internal/colltab"
	"golang.org/x/text/language"
)

type weightsTest struct {
	opt     opts
	in, out ColElems
}

type opts struct {
	lev int
	alt alternateHandling
	top int

	backwards bool
	caseLevel bool
}

// ignore returns an initialized boolean array based on the given Level.
// A negative value means using the default setting of quaternary.
func ignore(level colltab.Level) (ignore [colltab.NumLevels]bool) {
	if level < 0 {
		level = colltab.Quaternary
	}
	for i := range ignore {
		ignore[i] = level < colltab.Level(i)
	}
	return ignore
}

func makeCE(w []int) colltab.Elem {
	ce, err := colltab.MakeElem(w[0], w[1], w[2], uint8(w[3]))
	if err != nil {
		panic(err)
	}
	return ce
}

func (o opts) collator() *Collator {
	c := &Collator{
		options: options{
			ignore:      ignore(colltab.Level(o.lev - 1)),
			alternate:   o.alt,
			backwards:   o.backwards,
			caseLevel:   o.caseLevel,
			variableTop: uint32(o.top),
		},
	}
	return c
}

const (
	maxQ = 0x1FFFFF
)

func wpq(p, q int) Weights {
	return W(p, defaults.Secondary, defaults.Tertiary, q)
}

func wsq(s, q int) Weights {
	return W(0, s, defaults.Tertiary, q)
}

func wq(q int) Weights {
	return W(0, 0, 0, q)
}

var zero = W(0, 0, 0, 0)

var processTests = []weightsTest{
	// Shifted
	{ // simple sequence of non-variables
		opt: opts{alt: altShifted, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: altShifted, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: altShifted, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: altShifted, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: altShifted, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: altShifted, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), W(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: altShifted, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// ShiftTrimmed (same as Shifted)
	{ // simple sequence of non-variables
		opt: opts{alt: altShiftTrimmed, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wpq(200, maxQ), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // first is a variable
		opt: opts{alt: altShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{wq(200), wpq(300, maxQ), wpq(400, maxQ)},
	},
	{ // all but first are variable
		opt: opts{alt: altShiftTrimmed, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{wpq(1000, maxQ), wq(200), wq(300), wq(400)},
	},
	{ // first is a modifier
		opt: opts{alt: altShiftTrimmed, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{wsq(10, maxQ), wpq(1000, maxQ)},
	},
	{ // primary ignorables
		opt: opts{alt: altShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), wsq(15, maxQ), wpq(400, maxQ)},
	},
	{ // secondary ignorables
		opt: opts{alt: altShiftTrimmed, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), W(0, 0, 15, maxQ), wpq(400, maxQ)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: altShiftTrimmed, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{wq(200), zero, wpq(300, maxQ), zero, wpq(400, maxQ)},
	},

	// Blanked
	{ // simple sequence of non-variables
		opt: opts{alt: altBlanked, top: 100},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{W(200), W(300), W(400)},
	},
	{ // first is a variable
		opt: opts{alt: altBlanked, top: 250},
		in:  ColElems{W(200), W(300), W(400)},
		out: ColElems{zero, W(300), W(400)},
	},
	{ // all but first are variable
		opt: opts{alt: altBlanked, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{W(1000), zero, zero, zero},
	},
	{ // first is a modifier
		opt: opts{alt: altBlanked, top: 999},
		in:  ColElems{W(0, 10), W(1000)},
		out: ColElems{W(0, 10), W(1000)},
	},
	{ // primary ignorables
		opt: opts{alt: altBlanked, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{zero, zero, W(300), W(0, 15), W(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: altBlanked, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{zero, zero, W(300), W(0, 0, 15), W(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: altBlanked, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{zero, zero, W(300), zero, W(400)},
	},

	// Non-ignorable: input is always equal to output.
	{ // all but first are variable
		opt: opts{alt: altNonIgnorable, top: 999},
		in:  ColElems{W(1000), W(200), W(300), W(400)},
		out: ColElems{W(1000), W(200), W(300), W(400)},
	},
	{ // primary ignorables
		opt: opts{alt: altNonIgnorable, top: 250},
		in:  ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
		out: ColElems{W(200), W(0, 10), W(300), W(0, 15), W(400)},
	},
	{ // secondary ignorables
		opt: opts{alt: altNonIgnorable, top: 250},
		in:  ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
		out: ColElems{W(200), W(0, 0, 10), W(300), W(0, 0, 15), W(400)},
	},
	{ // tertiary ignorables, no change
		opt: opts{alt: altNonIgnorable, top: 250},
		in:  ColElems{W(200), zero, W(300), zero, W(400)},
		out: ColElems{W(200), zero, W(300), zero, W(400)},
	},
}

func TestProcessWeights(t *testing.T) {
	for i, tt := range processTests {
		in := convertFromWeights(tt.in)
		out := convertFromWeights(tt.out)
		processWeights(tt.opt.alt, uint32(tt.opt.top), in)
		for j, w := range in {
			if w != out[j] {
				t.Errorf("%d: Weights %d was %v; want %v", i, j, w, out[j])
			}
		}
	}
}

type keyFromElemTest struct {
	opt opts
	in  ColElems
	out []byte
}

var defS = byte(defaults.Secondary)
var defT = byte(defaults.Tertiary)

const sep = 0 // separator byte

var keyFromElemTests = []keyFromElemTest{
	{ // simple primary and secondary weights.
		opts{alt: altShifted},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with zero element that need to be removed
		opts{alt: altShifted},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, with large primary values
		opts{alt: altShifted},
		ColElems{W(0x200), W(0x8000), W(0, 0x30), W(0x12345)},
		[]byte{0x2, 0, 0x80, 0x80, 0x00, 0x81, 0x23, 0x45, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, but with the secondary level backwards
		opts{alt: altShifted, backwards: true},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, 0x30, 0, defS, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0xFF, 0xFF, 0xFF, // quaternary
		},
	},
	{ // same as first, ignoring quaternary level
		opts{alt: altShifted, lev: 3},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
	{ // same as first, ignoring tertiary level
		opts{alt: altShifted, lev: 2},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
		},
	},
	{ // same as first, ignoring secondary level
		opts{alt: altShifted, lev: 1},
		ColElems{W(0x200), zero, W(0x7FFF), W(0, 0x30), zero, W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00},
	},
	{ // simple primary and secondary weights.
		opts{alt: altShiftTrimmed, top: 0x250},
		ColElems{W(0x300), W(0x200), W(0x7FFF), W(0, 0x30), W(0x800)},
		[]byte{0x3, 0, 0x7F, 0xFF, 0x8, 0x00, // primary
			sep, sep, 0, defS, 0, defS, 0, 0x30, 0, defS, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
			sep, 0xFF, 0x2, 0, // quaternary
		},
	},
	{ // as first, primary with case level enabled
		opts{alt: altShifted, lev: 1, caseLevel: true},
		ColElems{W(0x200), W(0x7FFF), W(0, 0x30), W(0x100)},
		[]byte{0x2, 0, 0x7F, 0xFF, 0x1, 0x00, // primary
			sep, sep, // secondary
			sep, sep, defT, defT, defT, defT, // tertiary
		},
	},
}

func TestKeyFromElems(t *testing.T) {
	buf := Buffer{}
	for i, tt := range keyFromElemTests {
		buf.Reset()
		in := convertFromWeights(tt.in)
		processWeights(tt.opt.alt, uint32(tt.opt.top), in)
		tt.opt.collator().keyFromElems(&buf, in)
		res := buf.key
		if len(res) != len(tt.out) {
			t.Errorf("%d: len(ws) was %d; want %d (%X should be %X)", i, len(res), len(tt.out), res, tt.out)
		}
		n := len(res)
		if len(tt.out) < n {
			n = len(tt.out)
		}
		for j, c := range res[:n] {
			if c != tt.out[j] {
				t.Errorf("%d: byte %d was %X; want %X", i, j, c, tt.out[j])
			}
		}
	}
}

func TestGetColElems(t *testing.T) {
	for i, tt := range appendNextTests {
		c, err := makeTable(tt.in)
		if err != nil {
			// error is reported in TestAppendNext
			continue
		}
		// Create one large test per table
		str := make([]byte, 0, 4000)
		out := ColElems{}
		for len(str) < 3000 {
			for _, chk := range tt.chk {
				str = append(str, chk.in[:chk.n]...)
				out = append(out, chk.out...)
			}
		}
		for j, chk := range append(tt.chk, check{string(str), len(str), out}) {
			out := convertFromWeights(chk.out)
			ce := c.getColElems([]byte(chk.in)[:chk.n])
			if len(ce) != len(out) {
				t.Errorf("%d:%d: len(ws) was %d; want %d", i, j, len(ce), len(out))
				continue
			}
			cnt := 0
			for k, w := range ce {
				w, _ = colltab.MakeElem(w.Primary(), w.Secondary(), int(w.Tertiary()), 0)
				if w != out[k] {
					t.Errorf("%d:%d: Weights %d was %X; want %X", i, j, k, w, out[k])
					cnt++
				}
				if cnt > 10 {
					break
				}
			}
		}
	}
}

type keyTest struct {
	in  string
	out []byte
}

var keyTests = []keyTest{
	{"abc",
		[]byte{0, 100, 0, 200, 1, 44, 0, 0, 0, 32, 0, 32, 0, 32, 0, 0, 2, 2, 2, 0, 255, 255, 255},
	},
	{"a\u0301",
		[]byte{0, 102, 0, 0, 0, 32, 0, 0, 2, 0, 255},
	},
	{"aaaaa",
		[]byte{0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 0,
			0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 0,
			2, 2, 2, 2, 2, 0,
			255, 255, 255, 255, 255,
		},
	},
	// Issue 16391: incomplete rune at end of UTF-8 sequence.
	{"\xc2", []byte{133, 255, 253, 0, 0, 0, 32, 0, 0, 2, 0, 255}},
	{"\xc2a", []byte{133, 255, 253, 0, 100, 0, 0, 0, 32, 0, 32, 0, 0, 2, 2, 0, 255, 255}},
}

func TestKey(t *testing.T) {
	c, _ := makeTable(appendNextTests[4].in)
	c.alternate = altShifted
	c.ignore = ignore(colltab.Quaternary)
	buf := Buffer{}
	keys1 := [][]byte{}
	keys2 := [][]byte{}
	for _, tt := range keyTests {
		keys1 = append(keys1, c.Key(&buf, []byte(tt.in)))
		keys2 = append(keys2, c.KeyFromString(&buf, tt.in))
	}
	// Separate generation from testing to ensure buffers are not overwritten.
	for i, tt := range keyTests {
		if !bytes.Equal(keys1[i], tt.out) {
			t.Errorf("%d: Key(%q) = %d; want %d", i, tt.in, keys1[i], tt.out)
		}
		if !bytes.Equal(keys2[i], tt.out) {
			t.Errorf("%d: KeyFromString(%q) = %d; want %d", i, tt.in, keys2[i], tt.out)
		}
	}
}

type compareTest struct {
	a, b string
	res  int // comparison result
}

var compareTests = []compareTest{
	{"a\u0301", "a", 1},
	{"a\u0301b", "ab", 1},
	{"a", "a\u0301", -1},
	{"ab", "a\u0301b", -1},
	{"bc", "a\u0301c", 1},
	{"ab", "aB", -1},
	{"a\u0301", "a\u0301", 0},
	{"a", "a", 0},
	// Only clip prefixes of whole runes.
	{"\u302E", "\u302F", 1},
	// Don't clip prefixes when last rune of prefix may be part of contraction.
	{"a\u035E", "a\u0301\u035F", -1},
	{"a\u0301\u035Fb", "a\u0301\u035F", -1},
}

func TestCompare(t *testing.T) {
	c, _ := makeTable(appendNextTests[4].in)
	for i, tt := range compareTests {
		if res := c.Compare([]byte(tt.a), []byte(tt.b)); res != tt.res {
			t.Errorf("%d: Compare(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
		}
		if res := c.CompareString(tt.a, tt.b); res != tt.res {
			t.Errorf("%d: CompareString(%q, %q) == %d; want %d", i, tt.a, tt.b, res, tt.res)
		}
	}
}

func TestNumeric(t *testing.T) {
	c := New(language.English, Loose, Numeric)

	for i, tt := range []struct {
		a, b string
		want int
	}{
		{"1", "2", -1},
		{"2", "12", -1},
		{"２", "１２", -1}, // Fullwidth is sorted as usual.
		{"₂", "₁₂", 1},  // Subscript is not sorted as numbers.
		{"②", "①②", 1},  // Circled is not sorted as numbers.
		{ // Imperial Aramaic, is not sorted as number.
			"\U00010859",
			"\U00010858\U00010859",
			1,
		},
		{"12", "2", 1},
		{"A-1", "A-2", -1},
		{"A-2", "A-12", -1},
		{"A-12", "A-2", 1},
		{"A-0001", "A-1", 0},
	} {
		if got := c.CompareString(tt.a, tt.b); got != tt.want {
			t.Errorf("%d: CompareString(%s, %s) = %d; want %d", i, tt.a, tt.b, got, tt.want)
		}
	}
}
