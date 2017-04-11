// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "testing"

// TestCase is used for most tests.
type TestCase struct {
	in  []rune
	out []rune
}

func runTests(t *testing.T, name string, fm Form, tests []TestCase) {
	rb := reorderBuffer{}
	rb.init(fm, nil)
	for i, test := range tests {
		rb.setFlusher(nil, appendFlush)
		for j, rune := range test.in {
			b := []byte(string(rune))
			src := inputBytes(b)
			info := rb.f.info(src, 0)
			if j == 0 {
				rb.ss.first(info)
			} else {
				rb.ss.next(info)
			}
			if rb.insertFlush(src, 0, info) < 0 {
				t.Errorf("%s:%d: insert failed for rune %d", name, i, j)
			}
		}
		rb.doFlush()
		was := string(rb.out)
		want := string(test.out)
		if len(was) != len(want) {
			t.Errorf("%s:%d: length = %d; want %d", name, i, len(was), len(want))
		}
		if was != want {
			k, pfx := pidx(was, want)
			t.Errorf("%s:%d: \nwas  %s%+q; \nwant %s%+q", name, i, pfx, was[k:], pfx, want[k:])
		}
	}
}

func TestFlush(t *testing.T) {
	const (
		hello = "Hello "
		world = "world!"
	)
	buf := make([]byte, maxByteBufferSize)
	p := copy(buf, hello)
	out := buf[p:]
	rb := reorderBuffer{}
	rb.initString(NFC, world)
	if i := rb.flushCopy(out); i != 0 {
		t.Errorf("wrote bytes on flush of empty buffer. (len(out) = %d)", i)
	}

	for i := range world {
		// No need to set streamSafe values for this test.
		rb.insertFlush(rb.src, i, rb.f.info(rb.src, i))
		n := rb.flushCopy(out)
		out = out[n:]
		p += n
	}

	was := buf[:p]
	want := hello + world
	if string(was) != want {
		t.Errorf(`output after flush was "%s"; want "%s"`, string(was), want)
	}
	if rb.nrune != 0 {
		t.Errorf("non-null size of info buffer (rb.nrune == %d)", rb.nrune)
	}
	if rb.nbyte != 0 {
		t.Errorf("non-null size of byte buffer (rb.nbyte == %d)", rb.nbyte)
	}
}

var insertTests = []TestCase{
	{[]rune{'a'}, []rune{'a'}},
	{[]rune{0x300}, []rune{0x300}},
	{[]rune{0x300, 0x316}, []rune{0x316, 0x300}}, // CCC(0x300)==230; CCC(0x316)==220
	{[]rune{0x316, 0x300}, []rune{0x316, 0x300}},
	{[]rune{0x41, 0x316, 0x300}, []rune{0x41, 0x316, 0x300}},
	{[]rune{0x41, 0x300, 0x316}, []rune{0x41, 0x316, 0x300}},
	{[]rune{0x300, 0x316, 0x41}, []rune{0x316, 0x300, 0x41}},
	{[]rune{0x41, 0x300, 0x40, 0x316}, []rune{0x41, 0x300, 0x40, 0x316}},
}

func TestInsert(t *testing.T) {
	runTests(t, "TestInsert", NFD, insertTests)
}

var decompositionNFDTest = []TestCase{
	{[]rune{0xC0}, []rune{0x41, 0x300}},
	{[]rune{0xAC00}, []rune{0x1100, 0x1161}},
	{[]rune{0x01C4}, []rune{0x01C4}},
	{[]rune{0x320E}, []rune{0x320E}},
	{[]rune("음ẻ과"), []rune{0x110B, 0x1173, 0x11B7, 0x65, 0x309, 0x1100, 0x116A}},
}

var decompositionNFKDTest = []TestCase{
	{[]rune{0xC0}, []rune{0x41, 0x300}},
	{[]rune{0xAC00}, []rune{0x1100, 0x1161}},
	{[]rune{0x01C4}, []rune{0x44, 0x5A, 0x030C}},
	{[]rune{0x320E}, []rune{0x28, 0x1100, 0x1161, 0x29}},
}

func TestDecomposition(t *testing.T) {
	runTests(t, "TestDecompositionNFD", NFD, decompositionNFDTest)
	runTests(t, "TestDecompositionNFKD", NFKD, decompositionNFKDTest)
}

var compositionTest = []TestCase{
	{[]rune{0x41, 0x300}, []rune{0xC0}},
	{[]rune{0x41, 0x316}, []rune{0x41, 0x316}},
	{[]rune{0x41, 0x300, 0x35D}, []rune{0xC0, 0x35D}},
	{[]rune{0x41, 0x316, 0x300}, []rune{0xC0, 0x316}},
	// blocking starter
	{[]rune{0x41, 0x316, 0x40, 0x300}, []rune{0x41, 0x316, 0x40, 0x300}},
	{[]rune{0x1100, 0x1161}, []rune{0xAC00}},
	// parenthesized Hangul, alternate between ASCII and Hangul.
	{[]rune{0x28, 0x1100, 0x1161, 0x29}, []rune{0x28, 0xAC00, 0x29}},
}

func TestComposition(t *testing.T) {
	runTests(t, "TestComposition", NFC, compositionTest)
}
