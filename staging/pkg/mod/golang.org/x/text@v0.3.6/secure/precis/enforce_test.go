// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package precis

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/transform"
)

type testCase struct {
	input  string
	output string
	err    error
}

func doTests(t *testing.T, fn func(t *testing.T, p *Profile, tc testCase)) {
	for _, g := range enforceTestCases {
		for i, tc := range g.cases {
			name := fmt.Sprintf("%s:%d:%+q", g.name, i, tc.input)
			testtext.Run(t, name, func(t *testing.T) {
				fn(t, g.p, tc)
			})
		}
	}
}

func TestString(t *testing.T) {
	doTests(t, func(t *testing.T, p *Profile, tc testCase) {
		if e, err := p.String(tc.input); tc.err != err || e != tc.output {
			t.Errorf("got %+q (err: %v); want %+q (err: %v)", e, err, tc.output, tc.err)
		}
	})
}

func TestBytes(t *testing.T) {
	doTests(t, func(t *testing.T, p *Profile, tc testCase) {
		if e, err := p.Bytes([]byte(tc.input)); tc.err != err || string(e) != tc.output {
			t.Errorf("got %+q (err: %v); want %+q (err: %v)", string(e), err, tc.output, tc.err)
		}
	})

	t.Run("Copy", func(t *testing.T) {
		// Test that calling Bytes with something that doesn't transform returns a
		// copy.
		orig := []byte("hello")
		b, _ := NewFreeform().Bytes(orig)
		if reflect.ValueOf(b).Pointer() == reflect.ValueOf(orig).Pointer() {
			t.Error("original and result are the same slice; should be a copy")
		}
	})
}

func TestAppend(t *testing.T) {
	doTests(t, func(t *testing.T, p *Profile, tc testCase) {
		if e, err := p.Append(nil, []byte(tc.input)); tc.err != err || string(e) != tc.output {
			t.Errorf("got %+q (err: %v); want %+q (err: %v)", string(e), err, tc.output, tc.err)
		}
	})
}

func TestStringMallocs(t *testing.T) {
	if n := testtext.AllocsPerRun(100, func() { UsernameCaseMapped.String("helloworld") }); n > 0 {
		// TODO: reduce this to 0.
		t.Skipf("got %f allocs, want 0", n)
	}
}

func TestAppendMallocs(t *testing.T) {
	str := []byte("helloworld")
	out := make([]byte, 0, len(str))
	if n := testtext.AllocsPerRun(100, func() { UsernameCaseMapped.Append(out, str) }); n > 0 {
		t.Errorf("got %f allocs, want 0", n)
	}
}

func TestTransformMallocs(t *testing.T) {
	str := []byte("helloworld")
	out := make([]byte, 0, len(str))
	tr := UsernameCaseMapped.NewTransformer()
	if n := testtext.AllocsPerRun(100, func() {
		tr.Reset()
		tr.Transform(out, str, true)
	}); n > 0 {
		t.Errorf("got %f allocs, want 0", n)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestTransformerShortBuffers tests that the precis.Transformer implements the
// spirit, not just the letter (the method signatures), of the
// transform.Transformer interface.
//
// In particular, it tests that, if one or both of the dst or src buffers are
// short, so that multiple Transform calls are required to complete the overall
// transformation, the end result is identical to one Transform call with
// sufficiently long buffers.
func TestTransformerShortBuffers(t *testing.T) {
	srcUnit := []byte("a\u0300cce\u0301nts") // NFD normalization form.
	wantUnit := []byte("àccénts")            // NFC normalization form.
	src := bytes.Repeat(srcUnit, 16)
	want := bytes.Repeat(wantUnit, 16)
	const long = 4096
	dst := make([]byte, long)

	// 5, 7, 9, 11, 13, 16 and 17 are all pair-wise co-prime, which means that
	// slicing the dst and src buffers into 5, 7, 13 and 17 byte chunks will
	// fall at different places inside the repeated srcUnit's and wantUnit's.
	if len(srcUnit) != 11 || len(wantUnit) != 9 || len(src) > long || len(want) > long {
		t.Fatal("inconsistent lengths")
	}

	tr := NewFreeform().NewTransformer()
	for _, deltaD := range []int{5, 7, 13, 17, long} {
	loop:
		for _, deltaS := range []int{5, 7, 13, 17, long} {
			tr.Reset()
			d0 := 0
			s0 := 0
			for {
				d1 := min(len(dst), d0+deltaD)
				s1 := min(len(src), s0+deltaS)
				nDst, nSrc, err := tr.Transform(dst[d0:d1:d1], src[s0:s1:s1], s1 == len(src))
				d0 += nDst
				s0 += nSrc
				if err == nil {
					break
				}
				if err == transform.ErrShortDst || err == transform.ErrShortSrc {
					continue
				}
				t.Errorf("deltaD=%d, deltaS=%d: %v", deltaD, deltaS, err)
				continue loop
			}
			if s0 != len(src) {
				t.Errorf("deltaD=%d, deltaS=%d: s0: got %d, want %d", deltaD, deltaS, s0, len(src))
				continue
			}
			if d0 != len(want) {
				t.Errorf("deltaD=%d, deltaS=%d: d0: got %d, want %d", deltaD, deltaS, d0, len(want))
				continue
			}
			got := dst[:d0]
			if !bytes.Equal(got, want) {
				t.Errorf("deltaD=%d, deltaS=%d:\ngot  %q\nwant %q", deltaD, deltaS, got, want)
				continue
			}
		}
	}
}
