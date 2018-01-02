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
	"golang.org/x/text/secure/bidirule"
	"golang.org/x/text/transform"
)

type testCase struct {
	input  string
	output string
	err    error
}

var enforceTestCases = []struct {
	name  string
	p     *Profile
	cases []testCase
}{
	{"Basic", NewFreeform(), []testCase{
		{"e\u0301\u031f", "\u00e9\u031f", nil}, // normalize
	}},

	{"Context Rule 1", NewFreeform(), []testCase{
		// Rule 1: zero-width non-joiner (U+200C)
		// From RFC:
		//   False
		//   If Canonical_Combining_Class(Before(cp)) .eq.  Virama Then True;
		//   If RegExpMatch((Joining_Type:{L,D})(Joining_Type:T)*\u200C
		//          (Joining_Type:T)*(Joining_Type:{R,D})) Then True;
		//
		// Example runes for different joining types:
		// Join L: U+A872; PHAGS-PA SUPERFIXED LETTER RA
		// Join D: U+062C; HAH WITH DOT BELOW
		// Join T: U+0610; ARABIC SIGN SALLALLAHOU ALAYHE WASSALLAM
		// Join R: U+0627; ALEF
		// Virama: U+0A4D; GURMUKHI SIGN VIRAMA
		// Virama and Join T: U+0ACD; GUJARATI SIGN VIRAMA
		{"\u200c", "", errContext},
		{"\u200ca", "", errContext},
		{"a\u200c", "", errContext},
		{"\u200c\u0627", "", errContext},             // missing JoinStart
		{"\u062c\u200c", "", errContext},             // missing JoinEnd
		{"\u0610\u200c\u0610\u0627", "", errContext}, // missing JoinStart
		{"\u062c\u0610\u200c\u0610", "", errContext}, // missing JoinEnd

		// Variants of: D T* U+200c T* R
		{"\u062c\u200c\u0627", "\u062c\u200c\u0627", nil},
		{"\u062c\u0610\u200c\u0610\u0627", "\u062c\u0610\u200c\u0610\u0627", nil},
		{"\u062c\u0610\u0610\u200c\u0610\u0610\u0627", "\u062c\u0610\u0610\u200c\u0610\u0610\u0627", nil},
		{"\u062c\u0610\u200c\u0627", "\u062c\u0610\u200c\u0627", nil},
		{"\u062c\u200c\u0610\u0627", "\u062c\u200c\u0610\u0627", nil},

		// Variants of: L T* U+200c T* D
		{"\ua872\u200c\u062c", "\ua872\u200c\u062c", nil},
		{"\ua872\u0610\u200c\u0610\u062c", "\ua872\u0610\u200c\u0610\u062c", nil},
		{"\ua872\u0610\u0610\u200c\u0610\u0610\u062c", "\ua872\u0610\u0610\u200c\u0610\u0610\u062c", nil},
		{"\ua872\u0610\u200c\u062c", "\ua872\u0610\u200c\u062c", nil},
		{"\ua872\u200c\u0610\u062c", "\ua872\u200c\u0610\u062c", nil},

		// Virama
		{"\u0a4d\u200c", "\u0a4d\u200c", nil},
		{"\ua872\u0a4d\u200c", "\ua872\u0a4d\u200c", nil},
		{"\ua872\u0a4d\u0610\u200c", "", errContext},
		{"\ua872\u0a4d\u0610\u200c", "", errContext},

		{"\u0acd\u200c", "\u0acd\u200c", nil},
		{"\ua872\u0acd\u200c", "\ua872\u0acd\u200c", nil},
		{"\ua872\u0acd\u0610\u200c", "", errContext},
		{"\ua872\u0acd\u0610\u200c", "", errContext},

		// Using Virama as join T
		{"\ua872\u0acd\u200c\u062c", "\ua872\u0acd\u200c\u062c", nil},
		{"\ua872\u200c\u0acd\u062c", "\ua872\u200c\u0acd\u062c", nil},
	}},

	{"Context Rule 2", NewFreeform(), []testCase{
		// Rule 2: zero-width joiner (U+200D)
		{"\u200d", "", errContext},
		{"\u200da", "", errContext},
		{"a\u200d", "", errContext},

		{"\u0a4d\u200d", "\u0a4d\u200d", nil},
		{"\ua872\u0a4d\u200d", "\ua872\u0a4d\u200d", nil},
		{"\u0a4da\u200d", "", errContext},
	}},

	{"Context Rule 3", NewFreeform(), []testCase{
		// Rule 3: middle dot
		{"·", "", errContext},
		{"l·", "", errContext},
		{"·l", "", errContext},
		{"a·", "", errContext},
		{"l·a", "", errContext},
		{"a·a", "", errContext},
		{"l·l", "l·l", nil},
		{"al·la", "al·la", nil},
	}},

	{"Context Rule 4", NewFreeform(), []testCase{
		// Rule 4: Greek lower numeral U+0375
		{"͵", "", errContext},
		{"͵a", "", errContext},
		{"α͵", "", errContext},
		{"͵α", "͵α", nil},
		{"α͵α", "α͵α", nil},
		{"͵͵α", "͵͵α", nil}, // The numeric sign is itself Greek.
		{"α͵͵α", "α͵͵α", nil},
		{"α͵͵", "", errContext},
		{"α͵͵a", "", errContext},
	}},

	{"Context Rule 5+6", NewFreeform(), []testCase{
		// Rule 5+6: Hebrew preceding
		// U+05f3: Geresh
		{"׳", "", errContext},
		{"׳ה", "", errContext},
		{"a׳b", "", errContext},
		{"ש׳", "ש׳", nil},     // U+05e9 U+05f3
		{"ש׳׳׳", "ש׳׳׳", nil}, // U+05e9 U+05f3

		// U+05f4: Gershayim
		{"״", "", errContext},
		{"״ה", "", errContext},
		{"a״b", "", errContext},
		{"ש״", "ש״", nil},       // U+05e9 U+05f4
		{"ש״״״", "ש״״״", nil},   // U+05e9 U+05f4
		{"aש״״״", "aש״״״", nil}, // U+05e9 U+05f4
	}},

	{"Context Rule 7", NewFreeform(), []testCase{
		// Rule 7: Katakana middle Dot
		{"・", "", errContext},
		{"abc・", "", errContext},
		{"・def", "", errContext},
		{"abc・def", "", errContext},
		{"aヅc・def", "aヅc・def", nil},
		{"abc・dぶf", "abc・dぶf", nil},
		{"⺐bc・def", "⺐bc・def", nil},
	}},

	{"Context Rule 8+9", NewFreeform(), []testCase{
		// Rule 8+9: Arabic Indic Digit
		{"١٢٣٤٥۶", "", errContext},
		{"۱۲۳۴۵٦", "", errContext},
		{"١٢٣٤٥", "١٢٣٤٥", nil},
		{"۱۲۳۴۵", "۱۲۳۴۵", nil},
	}},

	{"Nickname", Nickname, []testCase{
		{"  Swan  of   Avon   ", "Swan of Avon", nil},
		{"", "", errEmptyString},
		{" ", "", errEmptyString},
		{"  ", "", errEmptyString},
		{"a\u00A0a\u1680a\u2000a\u2001a\u2002a\u2003a\u2004a\u2005a\u2006a\u2007a\u2008a\u2009a\u200Aa\u202Fa\u205Fa\u3000a", "a a a a a a a a a a a a a a a a a", nil},
		{"Foo", "Foo", nil},
		{"foo", "foo", nil},
		{"Foo Bar", "Foo Bar", nil},
		{"foo bar", "foo bar", nil},
		{"\u03A3", "\u03A3", nil},
		{"\u03C3", "\u03C3", nil},
		// Greek final sigma is left as is (do not fold!)
		{"\u03C2", "\u03C2", nil},
		{"\u265A", "♚", nil},
		{"Richard \u2163", "Richard IV", nil},
		{"\u212B", "Å", nil},
		{"\uFB00", "ff", nil}, // because of NFKC
		{"שa", "שa", nil},     // no bidi rule
		{"동일조건변경허락", "동일조건변경허락", nil},
	}},
	{"OpaqueString", OpaqueString, []testCase{
		{"  Swan  of   Avon   ", "  Swan  of   Avon   ", nil},
		{"", "", errEmptyString},
		{" ", " ", nil},
		{"  ", "  ", nil},
		{"a\u00A0a\u1680a\u2000a\u2001a\u2002a\u2003a\u2004a\u2005a\u2006a\u2007a\u2008a\u2009a\u200Aa\u202Fa\u205Fa\u3000a", "a a a a a a a a a a a a a a a a a", nil},
		{"Foo", "Foo", nil},
		{"foo", "foo", nil},
		{"Foo Bar", "Foo Bar", nil},
		{"foo bar", "foo bar", nil},
		{"\u03C3", "\u03C3", nil},
		{"Richard \u2163", "Richard \u2163", nil},
		{"\u212B", "Å", nil},
		{"Jack of \u2666s", "Jack of \u2666s", nil},
		{"my cat is a \u0009by", "", errDisallowedRune},
		{"שa", "שa", nil}, // no bidi rule
	}},
	{"UsernameCaseMapped", UsernameCaseMapped, []testCase{
		// TODO: Should this work?
		// {UsernameCaseMapped, "", "", errDisallowedRune},
		{"juliet@example.com", "juliet@example.com", nil},
		{"fussball", "fussball", nil},
		{"fu\u00DFball", "fu\u00DFball", nil},
		{"\u03C0", "\u03C0", nil},
		{"\u03A3", "\u03C3", nil},
		{"\u03C3", "\u03C3", nil},
		// Greek final sigma is left as is (do not fold!)
		{"\u03C2", "\u03C2", nil},
		{"\u0049", "\u0069", nil},
		{"\u0049", "\u0069", nil},
		{"\u03D2", "", errDisallowedRune},
		{"\u03B0", "\u03B0", nil},
		{"foo bar", "", errDisallowedRune},
		{"♚", "", errDisallowedRune},
		{"\u007E", "~", nil},
		{"a", "a", nil},
		{"!", "!", nil},
		{"²", "", errDisallowedRune},
		{"\t", "", errDisallowedRune},
		{"\n", "", errDisallowedRune},
		{"\u26D6", "", errDisallowedRune},
		{"\u26FF", "", errDisallowedRune},
		{"\uFB00", "", errDisallowedRune},
		{"\u1680", "", errDisallowedRune},
		{" ", "", errDisallowedRune},
		{"  ", "", errDisallowedRune},
		{"\u01C5", "", errDisallowedRune},
		{"\u16EE", "", errDisallowedRune}, // Nl RUNIC ARLAUG SYMBOL
		{"\u0488", "", errDisallowedRune}, // Me COMBINING CYRILLIC HUNDRED THOUSANDS SIGN
		{"\u212B", "\u00e5", nil},         // Angstrom sign, NFC -> U+00E5
		{"A\u030A", "å", nil},             // A + ring
		{"\u00C5", "å", nil},              // A with ring
		{"\u00E7", "ç", nil},              // c cedille
		{"\u0063\u0327", "ç", nil},        // c + cedille
		{"\u0158", "ř", nil},
		{"\u0052\u030C", "ř", nil},

		{"\u1E61", "\u1E61", nil}, // LATIN SMALL LETTER S WITH DOT ABOVE

		// Confusable characters ARE allowed and should NOT be mapped.
		{"\u0410", "\u0430", nil}, // CYRILLIC CAPITAL LETTER A

		// Full width should be mapped to the canonical decomposition.
		{"ＡＢ", "ab", nil},
		{"שc", "", bidirule.ErrInvalid}, // bidi rule

	}},
	{"UsernameCasePreserved", UsernameCasePreserved, []testCase{
		{"ABC", "ABC", nil},
		{"ＡＢ", "AB", nil},
		{"שc", "", bidirule.ErrInvalid}, // bidi rule
		{"\uFB00", "", errDisallowedRune},
		{"\u212B", "\u00c5", nil},    // Angstrom sign, NFC -> U+00E5
		{"ẛ", "", errDisallowedRune}, // LATIN SMALL LETTER LONG S WITH DOT ABOVE
	}},
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
	// Test that calling Bytes with something that doesn't transform returns a
	// copy.
	orig := []byte("hello")
	b, _ := NewFreeform().Bytes(orig)
	if reflect.ValueOf(b).Pointer() == reflect.ValueOf(orig).Pointer() {
		t.Error("original and result are the same slice; should be a copy")
	}
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
