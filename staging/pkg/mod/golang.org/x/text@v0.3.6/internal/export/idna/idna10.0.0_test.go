// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.10
// +build go1.10

package idna

import "testing"

// TestLabelErrors tests strings returned in case of error. All results should
// be identical to the reference implementation and can be verified at
// https://unicode.org/cldr/utility/idna.jsp. The reference implementation,
// however, seems to not display Bidi and ContextJ errors.
//
// In some cases the behavior of browsers is added as a comment. In all cases,
// whenever a resolve search returns an error here, Chrome will treat the input
// string as a search string (including those for Bidi and Context J errors),
// unless noted otherwise.
func TestLabelErrors(t *testing.T) {
	encode := func(s string) string { s, _ = encode(acePrefix, s); return s }
	type kind struct {
		name string
		f    func(string) (string, error)
	}
	punyA := kind{"PunycodeA", punycode.ToASCII}
	resolve := kind{"ResolveA", Lookup.ToASCII}
	display := kind{"ToUnicode", Display.ToUnicode}
	p := New(VerifyDNSLength(true), MapForLookup(), BidiRule())
	lengthU := kind{"CheckLengthU", p.ToUnicode}
	lengthA := kind{"CheckLengthA", p.ToASCII}
	p = New(MapForLookup(), StrictDomainName(false))
	std3 := kind{"STD3", p.ToASCII}
	p = New(MapForLookup(), CheckHyphens(false))
	hyphens := kind{"CheckHyphens", p.ToASCII}

	testCases := []struct {
		kind
		input   string
		want    string
		wantErr string
	}{
		{lengthU, "", "", "A4"}, // From UTS 46 conformance test.
		{lengthA, "", "", "A4"},

		{lengthU, "xn--", "", "A4"},
		{lengthU, "foo.xn--", "foo.", "A4"}, // TODO: is dropping xn-- correct?
		{lengthU, "xn--.foo", ".foo", "A4"},
		{lengthU, "foo.xn--.bar", "foo..bar", "A4"},

		{display, "xn--", "", ""},
		{display, "foo.xn--", "foo.", ""}, // TODO: is dropping xn-- correct?
		{display, "xn--.foo", ".foo", ""},
		{display, "foo.xn--.bar", "foo..bar", ""},

		{lengthA, "a..b", "a..b", "A4"},
		{punyA, ".b", ".b", ""},
		// For backwards compatibility, the Punycode profile does not map runes.
		{punyA, "\u3002b", "xn--b-83t", ""},
		{punyA, "..b", "..b", ""},

		{lengthA, ".b", ".b", "A4"},
		{lengthA, "\u3002b", ".b", "A4"},
		{lengthA, "..b", "..b", "A4"},
		{lengthA, "b..", "b..", ""},

		// Sharpened Bidi rules for Unicode 10.0.0. Apply for ALL labels in ANY
		// of the labels is RTL.
		{lengthA, "\ufe05\u3002\u3002\U0002603e\u1ce0", "..xn--t6f5138v", "A4"},
		{lengthA, "FAX\u2a77\U0001d186\u3002\U0001e942\U000e0181\u180c", "", "B6"},

		{resolve, "a..b", "a..b", ""},
		// Note that leading dots are not stripped. This is to be consistent
		// with the Punycode profile as well as the conformance test.
		{resolve, ".b", ".b", ""},
		{resolve, "\u3002b", ".b", ""},
		{resolve, "..b", "..b", ""},
		{resolve, "b..", "b..", ""},
		{resolve, "\xed", "", "P1"},

		// Raw punycode
		{punyA, "", "", ""},
		{punyA, "*.foo.com", "*.foo.com", ""},
		{punyA, "Foo.com", "Foo.com", ""},

		// STD3 rules
		{display, "*.foo.com", "*.foo.com", "P1"},
		{std3, "*.foo.com", "*.foo.com", ""},

		// Hyphens
		{display, "r3---sn-apo3qvuoxuxbt-j5pe.googlevideo.com", "r3---sn-apo3qvuoxuxbt-j5pe.googlevideo.com", "V2"},
		{hyphens, "r3---sn-apo3qvuoxuxbt-j5pe.googlevideo.com", "r3---sn-apo3qvuoxuxbt-j5pe.googlevideo.com", ""},
		{display, "-label-.com", "-label-.com", "V3"},
		{hyphens, "-label-.com", "-label-.com", ""},

		// Don't map U+2490 (DIGIT NINE FULL STOP). This is the behavior of
		// Chrome, Safari, and IE. Firefox will first map ⒐ to 9. and return
		// lab9.be.
		{resolve, "lab⒐be", "xn--labbe-zh9b", "P1"}, // encode("lab⒐be")
		{display, "lab⒐be", "lab⒐be", "P1"},

		{resolve, "plan⒐faß.de", "xn--planfass-c31e.de", "P1"}, // encode("plan⒐fass") + ".de"
		{display, "Plan⒐faß.de", "plan⒐faß.de", "P1"},

		// Chrome 54.0 recognizes the error and treats this input verbatim as a
		// search string.
		// Safari 10.0 (non-conform spec) decomposes "⒈" and computes the
		// punycode on the result using transitional mapping.
		// Firefox 49.0.1 goes haywire on this string and prints a bunch of what
		// seems to be nested punycode encodings.
		{resolve, "日本⒈co.ßßß.de", "xn--co-wuw5954azlb.ssssss.de", "P1"},
		{display, "日本⒈co.ßßß.de", "日本⒈co.ßßß.de", "P1"},

		{resolve, "a\u200Cb", "ab", ""},
		{display, "a\u200Cb", "a\u200Cb", "C"},

		{resolve, encode("a\u200Cb"), encode("a\u200Cb"), "C"},
		{display, "a\u200Cb", "a\u200Cb", "C"},

		{resolve, "grﻋﺮﺑﻲ.de", "xn--gr-gtd9a1b0g.de", "B"},
		{
			// Notice how the string gets transformed, even with an error.
			// Chrome will use the original string if it finds an error, so not
			// the transformed one.
			display,
			"gr\ufecb\ufeae\ufe91\ufef2.de",
			"gr\u0639\u0631\u0628\u064a.de",
			"B",
		},

		{resolve, "\u0671.\u03c3\u07dc", "xn--qib.xn--4xa21s", "B"}, // ٱ.σߜ
		{display, "\u0671.\u03c3\u07dc", "\u0671.\u03c3\u07dc", "B"},

		// normalize input
		{resolve, "a\u0323\u0322", "xn--jta191l", ""}, // ạ̢
		{display, "a\u0323\u0322", "\u1ea1\u0322", ""},

		// Non-normalized strings are not normalized when they originate from
		// punycode. Despite the error, Chrome, Safari and Firefox will attempt
		// to look up the input punycode.
		{resolve, encode("a\u0323\u0322") + ".com", "xn--a-tdbc.com", "V1"},
		{display, encode("a\u0323\u0322") + ".com", "a\u0323\u0322.com", "V1"},
	}

	for _, tc := range testCases {
		doTest(t, tc.f, tc.name, tc.input, tc.want, tc.wantErr)
	}
}
