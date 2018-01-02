// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package idna

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/internal/ucd"
)

func TestAllocToUnicode(t *testing.T) {
	avg := testtext.AllocsPerRun(1000, func() {
		ToUnicode("www.golang.org")
	})
	if avg > 0 {
		t.Errorf("got %f; want 0", avg)
	}
}

func TestAllocToASCII(t *testing.T) {
	avg := testtext.AllocsPerRun(1000, func() {
		ToASCII("www.golang.org")
	})
	if avg > 0 {
		t.Errorf("got %f; want 0", avg)
	}
}

func TestProfiles(t *testing.T) {
	testCases := []struct {
		name      string
		want, got *Profile
	}{
		{"Punycode", punycode, New()},
		{"Registration", registration, New(ValidateForRegistration())},
		{"Registration", registration, New(
			ValidateForRegistration(),
			VerifyDNSLength(true),
			BidiRule(),
		)},
		{"Lookup", lookup, New(MapForLookup(), BidiRule(), Transitional(true))},
		{"Display", display, New(MapForLookup(), BidiRule())},
	}
	for _, tc := range testCases {
		// Functions are not comparable, but the printed version will include
		// their pointers.
		got := fmt.Sprintf("%#v", tc.got)
		want := fmt.Sprintf("%#v", tc.want)
		if got != want {
			t.Errorf("%s: \ngot  %#v,\nwant %#v", tc.name, got, want)
		}
	}
}

// doTest performs a single test f(input) and verifies that the output matches
// out and that the returned error is expected. The errors string contains
// all allowed error codes as categorized in
// http://www.unicode.org/Public/idna/9.0.0/IdnaTest.txt:
// P: Processing
// V: Validity
// A: to ASCII
// B: Bidi
// C: Context J
func doTest(t *testing.T, f func(string) (string, error), name, input, want, errors string) {
	errors = strings.Trim(errors, "[]")
	test := "ok"
	if errors != "" {
		test = "err:" + errors
	}
	// Replace some of the escape sequences to make it easier to single out
	// tests on the command name.
	in := strings.Trim(strconv.QuoteToASCII(input), `"`)
	in = strings.Replace(in, `\u`, "#", -1)
	in = strings.Replace(in, `\U`, "#", -1)
	name = fmt.Sprintf("%s/%s/%s", name, in, test)

	testtext.Run(t, name, func(t *testing.T) {
		got, err := f(input)

		if err != nil {
			code := err.(interface {
				code() string
			}).code()
			if strings.Index(errors, code) == -1 {
				t.Errorf("error %q not in set of expected errors {%v}", code, errors)
			}
		} else if errors != "" {
			t.Errorf("no errors; want error in {%v}", errors)
		}

		if want != "" && got != want {
			t.Errorf(`string: got %+q; want %+q`, got, want)
		}
	})
}

// TestLabelErrors tests strings returned in case of error. All results should
// be identical to the reference implementation and can be verified at
// http://unicode.org/cldr/utility/idna.jsp. The reference implementation,
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
		// Only strip leading empty labels for certain profiles. Stripping
		// leading empty labels here but not for "empty" punycode above seems
		// inconsistent, but seems to be applied by both the conformance test
		// and Chrome. So we turn it off by default, support it as an option,
		// and enable it in profiles where it seems commonplace.
		{lengthA, ".b", "b", ""},
		{lengthA, "\u3002b", "b", ""},
		{lengthA, "..b", "b", ""},
		{lengthA, "b..", "b..", ""},

		{resolve, "a..b", "a..b", ""},
		{resolve, ".b", "b", ""},
		{resolve, "\u3002b", "b", ""},
		{resolve, "..b", "b", ""},
		{resolve, "b..", "b..", ""},

		// Raw punycode
		{punyA, "", "", ""},
		{punyA, "*.foo.com", "*.foo.com", ""},
		{punyA, "Foo.com", "Foo.com", ""},

		// STD3 rules
		{display, "*.foo.com", "*.foo.com", "P1"},
		{std3, "*.foo.com", "*.foo.com", ""},

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

func TestConformance(t *testing.T) {
	testtext.SkipIfNotLong(t)

	r := gen.OpenUnicodeFile("idna", "", "IdnaTest.txt")
	defer r.Close()

	section := "main"
	started := false
	p := ucd.New(r, ucd.CommentHandler(func(s string) {
		if started {
			section = strings.ToLower(strings.Split(s, " ")[0])
		}
	}))
	transitional := New(Transitional(true), VerifyDNSLength(true), BidiRule(), MapForLookup())
	nonTransitional := New(VerifyDNSLength(true), BidiRule(), MapForLookup())
	for p.Next() {
		started = true

		// What to test
		profiles := []*Profile{}
		switch p.String(0) {
		case "T":
			profiles = append(profiles, transitional)
		case "N":
			profiles = append(profiles, nonTransitional)
		case "B":
			profiles = append(profiles, transitional)
			profiles = append(profiles, nonTransitional)
		}

		src := unescape(p.String(1))

		wantToUnicode := unescape(p.String(2))
		if wantToUnicode == "" {
			wantToUnicode = src
		}
		wantToASCII := unescape(p.String(3))
		if wantToASCII == "" {
			wantToASCII = wantToUnicode
		}
		wantErrToUnicode := ""
		if strings.HasPrefix(wantToUnicode, "[") {
			wantErrToUnicode = wantToUnicode
			wantToUnicode = ""
		}
		wantErrToASCII := ""
		if strings.HasPrefix(wantToASCII, "[") {
			wantErrToASCII = wantToASCII
			wantToASCII = ""
		}

		// TODO: also do IDNA tests.
		// invalidInIDNA2008 := p.String(4) == "NV8"

		for _, p := range profiles {
			name := fmt.Sprintf("%s:%s", section, p)
			doTest(t, p.ToUnicode, name+":ToUnicode", src, wantToUnicode, wantErrToUnicode)
			doTest(t, p.ToASCII, name+":ToASCII", src, wantToASCII, wantErrToASCII)
		}
	}
}

func unescape(s string) string {
	s, err := strconv.Unquote(`"` + s + `"`)
	if err != nil {
		panic(err)
	}
	return s
}
