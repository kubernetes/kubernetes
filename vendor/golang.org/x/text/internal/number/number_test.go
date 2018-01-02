// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"fmt"
	"testing"

	"golang.org/x/text/internal/testtext"
	"golang.org/x/text/language"
)

func TestInfo(t *testing.T) {
	testCases := []struct {
		lang     string
		sym      SymbolType
		wantSym  string
		wantNine rune
	}{
		{"und", SymDecimal, ".", '9'},
		{"de", SymGroup, ".", '9'},
		{"de-BE", SymGroup, ".", '9'},          // inherits from de (no number data in CLDR)
		{"de-BE-oxendict", SymGroup, ".", '9'}, // inherits from de (no compact index)

		// U+096F DEVANAGARI DIGIT NINE ('९')
		{"de-BE-u-nu-deva", SymGroup, ".", '\u096f'}, // miss -> latn -> de
		{"de-Cyrl-BE", SymGroup, ",", '9'},           // inherits from root
		{"de-CH", SymGroup, "’", '9'},                // overrides values in de
		{"de-CH-oxendict", SymGroup, "’", '9'},       // inherits from de-CH (no compact index)
		{"de-CH-u-nu-deva", SymGroup, "’", '\u096f'}, // miss -> latn -> de-CH

		{"pa", SymExponential, "E", '9'},

		// "×۱۰^" -> U+00d7 U+06f1 U+06f0^"
		// U+06F0 EXTENDED ARABIC-INDIC DIGIT ZERO
		// U+06F1 EXTENDED ARABIC-INDIC DIGIT ONE
		// U+06F9 EXTENDED ARABIC-INDIC DIGIT NINE
		{"pa-u-nu-arabext", SymExponential, "\u00d7\u06f1\u06f0^", '\u06f9'},

		//  "གྲངས་མེད" - > U+0f42 U+0fb2 U+0f44 U+0f66 U+0f0b U+0f58 U+0f7a U+0f51
		// Examples:
		// U+0F29 TIBETAN DIGIT NINE (༩)
		{"dz", SymInfinity, "\u0f42\u0fb2\u0f44\u0f66\u0f0b\u0f58\u0f7a\u0f51", '\u0f29'}, // defaults to tibt
		{"dz-u-nu-latn", SymInfinity, "∞", '9'},                                           // select alternative
		{"dz-u-nu-tibt", SymInfinity, "\u0f42\u0fb2\u0f44\u0f66\u0f0b\u0f58\u0f7a\u0f51", '\u0f29'},
		{"en-u-nu-tibt", SymInfinity, "∞", '\u0f29'},

		// algorithmic number systems fall back to ASCII if Digits is used.
		{"en-u-nu-hanidec", SymPlusSign, "+", '9'},
		{"en-u-nu-roman", SymPlusSign, "+", '9'},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s:%v", tc.lang, tc.sym), func(t *testing.T) {
			info := InfoFromTag(language.MustParse(tc.lang))
			if got := info.Symbol(tc.sym); got != tc.wantSym {
				t.Errorf("sym: got %q; want %q", got, tc.wantSym)
			}
			if got := info.Digit('9'); got != tc.wantNine {
				t.Errorf("Digit(9): got %+q; want %+q", got, tc.wantNine)
			}
			var buf [4]byte
			if got := string(buf[:info.WriteDigit(buf[:], '9')]); got != string(tc.wantNine) {
				t.Errorf("WriteDigit(9): got %+q; want %+q", got, tc.wantNine)
			}
			if got := string(info.AppendDigit([]byte{}, 9)); got != string(tc.wantNine) {
				t.Errorf("AppendDigit(9): got %+q; want %+q", got, tc.wantNine)
			}
		})
	}
}

func TestFormats(t *testing.T) {
	testCases := []struct {
		lang    string
		pattern string
		index   []byte
	}{
		{"en", "#,##0.###", tagToDecimal},
		{"de", "#,##0.###", tagToDecimal},
		{"de-CH", "#,##0.###", tagToDecimal},
		{"pa", "#,##,##0.###", tagToDecimal},
		{"pa-Arab", "#,##0.###", tagToDecimal}, // Does NOT inherit from pa!
		{"mr", "#,##,##0.###", tagToDecimal},
		{"mr-IN", "#,##,##0.###", tagToDecimal}, // Inherits from mr.
		{"nl", "#E0", tagToScientific},
		{"nl-MX", "#E0", tagToScientific}, // Inherits through Tag.Parent.
		{"zgh", "#,##0 %", tagToPercent},
	}
	for _, tc := range testCases {
		testtext.Run(t, tc.lang, func(t *testing.T) {
			got := formatForLang(language.MustParse(tc.lang), tc.index)
			want, _ := ParsePattern(tc.pattern)
			if *got != *want {
				t.Errorf("\ngot  %#v;\nwant %#v", got, want)
			}
		})
	}
}
