// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package number

import (
	"reflect"
	"testing"
	"unsafe"
)

var testCases = []struct {
	pat  string
	want *Pattern
}{{
	"#",
	&Pattern{
		FormatWidth: 1,
		// TODO: Should MinIntegerDigits be 1?
	},
}, {
	"0",
	&Pattern{
		FormatWidth: 1,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 1,
		},
	},
}, {
	"+0",
	&Pattern{
		Affix:       "\x01+\x00",
		FormatWidth: 2,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 1,
		},
	},
}, {
	"0+",
	&Pattern{
		Affix:       "\x00\x01+",
		FormatWidth: 2,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 1,
		},
	},
}, {
	"0000",
	&Pattern{
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 4,
		},
	},
}, {
	".#",
	&Pattern{
		FormatWidth: 2,
		RoundingContext: RoundingContext{
			MaxFractionDigits: 1,
		},
	},
}, {
	"#0.###",
	&Pattern{
		FormatWidth: 6,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 3,
		},
	},
}, {
	"#0.######",
	&Pattern{
		FormatWidth: 9,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 6,
		},
	},
}, {
	"#,0",
	&Pattern{
		FormatWidth:  3,
		GroupingSize: [2]uint8{1, 0},
		RoundingContext: RoundingContext{
			MinIntegerDigits: 1,
		},
	},
}, {
	"#,0.00",
	&Pattern{
		FormatWidth:  6,
		GroupingSize: [2]uint8{1, 0},
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MinFractionDigits: 2,
			MaxFractionDigits: 2,
		},
	},
}, {
	"#,##0.###",
	&Pattern{
		FormatWidth:  9,
		GroupingSize: [2]uint8{3, 0},
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 3,
		},
	},
}, {
	"#,##,##0.###",
	&Pattern{
		FormatWidth:  12,
		GroupingSize: [2]uint8{3, 2},
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 3,
		},
	},
}, {
	// Ignore additional separators.
	"#,####,##,##0.###",
	&Pattern{
		FormatWidth:  17,
		GroupingSize: [2]uint8{3, 2},
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 3,
		},
	},
}, {
	"#E0",
	&Pattern{
		FormatWidth: 3,
		RoundingContext: RoundingContext{
			MaxIntegerDigits:  1,
			MinExponentDigits: 1,
		},
	},
}, {
	// At least one exponent digit is required. As long as this is true, one can
	// determine that scientific rendering is needed if MinExponentDigits > 0.
	"#E#",
	nil,
}, {
	"0E0",
	&Pattern{
		FormatWidth: 3,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MinExponentDigits: 1,
		},
	},
}, {
	"##0.###E00",
	&Pattern{
		FormatWidth: 10,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxIntegerDigits:  3,
			MaxFractionDigits: 3,
			MinExponentDigits: 2,
		},
	},
}, {
	"##00.0#E0",
	&Pattern{
		FormatWidth: 9,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  2,
			MaxIntegerDigits:  4,
			MinFractionDigits: 1,
			MaxFractionDigits: 2,
			MinExponentDigits: 1,
		},
	},
}, {
	"#00.0E+0",
	&Pattern{
		FormatWidth: 8,
		Flags:       AlwaysExpSign,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  2,
			MaxIntegerDigits:  3,
			MinFractionDigits: 1,
			MaxFractionDigits: 1,
			MinExponentDigits: 1,
		},
	},
}, {
	"0.0E++0",
	nil,
}, {
	"#0E+",
	nil,
}, {
	// significant digits
	"@",
	&Pattern{
		FormatWidth: 1,
		RoundingContext: RoundingContext{
			MinSignificantDigits: 1,
			MaxSignificantDigits: 1,
			MaxFractionDigits:    -1,
		},
	},
}, {
	// significant digits
	"@@@@",
	&Pattern{
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			MinSignificantDigits: 4,
			MaxSignificantDigits: 4,
			MaxFractionDigits:    -1,
		},
	},
}, {
	"@###",
	&Pattern{
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			MinSignificantDigits: 1,
			MaxSignificantDigits: 4,
			MaxFractionDigits:    -1,
		},
	},
}, {
	// Exponents in significant digits mode gets normalized.
	"@@E0",
	&Pattern{
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxIntegerDigits:  1,
			MinFractionDigits: 1,
			MaxFractionDigits: 1,
			MinExponentDigits: 1,
		},
	},
}, {
	"@###E00",
	&Pattern{
		FormatWidth: 7,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxIntegerDigits:  1,
			MinFractionDigits: 0,
			MaxFractionDigits: 3,
			MinExponentDigits: 2,
		},
	},
}, {
	// The significant digits mode does not allow fractions.
	"@###.#E0",
	nil,
}, {
	//alternative negative pattern
	"#0.###;(#0.###)",
	&Pattern{
		Affix:       "\x00\x00\x01(\x01)",
		NegOffset:   2,
		FormatWidth: 6,
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MaxFractionDigits: 3,
		},
	},
}, {
	// Rounding increment
	"1.05",
	&Pattern{
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			Increment:         105,
			IncrementScale:    2,
			MinIntegerDigits:  1,
			MinFractionDigits: 2,
			MaxFractionDigits: 2,
		},
	},
}, {
	// Rounding increment with grouping
	"1,05",
	&Pattern{
		FormatWidth:  4,
		GroupingSize: [2]uint8{2, 0},
		RoundingContext: RoundingContext{
			Increment:         105,
			IncrementScale:    0,
			MinIntegerDigits:  3,
			MinFractionDigits: 0,
			MaxFractionDigits: 0,
		},
	},
}, {
	"0.0%",
	&Pattern{
		Affix:       "\x00\x01%",
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			DigitShift:        2,
			MinIntegerDigits:  1,
			MinFractionDigits: 1,
			MaxFractionDigits: 1,
		},
	},
}, {
	"0.0‰",
	&Pattern{
		Affix:       "\x00\x03‰",
		FormatWidth: 4,
		RoundingContext: RoundingContext{
			DigitShift:        3,
			MinIntegerDigits:  1,
			MinFractionDigits: 1,
			MaxFractionDigits: 1,
		},
	},
}, {
	"#,##0.00¤",
	&Pattern{
		Affix:        "\x00\x02¤",
		FormatWidth:  9,
		GroupingSize: [2]uint8{3, 0},
		RoundingContext: RoundingContext{
			MinIntegerDigits:  1,
			MinFractionDigits: 2,
			MaxFractionDigits: 2,
		},
	},
}, {
	"#,##0.00 ¤;(#,##0.00 ¤)",
	&Pattern{Affix: "\x00\x04\u00a0¤\x01(\x05\u00a0¤)",
		NegOffset:    6,
		FormatWidth:  10,
		GroupingSize: [2]uint8{3, 0},
		RoundingContext: RoundingContext{
			DigitShift:        0,
			MinIntegerDigits:  1,
			MinFractionDigits: 2,
			MaxFractionDigits: 2,
		},
	},
}, {
	// padding
	"*x#",
	&Pattern{
		PadRune:     'x',
		FormatWidth: 1,
	},
}, {
	// padding
	"#*x",
	&Pattern{
		PadRune:     'x',
		FormatWidth: 1,
		Flags:       PadBeforeSuffix,
	},
}, {
	"*xpre#suf",
	&Pattern{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
	},
}, {
	"pre*x#suf",
	&Pattern{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadAfterPrefix,
	},
}, {
	"pre#*xsuf",
	&Pattern{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadBeforeSuffix,
	},
}, {
	"pre#suf*x",
	&Pattern{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadAfterSuffix,
	},
}, {
	`* #0 o''clock`,
	&Pattern{Affix: "\x00\x09 o\\'clock",
		FormatWidth: 10,
		PadRune:     32,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 0x1,
		},
	},
}, {
	`'123'* #0'456'`,
	&Pattern{Affix: "\x05'123'\x05'456'",
		FormatWidth: 8,
		PadRune:     32,
		RoundingContext: RoundingContext{
			MinIntegerDigits: 0x1,
		},
		Flags: PadAfterPrefix},
}, {
	// no duplicate padding
	"*xpre#suf*x", nil,
}, {
	// no duplicate padding
	"*xpre#suf*x", nil,
}}

func TestParsePattern(t *testing.T) {
	for i, tc := range testCases {
		t.Run(tc.pat, func(t *testing.T) {
			f, err := ParsePattern(tc.pat)
			if !reflect.DeepEqual(f, tc.want) {
				t.Errorf("%d:%s:\ngot %#v;\nwant %#v", i, tc.pat, f, tc.want)
			}
			if got, want := err != nil, tc.want == nil; got != want {
				t.Errorf("%d:%s:error: got %v; want %v", i, tc.pat, err, want)
			}
		})
	}
}

func TestPatternSize(t *testing.T) {
	if sz := unsafe.Sizeof(Pattern{}); sz > 56 {
		t.Errorf("got %d; want <= 56", sz)
	}

}
