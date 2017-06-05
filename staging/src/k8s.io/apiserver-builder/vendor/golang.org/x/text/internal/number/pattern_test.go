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
	want *Format
}{{
	"#",
	&Format{
		FormatWidth: 1,
		// TODO: Should MinIntegerDigits be 1?
	},
}, {
	"0",
	&Format{
		FormatWidth:      1,
		MinIntegerDigits: 1,
	},
}, {
	"0000",
	&Format{
		FormatWidth:      4,
		MinIntegerDigits: 4,
	},
}, {
	".#",
	&Format{
		FormatWidth:       2,
		MaxFractionDigits: 1,
	},
}, {
	"#0.###",
	&Format{
		FormatWidth:       6,
		MinIntegerDigits:  1,
		MaxFractionDigits: 3,
	},
}, {
	"#0.######",
	&Format{
		FormatWidth:       9,
		MinIntegerDigits:  1,
		MaxFractionDigits: 6,
	},
}, {
	"#,##0.###",
	&Format{
		FormatWidth:       9,
		GroupingSize:      [2]uint8{3, 0},
		MinIntegerDigits:  1,
		MaxFractionDigits: 3,
	},
}, {
	"#,##,##0.###",
	&Format{
		FormatWidth:       12,
		GroupingSize:      [2]uint8{3, 2},
		MinIntegerDigits:  1,
		MaxFractionDigits: 3,
	},
}, {
	// Ignore additional separators.
	"#,####,##,##0.###",
	&Format{
		FormatWidth:       17,
		GroupingSize:      [2]uint8{3, 2},
		MinIntegerDigits:  1,
		MaxFractionDigits: 3,
	},
}, {
	"#E0",
	&Format{
		FormatWidth:       3,
		MaxIntegerDigits:  1,
		MinExponentDigits: 1,
	},
}, {
	"0E0",
	&Format{
		FormatWidth:       3,
		MinIntegerDigits:  1,
		MinExponentDigits: 1,
	},
}, {
	"##00.0#E0",
	&Format{
		FormatWidth:       9,
		MinIntegerDigits:  2,
		MaxIntegerDigits:  4,
		MinFractionDigits: 1,
		MaxFractionDigits: 2,
		MinExponentDigits: 1,
	},
}, {
	"#00.0E+0",
	&Format{
		FormatWidth:       8,
		Flags:             AlwaysExpSign,
		MinIntegerDigits:  2,
		MaxIntegerDigits:  3,
		MinFractionDigits: 1,
		MaxFractionDigits: 1,
		MinExponentDigits: 1,
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
	&Format{
		FormatWidth:          1,
		MinSignificantDigits: 1,
		MaxSignificantDigits: 1,
	},
}, {
	// significant digits
	"@@@@",
	&Format{
		FormatWidth:          4,
		MinSignificantDigits: 4,
		MaxSignificantDigits: 4,
	},
}, {
	"@###",
	&Format{
		FormatWidth:          4,
		MinSignificantDigits: 1,
		MaxSignificantDigits: 4,
	},
}, {
	// Exponents in significant digits mode gets normalized.
	"@@E0",
	&Format{
		FormatWidth:       4,
		MinIntegerDigits:  1,
		MaxIntegerDigits:  1,
		MinFractionDigits: 1,
		MaxFractionDigits: 1,
		MinExponentDigits: 1,
	},
}, {
	"@###E00",
	&Format{
		FormatWidth:       7,
		MinIntegerDigits:  1,
		MaxIntegerDigits:  1,
		MinFractionDigits: 0,
		MaxFractionDigits: 3,
		MinExponentDigits: 2,
	},
}, {
	// The significant digits mode does not allow fractions.
	"@###.#E0",
	nil,
}, {
	//alternative negative pattern
	"#0.###;(#0.###)",
	&Format{
		Affix:             "\x00\x00\x01(\x01)",
		NegOffset:         2,
		FormatWidth:       6,
		MinIntegerDigits:  1,
		MaxFractionDigits: 3,
	},
}, {
	// Rounding increments
	"1.05",
	&Format{
		RoundIncrement:    105,
		FormatWidth:       4,
		MinIntegerDigits:  1,
		MinFractionDigits: 2,
		MaxFractionDigits: 2,
	},
}, {
	"0.0%",
	&Format{
		Affix:             "\x00\x01%",
		Multiplier:        100,
		FormatWidth:       4,
		MinIntegerDigits:  1,
		MinFractionDigits: 1,
		MaxFractionDigits: 1,
	},
}, {
	"0.0‰",
	&Format{
		Affix:             "\x00\x03‰",
		Multiplier:        1000,
		FormatWidth:       4,
		MinIntegerDigits:  1,
		MinFractionDigits: 1,
		MaxFractionDigits: 1,
	},
}, {
	"#,##0.00¤",
	&Format{
		Affix:             "\x00\x02¤",
		FormatWidth:       9,
		GroupingSize:      [2]uint8{3, 0},
		MinIntegerDigits:  1,
		MinFractionDigits: 2,
		MaxFractionDigits: 2,
	},
}, {
	"#,##0.00 ¤;(#,##0.00 ¤)",
	&Format{Affix: "\x00\x04\u00a0¤\x01(\x05\u00a0¤)",
		NegOffset:         6,
		Multiplier:        0,
		FormatWidth:       10,
		GroupingSize:      [2]uint8{3, 0},
		MinIntegerDigits:  1,
		MinFractionDigits: 2,
		MaxFractionDigits: 2,
	},
}, {
	// padding
	"*x#",
	&Format{
		PadRune:     'x',
		FormatWidth: 1,
	},
}, {
	// padding
	"#*x",
	&Format{
		PadRune:     'x',
		FormatWidth: 1,
		Flags:       PadBeforeSuffix,
	},
}, {
	"*xpre#suf",
	&Format{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
	},
}, {
	"pre*x#suf",
	&Format{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadAfterPrefix,
	},
}, {
	"pre#*xsuf",
	&Format{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadBeforeSuffix,
	},
}, {
	"pre#suf*x",
	&Format{
		Affix:       "\x03pre\x03suf",
		PadRune:     'x',
		FormatWidth: 7,
		Flags:       PadAfterSuffix,
	},
}, {
	// no duplicate padding
	"*xpre#suf*x", nil,
}, {
	// no duplicate padding
	"*xpre#suf*x", nil,
}}

func TestParsePattern(t *testing.T) {
	for i, tc := range testCases {
		f, err := ParsePattern(tc.pat)
		if !reflect.DeepEqual(f, tc.want) {
			t.Errorf("%d:%s:\ngot %#v;\nwant %#v", i, tc.pat, f, tc.want)
		}
		if got, want := err != nil, tc.want == nil; got != want {
			t.Errorf("%d:%s:error: got %v; want %v", i, tc.pat, err, want)
		}
	}
}

func TestPatternSize(t *testing.T) {
	if sz := unsafe.Sizeof(Format{}); sz > 48 {
		t.Errorf("got %d; want 48", sz)
	}

}
