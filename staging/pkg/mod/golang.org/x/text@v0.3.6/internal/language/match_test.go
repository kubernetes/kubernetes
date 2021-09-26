// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

import (
	"flag"
	"testing"
)

var verbose = flag.Bool("verbose", false, "set to true to print the internal tables of matchers")

func TestAddLikelySubtags(t *testing.T) {
	tests := []struct{ in, out string }{
		{"aa", "aa-Latn-ET"},
		{"aa-Latn", "aa-Latn-ET"},
		{"aa-Arab", "aa-Arab-ET"},
		{"aa-Arab-ER", "aa-Arab-ER"},
		{"kk", "kk-Cyrl-KZ"},
		{"kk-CN", "kk-Arab-CN"},
		{"cmn", "cmn"},
		{"zh-AU", "zh-Hant-AU"},
		{"zh-VN", "zh-Hant-VN"},
		{"zh-SG", "zh-Hans-SG"},
		{"zh-Hant", "zh-Hant-TW"},
		{"zh-Hani", "zh-Hani-CN"},
		{"und-Hani", "zh-Hani-CN"},
		{"und", "en-Latn-US"},
		{"und-GB", "en-Latn-GB"},
		{"und-CW", "pap-Latn-CW"},
		{"und-YT", "fr-Latn-YT"},
		{"und-Arab", "ar-Arab-EG"},
		{"und-AM", "hy-Armn-AM"},
		{"und-TW", "zh-Hant-TW"},
		{"und-002", "en-Latn-NG"},
		{"und-Latn-002", "en-Latn-NG"},
		{"en-Latn-002", "en-Latn-NG"},
		{"en-002", "en-Latn-NG"},
		{"en-001", "en-Latn-US"},
		{"und-003", "en-Latn-US"},
		{"und-GB", "en-Latn-GB"},
		{"Latn-001", "en-Latn-US"},
		{"en-001", "en-Latn-US"},
		{"es-419", "es-Latn-419"},
		{"he-145", "he-Hebr-IL"},
		{"ky-145", "ky-Latn-TR"},
		{"kk", "kk-Cyrl-KZ"},
		// Don't specialize duplicate and ambiguous matches.
		{"kk-034", "kk-Arab-034"}, // Matches IR and AF. Both are Arab.
		{"ku-145", "ku-Latn-TR"},  // Matches IQ, TR, and LB, but kk -> TR.
		{"und-Arab-CC", "ms-Arab-CC"},
		{"und-Arab-GB", "ks-Arab-GB"},
		{"und-Hans-CC", "zh-Hans-CC"},
		{"und-CC", "en-Latn-CC"},
		{"sr", "sr-Cyrl-RS"},
		{"sr-151", "sr-Latn-151"}, // Matches RO and RU.
		// We would like addLikelySubtags to generate the same results if the input
		// only changes by adding tags that would otherwise have been added
		// by the expansion.
		// In other words:
		//     und-AA -> xx-Scrp-AA   implies und-Scrp-AA -> xx-Scrp-AA
		//     und-AA -> xx-Scrp-AA   implies xx-AA -> xx-Scrp-AA
		//     und-Scrp -> xx-Scrp-AA implies und-Scrp-AA -> xx-Scrp-AA
		//     und-Scrp -> xx-Scrp-AA implies xx-Scrp -> xx-Scrp-AA
		//     xx -> xx-Scrp-AA       implies xx-Scrp -> xx-Scrp-AA
		//     xx -> xx-Scrp-AA       implies xx-AA -> xx-Scrp-AA
		//
		// The algorithm specified in
		//   https://unicode.org/reports/tr35/tr35-9.html#Supplemental_Data,
		// Section C.10, does not handle the first case. For example,
		// the CLDR data contains an entry und-BJ -> fr-Latn-BJ, but not
		// there is no rule for und-Latn-BJ.  According to spec, und-Latn-BJ
		// would expand to en-Latn-BJ, violating the aforementioned principle.
		// We deviate from the spec by letting und-Scrp-AA expand to xx-Scrp-AA
		// if a rule of the form und-AA -> xx-Scrp-AA is defined.
		// Note that as of version 23, CLDR has some explicitly specified
		// entries that do not conform to these rules. The implementation
		// will not correct these explicit inconsistencies. A later versions of CLDR
		// is supposed to fix this.
		{"und-Latn-BJ", "fr-Latn-BJ"},
		{"und-Bugi-ID", "bug-Bugi-ID"},
		// regions, scripts and languages without definitions
		{"und-Arab-AA", "ar-Arab-AA"},
		{"und-Afak-RE", "fr-Afak-RE"},
		{"und-Arab-GB", "ks-Arab-GB"},
		{"abp-Arab-GB", "abp-Arab-GB"},
		// script has preference over region
		{"und-Arab-NL", "ar-Arab-NL"},
		{"zza", "zza-Latn-TR"},
		// preserve variants and extensions
		{"de-1901", "de-Latn-DE-1901"},
		{"de-x-abc", "de-Latn-DE-x-abc"},
		{"de-1901-x-abc", "de-Latn-DE-1901-x-abc"},
		{"x-abc", "x-abc"}, // TODO: is this the desired behavior?
	}
	for i, tt := range tests {
		in, _ := Parse(tt.in)
		out, _ := Parse(tt.out)
		in, _ = in.addLikelySubtags()
		if in.String() != out.String() {
			t.Errorf("%d: add(%s) was %s; want %s", i, tt.in, in, tt.out)
		}
	}
}
func TestMinimize(t *testing.T) {
	tests := []struct{ in, out string }{
		{"aa", "aa"},
		{"aa-Latn", "aa"},
		{"aa-Latn-ET", "aa"},
		{"aa-ET", "aa"},
		{"aa-Arab", "aa-Arab"},
		{"aa-Arab-ER", "aa-Arab-ER"},
		{"aa-Arab-ET", "aa-Arab"},
		{"und", "und"},
		{"und-Latn", "und"},
		{"und-Latn-US", "und"},
		{"en-Latn-US", "en"},
		{"cmn", "cmn"},
		{"cmn-Hans", "cmn-Hans"},
		{"cmn-Hant", "cmn-Hant"},
		{"zh-AU", "zh-AU"},
		{"zh-VN", "zh-VN"},
		{"zh-SG", "zh-SG"},
		{"zh-Hant", "zh-Hant"},
		{"zh-Hant-TW", "zh-TW"},
		{"zh-Hans", "zh"},
		{"zh-Hani", "zh-Hani"},
		{"und-Hans", "und-Hans"},
		{"und-Hani", "und-Hani"},

		{"und-CW", "und-CW"},
		{"und-YT", "und-YT"},
		{"und-Arab", "und-Arab"},
		{"und-AM", "und-AM"},
		{"und-Arab-CC", "und-Arab-CC"},
		{"und-CC", "und-CC"},
		{"und-Latn-BJ", "und-BJ"},
		{"und-Bugi-ID", "und-Bugi"},
		{"bug-Bugi-ID", "bug-Bugi"},
		// regions, scripts and languages without definitions
		{"und-Arab-AA", "und-Arab-AA"},
		// preserve variants and extensions
		{"de-Latn-1901", "de-1901"},
		{"de-Latn-x-abc", "de-x-abc"},
		{"de-DE-1901-x-abc", "de-1901-x-abc"},
		{"x-abc", "x-abc"}, // TODO: is this the desired behavior?
	}
	for i, tt := range tests {
		in, _ := Parse(tt.in)
		out, _ := Parse(tt.out)
		min, _ := in.minimize()
		if min.String() != out.String() {
			t.Errorf("%d: min(%s) was %s; want %s", i, tt.in, min, tt.out)
		}
		max, _ := min.addLikelySubtags()
		if x, _ := in.addLikelySubtags(); x.String() != max.String() {
			t.Errorf("%d: max(min(%s)) = %s; want %s", i, tt.in, max, x)
		}
	}
}
