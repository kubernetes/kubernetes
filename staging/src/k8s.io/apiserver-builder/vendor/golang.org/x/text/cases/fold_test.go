// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import (
	"testing"

	"golang.org/x/text/internal/testtext"
)

func TestFold(t *testing.T) {
	testCases := []string{
		"βß\u13f8",        // "βssᏰ"
		"ab\u13fc\uab7aꭰ", // abᏴᎪᎠ
		"aﬃﬄaﬆ",           // affifflast
		"Iİiı\u0345",      // ii̇iıι
		"µµΜΜςσΣΣ",        // μμμμσσσσ
	}
	for _, tc := range testCases {
		testEntry := func(name string, c Caser, m func(r rune) string) {
			want := ""
			for _, r := range tc {
				want += m(r)
			}
			if got := c.String(tc); got != want {
				t.Errorf("%s(%s) = %+q; want %+q", name, tc, got, want)
			}
			dst := make([]byte, 256) // big enough to hold any result
			src := []byte(tc)
			v := testtext.AllocsPerRun(20, func() {
				c.Transform(dst, src, true)
			})
			if v > 0 {
				t.Errorf("%s(%s): number of allocs was %f; want 0", name, tc, v)
			}
		}
		testEntry("FullFold", Fold(), func(r rune) string {
			return runeFoldData(r).full
		})
		// TODO:
		// testEntry("SimpleFold", Fold(Compact), func(r rune) string {
		// 	return runeFoldData(r).simple
		// })
		// testEntry("SpecialFold", Fold(Turkic), func(r rune) string {
		// 	return runeFoldData(r).special
		// })
	}
}

func BenchmarkFullFold(b *testing.B)      { benchTransformer(b, Fold(), txtNonASCII) }
func BenchmarkFullFoldASCII(b *testing.B) { benchTransformer(b, Fold(), txtASCII) }
