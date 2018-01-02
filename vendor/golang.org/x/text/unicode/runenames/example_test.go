// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runenames_test

import (
	"fmt"

	"golang.org/x/text/unicode/runenames"
)

func Example() {
	runes := []rune{
		-1,
		'\U00000000',
		'\U0000001f',
		'\U00000020',
		'\U00000021',
		'\U00000041',
		'\U0000007e',
		'\U0000007f',
		'\U00000080',
		'\U000000e0',

		'\U0000037f',
		'\U00000380',
		'\U00000381',
		'\U00000382',
		'\U00000383',
		'\U00000384',
		'\U00000385',
		'\U00000386',
		'\U000007c0',

		'\U00002603',
		'\U000033ff',
		'\U00003400',
		'\U00003401',
		'\U00003402',
		'\U00004dc0',

		'\U00009fd5',
		'\U00009fd6',
		'\U00009fff',
		'\U0000a000',
		0xdc00, // '\U0000dc00' (Low Surrogate) is an invalid Go literal.
		'\U0000f800',
		'\U0000fffc',
		'\U0000fffd',
		'\U0000fffe',
		'\U0000ffff',

		'\U00010000',
		'\U0001f574',
		'\U0002fa1d',
		'\U0002fa1e',
		'\U000e0100',
		'\U000e01ef',
		'\U000e01f0',
		'\U00100000',
		'\U0010fffd',
		'\U0010fffe',
		'\U0010ffff',
	}

	for _, r := range runes {
		fmt.Printf("%08x %q\n", r, runenames.Name(r))
	}

	// Output:
	// -0000001 ""
	// 00000000 "<control>"
	// 0000001f "<control>"
	// 00000020 "SPACE"
	// 00000021 "EXCLAMATION MARK"
	// 00000041 "LATIN CAPITAL LETTER A"
	// 0000007e "TILDE"
	// 0000007f "<control>"
	// 00000080 "<control>"
	// 000000e0 "LATIN SMALL LETTER A WITH GRAVE"
	// 0000037f "GREEK CAPITAL LETTER YOT"
	// 00000380 ""
	// 00000381 ""
	// 00000382 ""
	// 00000383 ""
	// 00000384 "GREEK TONOS"
	// 00000385 "GREEK DIALYTIKA TONOS"
	// 00000386 "GREEK CAPITAL LETTER ALPHA WITH TONOS"
	// 000007c0 "NKO DIGIT ZERO"
	// 00002603 "SNOWMAN"
	// 000033ff "SQUARE GAL"
	// 00003400 "<CJK Ideograph Extension A>"
	// 00003401 "<CJK Ideograph Extension A>"
	// 00003402 "<CJK Ideograph Extension A>"
	// 00004dc0 "HEXAGRAM FOR THE CREATIVE HEAVEN"
	// 00009fd5 "<CJK Ideograph>"
	// 00009fd6 ""
	// 00009fff ""
	// 0000a000 "YI SYLLABLE IT"
	// 0000dc00 "<Low Surrogate>"
	// 0000f800 "<Private Use>"
	// 0000fffc "OBJECT REPLACEMENT CHARACTER"
	// 0000fffd "REPLACEMENT CHARACTER"
	// 0000fffe ""
	// 0000ffff ""
	// 00010000 "LINEAR B SYLLABLE B008 A"
	// 0001f574 "MAN IN BUSINESS SUIT LEVITATING"
	// 0002fa1d "CJK COMPATIBILITY IDEOGRAPH-2FA1D"
	// 0002fa1e ""
	// 000e0100 "VARIATION SELECTOR-17"
	// 000e01ef "VARIATION SELECTOR-256"
	// 000e01f0 ""
	// 00100000 "<Plane 16 Private Use>"
	// 0010fffd "<Plane 16 Private Use>"
	// 0010fffe ""
	// 0010ffff ""
}
