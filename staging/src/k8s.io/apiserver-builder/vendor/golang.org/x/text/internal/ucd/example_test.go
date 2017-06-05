// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ucd_test

import (
	"fmt"
	"strings"

	"golang.org/x/text/internal/ucd"
)

func Example() {
	// Read rune-by-rune from UnicodeData.
	var count int
	p := ucd.New(strings.NewReader(unicodeData))
	for p.Next() {
		count++
		if lower := p.Runes(ucd.SimpleLowercaseMapping); lower != nil {
			fmt.Printf("lower(%U) -> %U\n", p.Rune(0), lower[0])
		}
	}
	if err := p.Err(); err != nil {
		fmt.Println(err)
	}
	fmt.Println("Number of runes visited:", count)

	// Read raw ranges from Scripts.
	p = ucd.New(strings.NewReader(scripts), ucd.KeepRanges)
	for p.Next() {
		start, end := p.Range(0)
		fmt.Printf("%04X..%04X: %s\n", start, end, p.String(1))
	}
	if err := p.Err(); err != nil {
		fmt.Println(err)
	}

	// Output:
	// lower(U+00C0) -> U+00E0
	// lower(U+00C1) -> U+00E1
	// lower(U+00C2) -> U+00E2
	// lower(U+00C3) -> U+00E3
	// lower(U+00C4) -> U+00E4
	// Number of runes visited: 6594
	// 0000..001F: Common
	// 0020..0020: Common
	// 0021..0023: Common
	// 0024..0024: Common
}

// Excerpt from UnicodeData.txt
const unicodeData = `
00B9;SUPERSCRIPT ONE;No;0;EN;<super> 0031;;1;1;N;SUPERSCRIPT DIGIT ONE;;;;
00BA;MASCULINE ORDINAL INDICATOR;Lo;0;L;<super> 006F;;;;N;;;;;
00BB;RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK;Pf;0;ON;;;;;Y;RIGHT POINTING GUILLEMET;;;;
00BC;VULGAR FRACTION ONE QUARTER;No;0;ON;<fraction> 0031 2044 0034;;;1/4;N;FRACTION ONE QUARTER;;;;
00BD;VULGAR FRACTION ONE HALF;No;0;ON;<fraction> 0031 2044 0032;;;1/2;N;FRACTION ONE HALF;;;;
00BE;VULGAR FRACTION THREE QUARTERS;No;0;ON;<fraction> 0033 2044 0034;;;3/4;N;FRACTION THREE QUARTERS;;;;
00BF;INVERTED QUESTION MARK;Po;0;ON;;;;;N;;;;;
00C0;LATIN CAPITAL LETTER A WITH GRAVE;Lu;0;L;0041 0300;;;;N;LATIN CAPITAL LETTER A GRAVE;;;00E0;
00C1;LATIN CAPITAL LETTER A WITH ACUTE;Lu;0;L;0041 0301;;;;N;LATIN CAPITAL LETTER A ACUTE;;;00E1;
00C2;LATIN CAPITAL LETTER A WITH CIRCUMFLEX;Lu;0;L;0041 0302;;;;N;LATIN CAPITAL LETTER A CIRCUMFLEX;;;00E2;
00C3;LATIN CAPITAL LETTER A WITH TILDE;Lu;0;L;0041 0303;;;;N;LATIN CAPITAL LETTER A TILDE;;;00E3;
00C4;LATIN CAPITAL LETTER A WITH DIAERESIS;Lu;0;L;0041 0308;;;;N;LATIN CAPITAL LETTER A DIAERESIS;;;00E4;

# A legacy rune range.
3400;<CJK Ideograph Extension A, First>;Lo;0;L;;;;;N;;;;;
4DB5;<CJK Ideograph Extension A, Last>;Lo;0;L;;;;;N;;;;;
`

// Excerpt from Scripts.txt
const scripts = `
# Property:	Script
# ================================================

0000..001F    ; Common # Cc  [32] <control-0000>..<control-001F>
0020          ; Common # Zs       SPACE
0021..0023    ; Common # Po   [3] EXCLAMATION MARK..NUMBER SIGN
0024          ; Common # Sc       DOLLAR SIGN
`
