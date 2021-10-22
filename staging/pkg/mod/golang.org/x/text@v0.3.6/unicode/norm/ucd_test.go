// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"bufio"
	"bytes"
	"fmt"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"golang.org/x/text/internal/gen"
	"golang.org/x/text/internal/testtext"
)

var once sync.Once

func skipShort(t *testing.T) {
	testtext.SkipIfNotLong(t)

	once.Do(func() { loadTestData(t) })
}

// This regression test runs the test set in NormalizationTest.txt
// (taken from https://www.unicode.org/Public/<unicode.Version>/ucd/).
//
// NormalizationTest.txt has form:
// @Part0 # Specific cases
// #
// 1E0A;1E0A;0044 0307;1E0A;0044 0307; # (Ḋ; Ḋ; D◌̇; Ḋ; D◌̇; ) LATIN CAPITAL LETTER D WITH DOT ABOVE
// 1E0C;1E0C;0044 0323;1E0C;0044 0323; # (Ḍ; Ḍ; D◌̣; Ḍ; D◌̣; ) LATIN CAPITAL LETTER D WITH DOT BELOW
//
// Each test has 5 columns (c1, c2, c3, c4, c5), where
// (c1, c2, c3, c4, c5) == (c1, NFC(c1), NFD(c1), NFKC(c1), NFKD(c1))
//
// CONFORMANCE:
// 1. The following invariants must be true for all conformant implementations
//
//    NFC
//      c2 ==  NFC(c1) ==  NFC(c2) ==  NFC(c3)
//      c4 ==  NFC(c4) ==  NFC(c5)
//
//    NFD
//      c3 ==  NFD(c1) ==  NFD(c2) ==  NFD(c3)
//      c5 ==  NFD(c4) ==  NFD(c5)
//
//    NFKC
//      c4 == NFKC(c1) == NFKC(c2) == NFKC(c3) == NFKC(c4) == NFKC(c5)
//
//    NFKD
//      c5 == NFKD(c1) == NFKD(c2) == NFKD(c3) == NFKD(c4) == NFKD(c5)
//
// 2. For every code point X assigned in this version of Unicode that is not
//    specifically listed in Part 1, the following invariants must be true
//    for all conformant implementations:
//
//      X == NFC(X) == NFD(X) == NFKC(X) == NFKD(X)
//

// Column types.
const (
	cRaw = iota
	cNFC
	cNFD
	cNFKC
	cNFKD
	cMaxColumns
)

// Holds data from NormalizationTest.txt
var part []Part

type Part struct {
	name   string
	number int
	tests  []Test
}

type Test struct {
	name   string
	partnr int
	number int
	r      rune                // used for character by character test
	cols   [cMaxColumns]string // Each has 5 entries, see below.
}

func (t Test) Name() string {
	if t.number < 0 {
		return part[t.partnr].name
	}
	return fmt.Sprintf("%s:%d", part[t.partnr].name, t.number)
}

var partRe = regexp.MustCompile(`@Part(\d) # (.*)$`)
var testRe = regexp.MustCompile(`^` + strings.Repeat(`([\dA-F ]+);`, 5) + ` # (.*)$`)

var counter int

// Load the data form NormalizationTest.txt
func loadTestData(t *testing.T) {
	f := gen.OpenUCDFile("NormalizationTest.txt")
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		m := partRe.FindStringSubmatch(line)
		if m != nil {
			if len(m) < 3 {
				t.Fatal("Failed to parse Part: ", line)
			}
			i, err := strconv.Atoi(m[1])
			if err != nil {
				t.Fatal(err)
			}
			name := m[2]
			part = append(part, Part{name: name[:len(name)-1], number: i})
			continue
		}
		m = testRe.FindStringSubmatch(line)
		if m == nil || len(m) < 7 {
			t.Fatalf(`Failed to parse: "%s" result: %#v`, line, m)
		}
		test := Test{name: m[6], partnr: len(part) - 1, number: counter}
		counter++
		for j := 1; j < len(m)-1; j++ {
			for _, split := range strings.Split(m[j], " ") {
				r, err := strconv.ParseUint(split, 16, 64)
				if err != nil {
					t.Fatal(err)
				}
				if test.r == 0 {
					// save for CharacterByCharacterTests
					test.r = rune(r)
				}
				var buf [utf8.UTFMax]byte
				sz := utf8.EncodeRune(buf[:], rune(r))
				test.cols[j-1] += string(buf[:sz])
			}
		}
		part := &part[len(part)-1]
		part.tests = append(part.tests, test)
	}
	if scanner.Err() != nil {
		t.Fatal(scanner.Err())
	}
}

func cmpResult(t *testing.T, tc *Test, name string, f Form, gold, test, result string) {
	if gold != result {
		t.Errorf("%s:%s: %s(%+q)=%+q; want %+q: %s",
			tc.Name(), name, fstr[f], test, result, gold, tc.name)
	}
}

func cmpIsNormal(t *testing.T, tc *Test, name string, f Form, test string, result, want bool) {
	if result != want {
		t.Errorf("%s:%s: %s(%+q)=%v; want %v", tc.Name(), name, fstr[f], test, result, want)
	}
}

func doTest(t *testing.T, tc *Test, f Form, gold, test string) {
	testb := []byte(test)
	result := f.Bytes(testb)
	cmpResult(t, tc, "Bytes", f, gold, test, string(result))

	sresult := f.String(test)
	cmpResult(t, tc, "String", f, gold, test, sresult)

	acc := []byte{}
	i := Iter{}
	i.InitString(f, test)
	for !i.Done() {
		acc = append(acc, i.Next()...)
	}
	cmpResult(t, tc, "Iter.Next", f, gold, test, string(acc))

	buf := make([]byte, 128)
	acc = nil
	for p := 0; p < len(testb); {
		nDst, nSrc, _ := f.Transform(buf, testb[p:], true)
		acc = append(acc, buf[:nDst]...)
		p += nSrc
	}
	cmpResult(t, tc, "Transform", f, gold, test, string(acc))

	for i := range test {
		out := f.Append(f.Bytes([]byte(test[:i])), []byte(test[i:])...)
		cmpResult(t, tc, fmt.Sprintf(":Append:%d", i), f, gold, test, string(out))
	}
	cmpIsNormal(t, tc, "IsNormal", f, test, f.IsNormal([]byte(test)), test == gold)
	cmpIsNormal(t, tc, "IsNormalString", f, test, f.IsNormalString(test), test == gold)
}

func doConformanceTests(t *testing.T, tc *Test, partn int) {
	for i := 0; i <= 2; i++ {
		doTest(t, tc, NFC, tc.cols[1], tc.cols[i])
		doTest(t, tc, NFD, tc.cols[2], tc.cols[i])
		doTest(t, tc, NFKC, tc.cols[3], tc.cols[i])
		doTest(t, tc, NFKD, tc.cols[4], tc.cols[i])
	}
	for i := 3; i <= 4; i++ {
		doTest(t, tc, NFC, tc.cols[3], tc.cols[i])
		doTest(t, tc, NFD, tc.cols[4], tc.cols[i])
		doTest(t, tc, NFKC, tc.cols[3], tc.cols[i])
		doTest(t, tc, NFKD, tc.cols[4], tc.cols[i])
	}
}

func TestCharacterByCharacter(t *testing.T) {
	skipShort(t)
	tests := part[1].tests
	var last rune = 0
	for i := 0; i <= len(tests); i++ { // last one is special case
		var r rune
		if i == len(tests) {
			r = 0x2FA1E // Don't have to go to 0x10FFFF
		} else {
			r = tests[i].r
		}
		for last++; last < r; last++ {
			// Check all characters that were not explicitly listed in the test.
			tc := &Test{partnr: 1, number: -1}
			char := string(last)
			doTest(t, tc, NFC, char, char)
			doTest(t, tc, NFD, char, char)
			doTest(t, tc, NFKC, char, char)
			doTest(t, tc, NFKD, char, char)
		}
		if i < len(tests) {
			doConformanceTests(t, &tests[i], 1)
		}
	}
}

func TestStandardTests(t *testing.T) {
	skipShort(t)
	for _, j := range []int{0, 2, 3} {
		for _, test := range part[j].tests {
			doConformanceTests(t, &test, j)
		}
	}
}

// TestPerformance verifies that normalization is O(n). If any of the
// code does not properly check for maxCombiningChars, normalization
// may exhibit O(n**2) behavior.
func TestPerformance(t *testing.T) {
	skipShort(t)
	runtime.GOMAXPROCS(2)
	success := make(chan bool, 1)
	go func() {
		buf := bytes.Repeat([]byte("\u035D"), 1024*1024)
		buf = append(buf, "\u035B"...)
		NFC.Append(nil, buf...)
		success <- true
	}()
	timeout := time.After(1 * time.Second)
	select {
	case <-success:
		// test completed before the timeout
	case <-timeout:
		t.Errorf(`unexpectedly long time to complete PerformanceTest`)
	}
}
