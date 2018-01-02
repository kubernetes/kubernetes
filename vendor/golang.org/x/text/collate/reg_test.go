// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package collate

import (
	"archive/zip"
	"bufio"
	"bytes"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"path"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"golang.org/x/text/collate/build"
	"golang.org/x/text/internal/gen"
	"golang.org/x/text/language"
)

var long = flag.Bool("long", false,
	"run time-consuming tests, such as tests that fetch data online")

// This regression test runs tests for the test files in CollationTest.zip
// (taken from http://www.unicode.org/Public/UCA/<gen.UnicodeVersion()>/).
//
// The test files have the following form:
// # header
// 0009 0021;	# ('\u0009') <CHARACTER TABULATION>	[| | | 0201 025E]
// 0009 003F;	# ('\u0009') <CHARACTER TABULATION>	[| | | 0201 0263]
// 000A 0021;	# ('\u000A') <LINE FEED (LF)>	[| | | 0202 025E]
// 000A 003F;	# ('\u000A') <LINE FEED (LF)>	[| | | 0202 0263]
//
// The part before the semicolon is the hex representation of a sequence
// of runes. After the hash mark is a comment. The strings
// represented by rune sequence are in the file in sorted order, as
// defined by the DUCET.

type Test struct {
	name    string
	str     [][]byte
	comment []string
}

var versionRe = regexp.MustCompile(`# UCA Version: (.*)\n?$`)
var testRe = regexp.MustCompile(`^([\dA-F ]+);.*# (.*)\n?$`)

func TestCollation(t *testing.T) {
	if !gen.IsLocal() && !*long {
		t.Skip("skipping test to prevent downloading; to run use -long or use -local to specify a local source")
	}
	t.Skip("must first update to new file format to support test")
	for _, test := range loadTestData() {
		doTest(t, test)
	}
}

func Error(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

// parseUCA parses a Default Unicode Collation Element Table of the format
// specified in http://www.unicode.org/reports/tr10/#File_Format.
// It returns the variable top.
func parseUCA(builder *build.Builder) {
	r := gen.OpenUnicodeFile("UCA", "", "allkeys.txt")
	defer r.Close()
	input := bufio.NewReader(r)
	colelem := regexp.MustCompile(`\[([.*])([0-9A-F.]+)\]`)
	for i := 1; true; i++ {
		l, prefix, err := input.ReadLine()
		if err == io.EOF {
			break
		}
		Error(err)
		line := string(l)
		if prefix {
			log.Fatalf("%d: buffer overflow", i)
		}
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		if line[0] == '@' {
			if strings.HasPrefix(line[1:], "version ") {
				if v := strings.Split(line[1:], " ")[1]; v != gen.UnicodeVersion() {
					log.Fatalf("incompatible version %s; want %s", v, gen.UnicodeVersion())
				}
			}
		} else {
			// parse entries
			part := strings.Split(line, " ; ")
			if len(part) != 2 {
				log.Fatalf("%d: production rule without ';': %v", i, line)
			}
			lhs := []rune{}
			for _, v := range strings.Split(part[0], " ") {
				if v != "" {
					lhs = append(lhs, rune(convHex(i, v)))
				}
			}
			vars := []int{}
			rhs := [][]int{}
			for i, m := range colelem.FindAllStringSubmatch(part[1], -1) {
				if m[1] == "*" {
					vars = append(vars, i)
				}
				elem := []int{}
				for _, h := range strings.Split(m[2], ".") {
					elem = append(elem, convHex(i, h))
				}
				rhs = append(rhs, elem)
			}
			builder.Add(lhs, rhs, vars)
		}
	}
}

func convHex(line int, s string) int {
	r, e := strconv.ParseInt(s, 16, 32)
	if e != nil {
		log.Fatalf("%d: %v", line, e)
	}
	return int(r)
}

func loadTestData() []Test {
	f := gen.OpenUnicodeFile("UCA", "", "CollationTest.zip")
	buffer, err := ioutil.ReadAll(f)
	f.Close()
	Error(err)
	archive, err := zip.NewReader(bytes.NewReader(buffer), int64(len(buffer)))
	Error(err)
	tests := []Test{}
	for _, f := range archive.File {
		// Skip the short versions, which are simply duplicates of the long versions.
		if strings.Contains(f.Name, "SHORT") || f.FileInfo().IsDir() {
			continue
		}
		ff, err := f.Open()
		Error(err)
		defer ff.Close()
		scanner := bufio.NewScanner(ff)
		test := Test{name: path.Base(f.Name)}
		for scanner.Scan() {
			line := scanner.Text()
			if len(line) <= 1 || line[0] == '#' {
				if m := versionRe.FindStringSubmatch(line); m != nil {
					if m[1] != gen.UnicodeVersion() {
						log.Printf("warning:%s: version is %s; want %s", f.Name, m[1], gen.UnicodeVersion())
					}
				}
				continue
			}
			m := testRe.FindStringSubmatch(line)
			if m == nil || len(m) < 3 {
				log.Fatalf(`Failed to parse: "%s" result: %#v`, line, m)
			}
			str := []byte{}
			// In the regression test data (unpaired) surrogates are assigned a weight
			// corresponding to their code point value.  However, utf8.DecodeRune,
			// which is used to compute the implicit weight, assigns FFFD to surrogates.
			// We therefore skip tests with surrogates.  This skips about 35 entries
			// per test.
			valid := true
			for _, split := range strings.Split(m[1], " ") {
				r, err := strconv.ParseUint(split, 16, 64)
				Error(err)
				valid = valid && utf8.ValidRune(rune(r))
				str = append(str, string(rune(r))...)
			}
			if valid {
				test.str = append(test.str, str)
				test.comment = append(test.comment, m[2])
			}
		}
		if scanner.Err() != nil {
			log.Fatal(scanner.Err())
		}
		tests = append(tests, test)
	}
	return tests
}

var errorCount int

func runes(b []byte) []rune {
	return []rune(string(b))
}

var shifted = language.MustParse("und-u-ka-shifted-ks-level4")

func doTest(t *testing.T, tc Test) {
	bld := build.NewBuilder()
	parseUCA(bld)
	w, err := bld.Build()
	Error(err)
	var tag language.Tag
	if !strings.Contains(tc.name, "NON_IGNOR") {
		tag = shifted
	}
	c := NewFromTable(w, OptionsFromTag(tag))
	b := &Buffer{}
	prev := tc.str[0]
	for i := 1; i < len(tc.str); i++ {
		b.Reset()
		s := tc.str[i]
		ka := c.Key(b, prev)
		kb := c.Key(b, s)
		if r := bytes.Compare(ka, kb); r == 1 {
			t.Errorf("%s:%d: Key(%.4X) < Key(%.4X) (%X < %X) == %d; want -1 or 0", tc.name, i, []rune(string(prev)), []rune(string(s)), ka, kb, r)
			prev = s
			continue
		}
		if r := c.Compare(prev, s); r == 1 {
			t.Errorf("%s:%d: Compare(%.4X, %.4X) == %d; want -1 or 0", tc.name, i, runes(prev), runes(s), r)
		}
		if r := c.Compare(s, prev); r == -1 {
			t.Errorf("%s:%d: Compare(%.4X, %.4X) == %d; want 1 or 0", tc.name, i, runes(s), runes(prev), r)
		}
		prev = s
	}
}
