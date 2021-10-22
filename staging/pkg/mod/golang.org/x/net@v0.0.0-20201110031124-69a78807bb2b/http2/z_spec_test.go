// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
)

var coverSpec = flag.Bool("coverspec", false, "Run spec coverage tests")

// The global map of sentence coverage for the http2 spec.
var defaultSpecCoverage specCoverage

var loadSpecOnce sync.Once

func loadSpec() {
	if f, err := os.Open("testdata/draft-ietf-httpbis-http2.xml"); err != nil {
		panic(err)
	} else {
		defaultSpecCoverage = readSpecCov(f)
		f.Close()
	}
}

// covers marks all sentences for section sec in defaultSpecCoverage. Sentences not
// "covered" will be included in report outputted by TestSpecCoverage.
func covers(sec, sentences string) {
	loadSpecOnce.Do(loadSpec)
	defaultSpecCoverage.cover(sec, sentences)
}

type specPart struct {
	section  string
	sentence string
}

func (ss specPart) Less(oo specPart) bool {
	atoi := func(s string) int {
		n, err := strconv.Atoi(s)
		if err != nil {
			panic(err)
		}
		return n
	}
	a := strings.Split(ss.section, ".")
	b := strings.Split(oo.section, ".")
	for len(a) > 0 {
		if len(b) == 0 {
			return false
		}
		x, y := atoi(a[0]), atoi(b[0])
		if x == y {
			a, b = a[1:], b[1:]
			continue
		}
		return x < y
	}
	if len(b) > 0 {
		return true
	}
	return false
}

type bySpecSection []specPart

func (a bySpecSection) Len() int           { return len(a) }
func (a bySpecSection) Less(i, j int) bool { return a[i].Less(a[j]) }
func (a bySpecSection) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

type specCoverage struct {
	coverage map[specPart]bool
	d        *xml.Decoder
}

func joinSection(sec []int) string {
	s := fmt.Sprintf("%d", sec[0])
	for _, n := range sec[1:] {
		s = fmt.Sprintf("%s.%d", s, n)
	}
	return s
}

func (sc specCoverage) readSection(sec []int) {
	var (
		buf = new(bytes.Buffer)
		sub = 0
	)
	for {
		tk, err := sc.d.Token()
		if err != nil {
			if err == io.EOF {
				return
			}
			panic(err)
		}
		switch v := tk.(type) {
		case xml.StartElement:
			if skipElement(v) {
				if err := sc.d.Skip(); err != nil {
					panic(err)
				}
				if v.Name.Local == "section" {
					sub++
				}
				break
			}
			switch v.Name.Local {
			case "section":
				sub++
				sc.readSection(append(sec, sub))
			case "xref":
				buf.Write(sc.readXRef(v))
			}
		case xml.CharData:
			if len(sec) == 0 {
				break
			}
			buf.Write(v)
		case xml.EndElement:
			if v.Name.Local == "section" {
				sc.addSentences(joinSection(sec), buf.String())
				return
			}
		}
	}
}

func (sc specCoverage) readXRef(se xml.StartElement) []byte {
	var b []byte
	for {
		tk, err := sc.d.Token()
		if err != nil {
			panic(err)
		}
		switch v := tk.(type) {
		case xml.CharData:
			if b != nil {
				panic("unexpected CharData")
			}
			b = []byte(string(v))
		case xml.EndElement:
			if v.Name.Local != "xref" {
				panic("expected </xref>")
			}
			if b != nil {
				return b
			}
			sig := attrSig(se)
			switch sig {
			case "target":
				return []byte(fmt.Sprintf("[%s]", attrValue(se, "target")))
			case "fmt-of,rel,target", "fmt-,,rel,target":
				return []byte(fmt.Sprintf("[%s, %s]", attrValue(se, "target"), attrValue(se, "rel")))
			case "fmt-of,sec,target", "fmt-,,sec,target":
				return []byte(fmt.Sprintf("[section %s of %s]", attrValue(se, "sec"), attrValue(se, "target")))
			case "fmt-of,rel,sec,target":
				return []byte(fmt.Sprintf("[section %s of %s, %s]", attrValue(se, "sec"), attrValue(se, "target"), attrValue(se, "rel")))
			default:
				panic(fmt.Sprintf("unknown attribute signature %q in %#v", sig, fmt.Sprintf("%#v", se)))
			}
		default:
			panic(fmt.Sprintf("unexpected tag %q", v))
		}
	}
}

var skipAnchor = map[string]bool{
	"intro":    true,
	"Overview": true,
}

var skipTitle = map[string]bool{
	"Acknowledgements":            true,
	"Change Log":                  true,
	"Document Organization":       true,
	"Conventions and Terminology": true,
}

func skipElement(s xml.StartElement) bool {
	switch s.Name.Local {
	case "artwork":
		return true
	case "section":
		for _, attr := range s.Attr {
			switch attr.Name.Local {
			case "anchor":
				if skipAnchor[attr.Value] || strings.HasPrefix(attr.Value, "changes.since.") {
					return true
				}
			case "title":
				if skipTitle[attr.Value] {
					return true
				}
			}
		}
	}
	return false
}

func readSpecCov(r io.Reader) specCoverage {
	sc := specCoverage{
		coverage: map[specPart]bool{},
		d:        xml.NewDecoder(r)}
	sc.readSection(nil)
	return sc
}

func (sc specCoverage) addSentences(sec string, sentence string) {
	for _, s := range parseSentences(sentence) {
		sc.coverage[specPart{sec, s}] = false
	}
}

func (sc specCoverage) cover(sec string, sentence string) {
	for _, s := range parseSentences(sentence) {
		p := specPart{sec, s}
		if _, ok := sc.coverage[p]; !ok {
			panic(fmt.Sprintf("Not found in spec: %q, %q", sec, s))
		}
		sc.coverage[specPart{sec, s}] = true
	}

}

var whitespaceRx = regexp.MustCompile(`\s+`)

func parseSentences(sens string) []string {
	sens = strings.TrimSpace(sens)
	if sens == "" {
		return nil
	}
	ss := strings.Split(whitespaceRx.ReplaceAllString(sens, " "), ". ")
	for i, s := range ss {
		s = strings.TrimSpace(s)
		if !strings.HasSuffix(s, ".") {
			s += "."
		}
		ss[i] = s
	}
	return ss
}

func TestSpecParseSentences(t *testing.T) {
	tests := []struct {
		ss   string
		want []string
	}{
		{"Sentence 1. Sentence 2.",
			[]string{
				"Sentence 1.",
				"Sentence 2.",
			}},
		{"Sentence 1.  \nSentence 2.\tSentence 3.",
			[]string{
				"Sentence 1.",
				"Sentence 2.",
				"Sentence 3.",
			}},
	}

	for i, tt := range tests {
		got := parseSentences(tt.ss)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%d: got = %q, want %q", i, got, tt.want)
		}
	}
}

func TestSpecCoverage(t *testing.T) {
	if !*coverSpec {
		t.Skip()
	}

	loadSpecOnce.Do(loadSpec)

	var (
		list     []specPart
		cv       = defaultSpecCoverage.coverage
		total    = len(cv)
		complete = 0
	)

	for sp, touched := range defaultSpecCoverage.coverage {
		if touched {
			complete++
		} else {
			list = append(list, sp)
		}
	}
	sort.Stable(bySpecSection(list))

	if testing.Short() && len(list) > 5 {
		list = list[:5]
	}

	for _, p := range list {
		t.Errorf("\tSECTION %s: %s", p.section, p.sentence)
	}

	t.Logf("%d/%d (%d%%) sentences covered", complete, total, (complete/total)*100)
}

func attrSig(se xml.StartElement) string {
	var names []string
	for _, attr := range se.Attr {
		if attr.Name.Local == "fmt" {
			names = append(names, "fmt-"+attr.Value)
		} else {
			names = append(names, attr.Name.Local)
		}
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}

func attrValue(se xml.StartElement, attr string) string {
	for _, a := range se.Attr {
		if a.Name.Local == attr {
			return a.Value
		}
	}
	panic("unknown attribute " + attr)
}

func TestSpecPartLess(t *testing.T) {
	tests := []struct {
		sec1, sec2 string
		want       bool
	}{
		{"6.2.1", "6.2", false},
		{"6.2", "6.2.1", true},
		{"6.10", "6.10.1", true},
		{"6.10", "6.1.1", false}, // 10, not 1
		{"6.1", "6.1", false},    // equal, so not less
	}
	for _, tt := range tests {
		got := (specPart{tt.sec1, "foo"}).Less(specPart{tt.sec2, "foo"})
		if got != tt.want {
			t.Errorf("Less(%q, %q) = %v; want %v", tt.sec1, tt.sec2, got, tt.want)
		}
	}
}
