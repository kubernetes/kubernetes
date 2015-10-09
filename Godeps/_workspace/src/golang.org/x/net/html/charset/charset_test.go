// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charset

import (
	"bytes"
	"encoding/xml"
	"io/ioutil"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/text/transform"
)

func transformString(t transform.Transformer, s string) (string, error) {
	r := transform.NewReader(strings.NewReader(s), t)
	b, err := ioutil.ReadAll(r)
	return string(b), err
}

var testCases = []struct {
	utf8, other, otherEncoding string
}{
	{"Résumé", "Résumé", "utf8"},
	{"Résumé", "R\xe9sum\xe9", "latin1"},
	{"これは漢字です。", "S0\x8c0o0\"oW[g0Y0\x020", "UTF-16LE"},
	{"これは漢字です。", "0S0\x8c0oo\"[W0g0Y0\x02", "UTF-16BE"},
	{"Hello, world", "Hello, world", "ASCII"},
	{"Gdańsk", "Gda\xf1sk", "ISO-8859-2"},
	{"Ââ Čč Đđ Ŋŋ Õõ Šš Žž Åå Ää", "\xc2\xe2 \xc8\xe8 \xa9\xb9 \xaf\xbf \xd5\xf5 \xaa\xba \xac\xbc \xc5\xe5 \xc4\xe4", "ISO-8859-10"},
	{"สำหรับ", "\xca\xd3\xcb\xc3\u047a", "ISO-8859-11"},
	{"latviešu", "latvie\xf0u", "ISO-8859-13"},
	{"Seònaid", "Se\xf2naid", "ISO-8859-14"},
	{"€1 is cheap", "\xa41 is cheap", "ISO-8859-15"},
	{"românește", "rom\xe2ne\xbate", "ISO-8859-16"},
	{"nutraĵo", "nutra\xbco", "ISO-8859-3"},
	{"Kalâdlit", "Kal\xe2dlit", "ISO-8859-4"},
	{"русский", "\xe0\xe3\xe1\xe1\xda\xd8\xd9", "ISO-8859-5"},
	{"ελληνικά", "\xe5\xeb\xeb\xe7\xed\xe9\xea\xdc", "ISO-8859-7"},
	{"Kağan", "Ka\xf0an", "ISO-8859-9"},
	{"Résumé", "R\x8esum\x8e", "macintosh"},
	{"Gdańsk", "Gda\xf1sk", "windows-1250"},
	{"русский", "\xf0\xf3\xf1\xf1\xea\xe8\xe9", "windows-1251"},
	{"Résumé", "R\xe9sum\xe9", "windows-1252"},
	{"ελληνικά", "\xe5\xeb\xeb\xe7\xed\xe9\xea\xdc", "windows-1253"},
	{"Kağan", "Ka\xf0an", "windows-1254"},
	{"עִבְרִית", "\xf2\xc4\xe1\xc0\xf8\xc4\xe9\xfa", "windows-1255"},
	{"العربية", "\xc7\xe1\xda\xd1\xc8\xed\xc9", "windows-1256"},
	{"latviešu", "latvie\xf0u", "windows-1257"},
	{"Việt", "Vi\xea\xf2t", "windows-1258"},
	{"สำหรับ", "\xca\xd3\xcb\xc3\u047a", "windows-874"},
	{"русский", "\xd2\xd5\xd3\xd3\xcb\xc9\xca", "KOI8-R"},
	{"українська", "\xd5\xcb\xd2\xc1\xa7\xce\xd3\xd8\xcb\xc1", "KOI8-U"},
	{"Hello 常用國字標準字體表", "Hello \xb1`\xa5\u03b0\xea\xa6r\xbc\u0437\u01e6r\xc5\xe9\xaa\xed", "big5"},
	{"Hello 常用國字標準字體表", "Hello \xb3\xa3\xd3\xc3\x87\xf8\xd7\xd6\x98\xcb\x9c\xca\xd7\xd6\xf3\x77\xb1\xed", "gbk"},
	{"Hello 常用國字標準字體表", "Hello \xb3\xa3\xd3\xc3\x87\xf8\xd7\xd6\x98\xcb\x9c\xca\xd7\xd6\xf3\x77\xb1\xed", "gb18030"},
	{"עִבְרִית", "\x81\x30\xfb\x30\x81\x30\xf6\x34\x81\x30\xf9\x33\x81\x30\xf6\x30\x81\x30\xfb\x36\x81\x30\xf6\x34\x81\x30\xfa\x31\x81\x30\xfb\x38", "gb18030"},
	{"㧯", "\x82\x31\x89\x38", "gb18030"},
	{"これは漢字です。", "\x82\xb1\x82\xea\x82\xcd\x8a\xbf\x8e\x9a\x82\xc5\x82\xb7\x81B", "SJIS"},
	{"Hello, 世界!", "Hello, \x90\xa2\x8aE!", "SJIS"},
	{"ｲｳｴｵｶ", "\xb2\xb3\xb4\xb5\xb6", "SJIS"},
	{"これは漢字です。", "\xa4\xb3\xa4\xec\xa4\u03f4\xc1\xbb\xfa\xa4\u01e4\xb9\xa1\xa3", "EUC-JP"},
	{"Hello, 世界!", "Hello, \x1b$B@$3&\x1b(B!", "ISO-2022-JP"},
	{"네이트 | 즐거움의 시작, 슈파스(Spaβ) NATE", "\xb3\xd7\xc0\xcc\xc6\xae | \xc1\xf1\xb0\xc5\xbf\xf2\xc0\xc7 \xbd\xc3\xc0\xdb, \xbd\xb4\xc6\xc4\xbd\xba(Spa\xa5\xe2) NATE", "EUC-KR"},
}

func TestDecode(t *testing.T) {
	for _, tc := range testCases {
		e, _ := Lookup(tc.otherEncoding)
		if e == nil {
			t.Errorf("%s: not found", tc.otherEncoding)
			continue
		}
		s, err := transformString(e.NewDecoder(), tc.other)
		if err != nil {
			t.Errorf("%s: decode %q: %v", tc.otherEncoding, tc.other, err)
			continue
		}
		if s != tc.utf8 {
			t.Errorf("%s: got %q, want %q", tc.otherEncoding, s, tc.utf8)
		}
	}
}

func TestEncode(t *testing.T) {
	for _, tc := range testCases {
		e, _ := Lookup(tc.otherEncoding)
		if e == nil {
			t.Errorf("%s: not found", tc.otherEncoding)
			continue
		}
		s, err := transformString(e.NewEncoder(), tc.utf8)
		if err != nil {
			t.Errorf("%s: encode %q: %s", tc.otherEncoding, tc.utf8, err)
			continue
		}
		if s != tc.other {
			t.Errorf("%s: got %q, want %q", tc.otherEncoding, s, tc.other)
		}
	}
}

// TestNames verifies that you can pass an encoding's name to Lookup and get
// the same encoding back (except for "replacement").
func TestNames(t *testing.T) {
	for _, e := range encodings {
		if e.name == "replacement" {
			continue
		}
		_, got := Lookup(e.name)
		if got != e.name {
			t.Errorf("got %q, want %q", got, e.name)
			continue
		}
	}
}

var sniffTestCases = []struct {
	filename, declared, want string
}{
	{"HTTP-charset.html", "text/html; charset=iso-8859-15", "iso-8859-15"},
	{"UTF-16LE-BOM.html", "", "utf-16le"},
	{"UTF-16BE-BOM.html", "", "utf-16be"},
	{"meta-content-attribute.html", "text/html", "iso-8859-15"},
	{"meta-charset-attribute.html", "text/html", "iso-8859-15"},
	{"No-encoding-declaration.html", "text/html", "utf-8"},
	{"HTTP-vs-UTF-8-BOM.html", "text/html; charset=iso-8859-15", "utf-8"},
	{"HTTP-vs-meta-content.html", "text/html; charset=iso-8859-15", "iso-8859-15"},
	{"HTTP-vs-meta-charset.html", "text/html; charset=iso-8859-15", "iso-8859-15"},
	{"UTF-8-BOM-vs-meta-content.html", "text/html", "utf-8"},
	{"UTF-8-BOM-vs-meta-charset.html", "text/html", "utf-8"},
}

func TestSniff(t *testing.T) {
	switch runtime.GOOS {
	case "nacl": // platforms that don't permit direct file system access
		t.Skipf("not supported on %q", runtime.GOOS)
	}

	for _, tc := range sniffTestCases {
		content, err := ioutil.ReadFile("testdata/" + tc.filename)
		if err != nil {
			t.Errorf("%s: error reading file: %v", tc.filename, err)
			continue
		}

		_, name, _ := DetermineEncoding(content, tc.declared)
		if name != tc.want {
			t.Errorf("%s: got %q, want %q", tc.filename, name, tc.want)
			continue
		}
	}
}

func TestReader(t *testing.T) {
	switch runtime.GOOS {
	case "nacl": // platforms that don't permit direct file system access
		t.Skipf("not supported on %q", runtime.GOOS)
	}

	for _, tc := range sniffTestCases {
		content, err := ioutil.ReadFile("testdata/" + tc.filename)
		if err != nil {
			t.Errorf("%s: error reading file: %v", tc.filename, err)
			continue
		}

		r, err := NewReader(bytes.NewReader(content), tc.declared)
		if err != nil {
			t.Errorf("%s: error creating reader: %v", tc.filename, err)
			continue
		}

		got, err := ioutil.ReadAll(r)
		if err != nil {
			t.Errorf("%s: error reading from charset.NewReader: %v", tc.filename, err)
			continue
		}

		e, _ := Lookup(tc.want)
		want, err := ioutil.ReadAll(transform.NewReader(bytes.NewReader(content), e.NewDecoder()))
		if err != nil {
			t.Errorf("%s: error decoding with hard-coded charset name: %v", tc.filename, err)
			continue
		}

		if !bytes.Equal(got, want) {
			t.Errorf("%s: got %q, want %q", tc.filename, got, want)
			continue
		}
	}
}

var metaTestCases = []struct {
	meta, want string
}{
	{"", ""},
	{"text/html", ""},
	{"text/html; charset utf-8", ""},
	{"text/html; charset=latin-2", "latin-2"},
	{"text/html; charset; charset = utf-8", "utf-8"},
	{`charset="big5"`, "big5"},
	{"charset='shift_jis'", "shift_jis"},
}

func TestFromMeta(t *testing.T) {
	for _, tc := range metaTestCases {
		got := fromMetaElement(tc.meta)
		if got != tc.want {
			t.Errorf("%q: got %q, want %q", tc.meta, got, tc.want)
		}
	}
}

func TestXML(t *testing.T) {
	const s = "<?xml version=\"1.0\" encoding=\"windows-1252\"?><a><Word>r\xe9sum\xe9</Word></a>"

	d := xml.NewDecoder(strings.NewReader(s))
	d.CharsetReader = NewReaderLabel

	var a struct {
		Word string
	}
	err := d.Decode(&a)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	want := "résumé"
	if a.Word != want {
		t.Errorf("got %q, want %q", a.Word, want)
	}
}
