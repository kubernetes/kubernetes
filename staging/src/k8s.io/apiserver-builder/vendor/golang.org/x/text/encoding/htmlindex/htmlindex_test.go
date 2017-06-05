// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package htmlindex

import (
	"testing"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/internal/identifier"
	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/language"
)

func TestGet(t *testing.T) {
	for i, tc := range []struct {
		name      string
		canonical string
		err       error
	}{
		{"utf-8", "utf-8", nil},
		{"  utf-8  ", "utf-8", nil},
		{"  l5  ", "windows-1254", nil},
		{"latin5 ", "windows-1254", nil},
		{"latin 5", "", errInvalidName},
		{"latin-5", "", errInvalidName},
	} {
		enc, err := Get(tc.name)
		if err != tc.err {
			t.Errorf("%d: error was %v; want %v", i, err, tc.err)
		}
		if err != nil {
			continue
		}
		if got, err := Name(enc); got != tc.canonical {
			t.Errorf("%d: Name(Get(%q)) = %q; want %q (%v)", i, tc.name, got, tc.canonical, err)
		}
	}
}

func TestTables(t *testing.T) {
	for name, index := range nameMap {
		got, err := Get(name)
		if err != nil {
			t.Errorf("%s:err: expected non-nil error", name)
		}
		if want := encodings[index]; got != want {
			t.Errorf("%s:encoding: got %v; want %v", name, got, want)
		}
		mib, _ := got.(identifier.Interface).ID()
		if mibMap[mib] != index {
			t.Errorf("%s:mibMab: got %d; want %d", name, mibMap[mib], index)
		}
	}
}

func TestName(t *testing.T) {
	for i, tc := range []struct {
		desc string
		enc  encoding.Encoding
		name string
		err  error
	}{{
		"defined encoding",
		charmap.ISO8859_2,
		"iso-8859-2",
		nil,
	}, {
		"defined Unicode encoding",
		unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM),
		"utf-16be",
		nil,
	}, {
		"undefined Unicode encoding in HTML standard",
		unicode.UTF16(unicode.BigEndian, unicode.UseBOM),
		"",
		errUnsupported,
	}, {
		"undefined other encoding in HTML standard",
		charmap.CodePage437,
		"",
		errUnsupported,
	}, {
		"unknown encoding",
		encoding.Nop,
		"",
		errUnknown,
	}} {
		name, err := Name(tc.enc)
		if name != tc.name || err != tc.err {
			t.Errorf("%d:%s: got %q, %v; want %q, %v", i, tc.desc, name, err, tc.name, tc.err)
		}
	}
}

func TestLanguageDefault(t *testing.T) {
	for _, tc := range []struct{ tag, want string }{
		{"und", "windows-1252"}, // The default value.
		{"ar", "windows-1256"},
		{"ba", "windows-1251"},
		{"be", "windows-1251"},
		{"bg", "windows-1251"},
		{"cs", "windows-1250"},
		{"el", "iso-8859-7"},
		{"et", "windows-1257"},
		{"fa", "windows-1256"},
		{"he", "windows-1255"},
		{"hr", "windows-1250"},
		{"hu", "iso-8859-2"},
		{"ja", "shift_jis"},
		{"kk", "windows-1251"},
		{"ko", "euc-kr"},
		{"ku", "windows-1254"},
		{"ky", "windows-1251"},
		{"lt", "windows-1257"},
		{"lv", "windows-1257"},
		{"mk", "windows-1251"},
		{"pl", "iso-8859-2"},
		{"ru", "windows-1251"},
		{"sah", "windows-1251"},
		{"sk", "windows-1250"},
		{"sl", "iso-8859-2"},
		{"sr", "windows-1251"},
		{"tg", "windows-1251"},
		{"th", "windows-874"},
		{"tr", "windows-1254"},
		{"tt", "windows-1251"},
		{"uk", "windows-1251"},
		{"vi", "windows-1258"},
		{"zh-hans", "gb18030"},
		{"zh-hant", "big5"},
		// Variants and close approximates of the above.
		{"ar_EG", "windows-1256"},
		{"bs", "windows-1250"}, // Bosnian Latin maps to Croatian.
		// Use default fallback in case of miss.
		{"nl", "windows-1252"},
	} {
		if got := LanguageDefault(language.MustParse(tc.tag)); got != tc.want {
			t.Errorf("LanguageDefault(%s) = %s; want %s", tc.tag, got, tc.want)
		}
	}
}
