// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"golang.org/x/text/internal"
	"golang.org/x/text/internal/format"
	"golang.org/x/text/language"
	"golang.org/x/text/message/catalog"
)

type formatFunc func(s fmt.State, v rune)

func (f formatFunc) Format(s fmt.State, v rune) { f(s, v) }

func TestBinding(t *testing.T) {
	testCases := []struct {
		tag   string
		value interface{}
		want  string
	}{
		{"en", 1, "1"},
		{"en", "2", "2"},
		{ // Language is passed.
			"en",
			formatFunc(func(fs fmt.State, v rune) {
				s := fs.(format.State)
				io.WriteString(s, s.Language().String())
			}),
			"en",
		},
	}
	for i, tc := range testCases {
		p := NewPrinter(language.MustParse(tc.tag))
		if got := p.Sprint(tc.value); got != tc.want {
			t.Errorf("%d:%s:Sprint(%v) = %q; want %q", i, tc.tag, tc.value, got, tc.want)
		}
		var buf bytes.Buffer
		p.Fprint(&buf, tc.value)
		if got := buf.String(); got != tc.want {
			t.Errorf("%d:%s:Fprint(%v) = %q; want %q", i, tc.tag, tc.value, got, tc.want)
		}
	}
}

func TestLocalization(t *testing.T) {
	type test struct {
		tag  string
		key  Reference
		args []interface{}
		want string
	}
	args := func(x ...interface{}) []interface{} { return x }
	empty := []interface{}{}
	joe := []interface{}{"Joe"}
	joeAndMary := []interface{}{"Joe", "Mary"}

	testCases := []struct {
		desc string
		cat  []entry
		test []test
	}{{
		desc: "empty",
		test: []test{
			{"en", "key", empty, "key"},
			{"en", "", empty, ""},
			{"nl", "", empty, ""},
		},
	}, {
		desc: "hierarchical languages",
		cat: []entry{
			{"en", "hello %s", "Hello %s!"},
			{"en-GB", "hello %s", "Hellø %s!"},
			{"en-US", "hello %s", "Howdy %s!"},
			{"en", "greetings %s and %s", "Greetings %s and %s!"},
		},
		test: []test{
			{"und", "hello %s", joe, "hello Joe"},
			{"nl", "hello %s", joe, "hello Joe"},
			{"en", "hello %s", joe, "Hello Joe!"},
			{"en-US", "hello %s", joe, "Howdy Joe!"},
			{"en-GB", "hello %s", joe, "Hellø Joe!"},
			{"en-oxendict", "hello %s", joe, "Hello Joe!"},
			{"en-US-oxendict-u-ms-metric", "hello %s", joe, "Howdy Joe!"},

			{"und", "greetings %s and %s", joeAndMary, "greetings Joe and Mary"},
			{"nl", "greetings %s and %s", joeAndMary, "greetings Joe and Mary"},
			{"en", "greetings %s and %s", joeAndMary, "Greetings Joe and Mary!"},
			{"en-US", "greetings %s and %s", joeAndMary, "Greetings Joe and Mary!"},
			{"en-GB", "greetings %s and %s", joeAndMary, "Greetings Joe and Mary!"},
			{"en-oxendict", "greetings %s and %s", joeAndMary, "Greetings Joe and Mary!"},
			{"en-US-oxendict-u-ms-metric", "greetings %s and %s", joeAndMary, "Greetings Joe and Mary!"},
		},
	}, {
		desc: "references",
		cat: []entry{
			{"en", "hello", "Hello!"},
		},
		test: []test{
			{"en", "hello", empty, "Hello!"},
			{"en", Key("hello", "fallback"), empty, "Hello!"},
			{"en", Key("xxx", "fallback"), empty, "fallback"},
			{"und", Key("hello", "fallback"), empty, "fallback"},
		},
	}, {
		desc: "zero substitution", // work around limitation of fmt
		cat: []entry{
			{"en", "hello %s", "Hello!"},
			{"en", "hi %s and %s", "Hello %[2]s!"},
		},
		test: []test{
			{"en", "hello %s", joe, "Hello!"},
			{"en", "hello %s", joeAndMary, "Hello!"},
			{"en", "hi %s and %s", joeAndMary, "Hello Mary!"},
			// The following tests resolve to the fallback string.
			{"und", "hello", joeAndMary, "hello"},
			{"und", "hello %%%%", joeAndMary, "hello %%"},
			{"und", "hello %#%%4.2%  ", joeAndMary, "hello %%  "},
			{"und", "hello %s", joeAndMary, "hello Joe%!(EXTRA string=Mary)"},
			{"und", "hello %+%%s", joeAndMary, "hello %Joe%!(EXTRA string=Mary)"},
			{"und", "hello %-42%%s ", joeAndMary, "hello %Joe %!(EXTRA string=Mary)"},
		},
	}, {
		desc: "number formatting", // work around limitation of fmt
		cat: []entry{
			{"und", "files", "%d files left"},
			{"und", "meters", "%.2f meters"},
			{"de", "files", "%d Dateien übrig"},
		},
		test: []test{
			{"en", "meters", args(3000.2), "3,000.20 meters"},
			{"en-u-nu-gujr", "files", args(123456), "૧૨૩,૪૫૬ files left"},
			{"de", "files", args(1234), "1.234 Dateien übrig"},
			{"de-CH", "files", args(1234), "1’234 Dateien übrig"},
			{"de-CH-u-nu-mong", "files", args(1234), "᠑’᠒᠓᠔ Dateien übrig"},
		},
	}}

	for _, tc := range testCases {
		cat, _ := initCat(tc.cat)

		for i, pt := range tc.test {
			t.Run(fmt.Sprintf("%s:%d", tc.desc, i), func(t *testing.T) {
				p := NewPrinter(language.MustParse(pt.tag), Catalog(cat))

				if got := p.Sprintf(pt.key, pt.args...); got != pt.want {
					t.Errorf("Sprintf(%q, %v) = %s; want %s",
						pt.key, pt.args, got, pt.want)
					return // Next error will likely be the same.
				}

				w := &bytes.Buffer{}
				p.Fprintf(w, pt.key, pt.args...)
				if got := w.String(); got != pt.want {
					t.Errorf("Fprintf(%q, %v) = %s; want %s",
						pt.key, pt.args, got, pt.want)
				}
			})
		}
	}
}

type entry struct{ tag, key, msg string }

func initCat(entries []entry) (*catalog.Catalog, []language.Tag) {
	tags := []language.Tag{}
	cat := catalog.New()
	for _, e := range entries {
		tag := language.MustParse(e.tag)
		tags = append(tags, tag)
		cat.SetString(tag, e.key, e.msg)
	}
	return cat, internal.UniqueTags(tags)
}
