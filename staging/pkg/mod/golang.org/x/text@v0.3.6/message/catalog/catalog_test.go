// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catalog

import (
	"bytes"
	"path"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
)

type entry struct {
	tag, key string
	msg      interface{}
}

func langs(s string) []language.Tag {
	t, _, _ := language.ParseAcceptLanguage(s)
	return t
}

type testCase struct {
	desc     string
	cat      []entry
	lookup   []entry
	fallback string
	match    []string
	tags     []language.Tag
}

var testCases = []testCase{{
	desc: "empty catalog",
	lookup: []entry{
		{"en", "key", ""},
		{"en", "", ""},
		{"nl", "", ""},
	},
	match: []string{
		"gr -> und",
		"en-US -> und",
		"af -> und",
	},
	tags: nil, // not an empty list.
}, {
	desc: "one entry",
	cat: []entry{
		{"en", "hello", "Hello!"},
	},
	lookup: []entry{
		{"und", "hello", ""},
		{"nl", "hello", ""},
		{"en", "hello", "Hello!"},
		{"en-US", "hello", "Hello!"},
		{"en-GB", "hello", "Hello!"},
		{"en-oxendict", "hello", "Hello!"},
		{"en-oxendict-u-ms-metric", "hello", "Hello!"},
	},
	match: []string{
		"gr -> en",
		"en-US -> en-u-rg-uszzzz",
	},
	tags: langs("en"),
}, {
	desc: "hierarchical languages",
	cat: []entry{
		{"en", "hello", "Hello!"},
		{"en-GB", "hello", "Hellø!"},
		{"en-US", "hello", "Howdy!"},
		{"en", "greetings", "Greetings!"},
		{"gsw", "hello", "Grüetzi!"},
	},
	lookup: []entry{
		{"und", "hello", ""},
		{"nl", "hello", ""},
		{"en", "hello", "Hello!"},
		{"en-US", "hello", "Howdy!"},
		{"en-GB", "hello", "Hellø!"},
		{"en-oxendict", "hello", "Hello!"},
		{"en-US-oxendict-u-ms-metric", "hello", "Howdy!"},

		{"und", "greetings", ""},
		{"nl", "greetings", ""},
		{"en", "greetings", "Greetings!"},
		{"en-US", "greetings", "Greetings!"},
		{"en-GB", "greetings", "Greetings!"},
		{"en-oxendict", "greetings", "Greetings!"},
		{"en-US-oxendict-u-ms-metric", "greetings", "Greetings!"},
	},
	fallback: "gsw",
	match: []string{
		"gr -> gsw",
		"en-US -> en-US",
	},
	tags: langs("gsw, en, en-GB, en-US"),
}, {
	desc: "variables",
	cat: []entry{
		{"en", "hello %s", []Message{
			Var("person", String("Jane")),
			String("Hello ${person}!"),
		}},
		{"en", "hello error", []Message{
			Var("person", String("Jane")),
			noMatchMessage{}, // trigger sequence path.
			String("Hello ${person."),
		}},
		{"en", "fallback to var value", []Message{
			Var("you", noMatchMessage{}, noMatchMessage{}),
			String("Hello ${you}."),
		}},
		{"en", "scopes", []Message{
			Var("person1", String("Mark")),
			Var("person2", String("Jane")),
			Var("couple",
				Var("person1", String("Joe")),
				String("${person1} and ${person2}")),
			String("Hello ${couple}."),
		}},
		{"en", "missing var", String("Hello ${missing}.")},
	},
	lookup: []entry{
		{"en", "hello %s", "Hello Jane!"},
		{"en", "hello error", "Hello $!(MISSINGBRACE)"},
		{"en", "fallback to var value", "Hello you."},
		{"en", "scopes", "Hello Joe and Jane."},
		{"en", "missing var", "Hello missing."},
	},
	tags: langs("en"),
}, {
	desc: "macros",
	cat: []entry{
		{"en", "macro1", String("Hello ${macro1(1)}.")},
		{"en", "macro2", String("Hello ${ macro1(2) }!")},
		{"en", "macroWS", String("Hello ${ macro1( 2 ) }!")},
		{"en", "missing", String("Hello ${ missing(1 }.")},
		{"en", "badnum", String("Hello ${ badnum(1b) }.")},
		{"en", "undefined", String("Hello ${ undefined(1) }.")},
		{"en", "macroU", String("Hello ${ macroU(2) }!")},
	},
	lookup: []entry{
		{"en", "macro1", "Hello Joe."},
		{"en", "macro2", "Hello Joe!"},
		{"en-US", "macroWS", "Hello Joe!"},
		{"en-NL", "missing", "Hello $!(MISSINGPAREN)."},
		{"en", "badnum", "Hello $!(BADNUM)."},
		{"en", "undefined", "Hello undefined."},
		{"en", "macroU", "Hello macroU!"},
	},
	tags: langs("en"),
}}

func setMacros(b *Builder) {
	b.SetMacro(language.English, "macro1", String("Joe"))
	b.SetMacro(language.Und, "macro2", String("${macro1(1)}"))
	b.SetMacro(language.English, "macroU", noMatchMessage{})
}

type buildFunc func(t *testing.T, tc testCase) Catalog

func initBuilder(t *testing.T, tc testCase) Catalog {
	options := []Option{}
	if tc.fallback != "" {
		options = append(options, Fallback(language.MustParse(tc.fallback)))
	}
	cat := NewBuilder(options...)
	for _, e := range tc.cat {
		tag := language.MustParse(e.tag)
		switch msg := e.msg.(type) {
		case string:

			cat.SetString(tag, e.key, msg)
		case Message:
			cat.Set(tag, e.key, msg)
		case []Message:
			cat.Set(tag, e.key, msg...)
		}
	}
	setMacros(cat)
	return cat
}

type dictionary map[string]string

func (d dictionary) Lookup(key string) (data string, ok bool) {
	data, ok = d[key]
	return data, ok
}

func initCatalog(t *testing.T, tc testCase) Catalog {
	m := map[string]Dictionary{}
	for _, e := range tc.cat {
		m[e.tag] = dictionary{}
	}
	for _, e := range tc.cat {
		var msg Message
		switch x := e.msg.(type) {
		case string:
			msg = String(x)
		case Message:
			msg = x
		case []Message:
			msg = firstInSequence(x)
		}
		data, _ := catmsg.Compile(language.MustParse(e.tag), nil, msg)
		m[e.tag].(dictionary)[e.key] = data
	}
	options := []Option{}
	if tc.fallback != "" {
		options = append(options, Fallback(language.MustParse(tc.fallback)))
	}
	c, err := NewFromMap(m, options...)
	if err != nil {
		t.Fatal(err)
	}
	// TODO: implement macros for fixed catalogs.
	b := NewBuilder()
	setMacros(b)
	c.(*catalog).macros.index = b.macros.index
	return c
}

func TestMatcher(t *testing.T) {
	test := func(t *testing.T, init buildFunc) {
		for _, tc := range testCases {
			for _, s := range tc.match {
				a := strings.Split(s, "->")
				t.Run(path.Join(tc.desc, a[0]), func(t *testing.T) {
					cat := init(t, tc)
					got, _ := language.MatchStrings(cat.Matcher(), a[0])
					want := language.MustParse(strings.TrimSpace(a[1]))
					if got != want {
						t.Errorf("got %q; want %q", got, want)
					}
				})
			}
		}
	}
	t.Run("Builder", func(t *testing.T) { test(t, initBuilder) })
	t.Run("Catalog", func(t *testing.T) { test(t, initCatalog) })
}

func TestCatalog(t *testing.T) {
	test := func(t *testing.T, init buildFunc) {
		for _, tc := range testCases {
			cat := init(t, tc)
			wantTags := tc.tags
			if got := cat.Languages(); !reflect.DeepEqual(got, wantTags) {
				t.Errorf("%s:Languages: got %v; want %v", tc.desc, got, wantTags)
			}

			for _, e := range tc.lookup {
				t.Run(path.Join(tc.desc, e.tag, e.key), func(t *testing.T) {
					tag := language.MustParse(e.tag)
					buf := testRenderer{}
					ctx := cat.Context(tag, &buf)
					want := e.msg.(string)
					err := ctx.Execute(e.key)
					gotFound := err != ErrNotFound
					wantFound := want != ""
					if gotFound != wantFound {
						t.Fatalf("err: got %v (%v); want %v", gotFound, err, wantFound)
					}
					if got := buf.buf.String(); got != want {
						t.Errorf("Lookup:\ngot  %q\nwant %q", got, want)
					}
				})
			}
		}
	}
	t.Run("Builder", func(t *testing.T) { test(t, initBuilder) })
	t.Run("Catalog", func(t *testing.T) { test(t, initCatalog) })
}

type testRenderer struct {
	buf bytes.Buffer
}

func (f *testRenderer) Arg(i int) interface{} { return nil }
func (f *testRenderer) Render(s string)       { f.buf.WriteString(s) }

var msgNoMatch = catmsg.Register("no match", func(d *catmsg.Decoder) bool {
	return false // no match
})

type noMatchMessage struct{}

func (noMatchMessage) Compile(e *catmsg.Encoder) error {
	e.EncodeMessageType(msgNoMatch)
	return catmsg.ErrIncomplete
}
