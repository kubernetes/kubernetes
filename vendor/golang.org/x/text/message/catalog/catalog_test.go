// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catalog

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"golang.org/x/text/internal"
	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
)

type entry struct {
	tag, key string
	msg      interface{}
}

var testCases = []struct {
	desc   string
	cat    []entry
	lookup []entry
}{{
	desc: "empty catalog",
	lookup: []entry{
		{"en", "key", ""},
		{"en", "", ""},
		{"nl", "", ""},
	},
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
}, {
	desc: "hierarchical languages",
	cat: []entry{
		{"en", "hello", "Hello!"},
		{"en-GB", "hello", "Hellø!"},
		{"en-US", "hello", "Howdy!"},
		{"en", "greetings", "Greetings!"},
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
	}}}

func initCat(entries []entry) (*Catalog, []language.Tag) {
	tags := []language.Tag{}
	cat := New()
	for _, e := range entries {
		tag := language.MustParse(e.tag)
		tags = append(tags, tag)
		switch msg := e.msg.(type) {
		case string:
			cat.SetString(tag, e.key, msg)
		case Message:
			cat.Set(tag, e.key, msg)
		case []Message:
			cat.Set(tag, e.key, msg...)
		}
	}
	return cat, internal.UniqueTags(tags)
}

func TestCatalog(t *testing.T) {
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s", tc.desc), func(t *testing.T) {
			cat, wantTags := initCat(tc.cat)
			cat.SetMacro(language.English, "macro1", String("Joe"))
			cat.SetMacro(language.Und, "macro2", String("${macro1(1)}"))
			cat.SetMacro(language.English, "macroU", noMatchMessage{})

			if got := cat.Languages(); !reflect.DeepEqual(got, wantTags) {
				t.Errorf("%s:Languages: got %v; want %v", tc.desc, got, wantTags)
			}

			for _, e := range tc.lookup {
				t.Run(fmt.Sprintf("%s/%s", e.tag, e.key), func(t *testing.T) {
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
		})
	}
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
