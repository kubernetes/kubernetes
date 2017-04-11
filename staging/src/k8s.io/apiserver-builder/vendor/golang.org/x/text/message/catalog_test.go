// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"reflect"
	"testing"

	"golang.org/x/text/internal"
	"golang.org/x/text/language"
)

type entry struct{ tag, key, msg string }

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
}}

func initCat(entries []entry) (*Catalog, []language.Tag) {
	tags := []language.Tag{}
	cat := newCatalog()
	for _, e := range entries {
		tag := language.MustParse(e.tag)
		tags = append(tags, tag)
		cat.SetString(tag, e.key, e.msg)
	}
	return cat, internal.UniqueTags(tags)
}

func TestCatalog(t *testing.T) {
	for _, tc := range testCases {
		cat, wantTags := initCat(tc.cat)

		// languages
		if got := cat.Languages(); !reflect.DeepEqual(got, wantTags) {
			t.Errorf("%s:Languages: got %v; want %v", tc.desc, got, wantTags)
		}

		// Lookup
		for _, e := range tc.lookup {
			tag := language.MustParse(e.tag)
			msg, ok := cat.get(tag, e.key)
			if okWant := e.msg != ""; ok != okWant || msg != e.msg {
				t.Errorf("%s:Lookup(%s, %s) = %s, %v; want %s, %v", tc.desc, tag, e.key, msg, ok, e.msg, okWant)
			}
		}
	}
}
