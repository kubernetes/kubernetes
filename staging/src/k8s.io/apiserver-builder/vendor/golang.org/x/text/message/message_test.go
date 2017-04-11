// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package message

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"golang.org/x/text/internal/format"
	"golang.org/x/text/language"
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

func TestFormatSelection(t *testing.T) {
	type test struct {
		tag  string
		key  Reference
		args []interface{}
		want string
	}
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
	}}

	for _, tc := range testCases {
		cat, _ := initCat(tc.cat)

		for i, pt := range tc.test {
			p := cat.Printer(language.MustParse(pt.tag))

			if got := p.Sprintf(pt.key, pt.args...); got != pt.want {
				t.Errorf("%s:%d:Sprintf(%s, %v) = %s; want %s",
					tc.desc, i, pt.key, pt.args, got, pt.want)
				continue // Next error will likely be the same.
			}

			w := &bytes.Buffer{}
			p.Fprintf(w, pt.key, pt.args...)
			if got := w.String(); got != pt.want {
				t.Errorf("%s:%d:Fprintf(%s, %v) = %s; want %s",
					tc.desc, i, pt.key, pt.args, got, pt.want)
			}
		}
	}
}
