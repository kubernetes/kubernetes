/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package flag

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStringColonSeparatedMultimapStringString(t *testing.T) {
	var nilMap map[string][]string
	cases := []struct {
		desc   string
		m      *ColonSeparatedMultimapStringString
		expect string
	}{
		{"nil", NewColonSeparatedMultimapStringString(&nilMap), ""},
		{"empty", NewColonSeparatedMultimapStringString(&map[string][]string{}), ""},
		{"empty key", NewColonSeparatedMultimapStringString(
			&map[string][]string{
				"": {"foo"},
			}),
			":foo"},
		{"one key", NewColonSeparatedMultimapStringString(
			&map[string][]string{
				"one": {"foo"},
			}),
			"one:foo"},
		{"two keys", NewColonSeparatedMultimapStringString(
			&map[string][]string{
				"one": {"foo"},
				"two": {"bar"},
			}),
			"one:foo,two:bar"},
		{"two keys, multiple items in one key", NewColonSeparatedMultimapStringString(
			&map[string][]string{
				"one": {"foo", "baz"},
				"two": {"bar"},
			}),
			"one:foo,one:baz,two:bar"},
		{"three keys, multiple items in one key", NewColonSeparatedMultimapStringString(
			&map[string][]string{
				"a": {"hello"},
				"b": {"again", "beautiful"},
				"c": {"world"},
			}),
			"a:hello,b:again,b:beautiful,c:world"},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			str := c.m.String()
			assert.Equalf(t, c.expect, str, "expect %q but got %q", c.expect, str)
		})
	}
}

func TestSetColonSeparatedMultimapStringString(t *testing.T) {
	var nilMap map[string][]string
	cases := []struct {
		desc           string
		vals           []string
		start          *ColonSeparatedMultimapStringString
		expect         *ColonSeparatedMultimapStringString
		expectedToFail bool
		err            string
	}{
		// we initialize the map with a default key that should be cleared by Set

		{
			desc:  "clears defaults",
			vals:  []string{""},
			start: NewColonSeparatedMultimapStringString(&map[string][]string{"default": {}}),
			expect: &ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{},
			},
			expectedToFail: false,
			err:            "",
		},
		// make sure we still allocate for "initialized" multimaps where Multimap was initially set to a nil map
		{
			"allocates map if currently nil",
			[]string{""},
			&ColonSeparatedMultimapStringString{initialized: true, Multimap: &nilMap},
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{},
			},
			false,
			"",
		},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{
			"empty",
			[]string{""},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{},
			},
			false,
			"",
		},
		{
			"empty key",
			[]string{":foo"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"": {"foo"},
				},
			},
			false,
			"",
		},
		{
			"one key",
			[]string{"one:foo"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
				},
			},
			false,
			"",
		},
		{
			"two keys",
			[]string{"one:foo,two:bar"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
					"two": {"bar"},
				},
			},
			false,
			"",
		},

		{
			"two keys with space",
			[]string{"one:foo, two:bar"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
					"two": {"bar"},
				},
			},
			false,
			"",
		},
		{
			"two keys, multiple items in one key",
			[]string{"one: foo, two:bar, one:baz"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo", "baz"},
					"two": {"bar"},
				},
			},
			false,
			"",
		},
		{
			"three keys, multiple items in one key",
			[]string{"a:hello,b:again,c:world,b:beautiful"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"a": {"hello"},
					"b": {"again", "beautiful"},
					"c": {"world"},
				},
			},
			false,
			"",
		},
		{
			"three keys, multiple items in one key, multiple Set invocations",
			[]string{"a:hello,b:again", "c:world", "b:beautiful"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"a": {"hello"},
					"b": {"again", "beautiful"},
					"c": {"world"},
				},
			},
			false,
			"",
		},
		{
			"missing value",
			[]string{"a"},
			NewColonSeparatedMultimapStringString(&nilMap),
			nil,
			true,
			"malformed pair, expect string:string",
		},
		{
			"no target",
			[]string{"a:foo"},
			NewColonSeparatedMultimapStringString(nil),
			nil,
			true,
			"no target (nil pointer to map[string][]string)",
		},
	}

	for _, c := range cases {
		nilMap = nil
		t.Run(c.desc, func(t *testing.T) {
			var err error
			for _, val := range c.vals {
				if err = c.start.Set(val); err != nil {
					break
				}
			}
			if c.expectedToFail {
				assert.Equalf(t, c.err, err.Error(), "expected error %s but got %v", c.err, err)
			} else {
				assert.Nil(t, err, "unexpected error: %v", err)
				assert.Equalf(t, c.expect, c.start, "expect %#v but got %#v", c.expect, c.start)
			}
		})
	}
}

func TestRoundTripColonSeparatedMultimapStringString(t *testing.T) {
	cases := []struct {
		desc   string
		vals   []string
		expect string
	}{
		{"empty", []string{""}, ""},
		{"empty key", []string{":foo"}, ":foo"},
		{"one key", []string{"one:foo"}, "one:foo"},
		{"two keys", []string{"one:foo,two:bar"}, "one:foo,two:bar"},
		{"two keys, multiple items in one key", []string{"one:foo, two:bar, one:baz"}, "one:foo,one:baz,two:bar"},
		{"three keys, multiple items in one key", []string{"a:hello,b:again,c:world,b:beautiful"}, "a:hello,b:again,b:beautiful,c:world"},
		{"three keys, multiple items in one key, multiple Set invocations", []string{"a:hello,b:again", "c:world", "b:beautiful"}, "a:hello,b:again,b:beautiful,c:world"},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			m := NewColonSeparatedMultimapStringString(&map[string][]string{})
			for _, val := range c.vals {
				err := m.Set(val)
				assert.Nil(t, err, "unexpected error: %v", err)
			}
			str := m.String()
			assert.Equalf(t, c.expect, str, "expect %q but got %q", c.expect, str)
		})
	}
}

func TestEmptyColonSeparatedMultimapStringString(t *testing.T) {
	var nilMap map[string][]string
	cases := []struct {
		desc   string
		val    *ColonSeparatedMultimapStringString
		expect bool
	}{
		{"nil", NewColonSeparatedMultimapStringString(&nilMap), true},
		{"empty", NewColonSeparatedMultimapStringString(&map[string][]string{}), true},
		{"populated", NewColonSeparatedMultimapStringString(&map[string][]string{"foo": {}}), false},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			result := c.val.Empty()
			assert.Equalf(t, c.expect, result, "expect %t but got %t", c.expect, result)
		})
	}
}
