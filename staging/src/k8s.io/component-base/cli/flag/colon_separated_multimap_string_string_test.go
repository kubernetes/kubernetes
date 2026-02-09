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
	"reflect"
	"testing"
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
			if c.expect != str {
				t.Fatalf("expect %q but got %q", c.expect, str)
			}
		})
	}
}

func TestSetColonSeparatedMultimapStringString(t *testing.T) {
	var nilMap map[string][]string
	cases := []struct {
		desc   string
		vals   []string
		start  *ColonSeparatedMultimapStringString
		expect *ColonSeparatedMultimapStringString
		err    string
	}{
		// we initialize the map with a default key that should be cleared by Set
		{"clears defaults", []string{""},
			NewColonSeparatedMultimapStringString(&map[string][]string{"default": {}}),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{}}, ""},
		// make sure we still allocate for "initialized" multimaps where Multimap was initially set to a nil map
		{"allocates map if currently nil", []string{""},
			&ColonSeparatedMultimapStringString{initialized: true, Multimap: &nilMap},
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{},
			}, ""},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{"empty", []string{""},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap:    &map[string][]string{}}, ""},
		{"empty key no colon", []string{"foo"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"": {"foo"},
				}}, "malformed pair, expect string:string"},
		{"empty key no colon allowed", []string{"foo"},
			NewColonSeparatedMultimapStringStringAllowDefaultEmptyKey(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized:          true,
				allowDefaultEmptyKey: true,
				Multimap: &map[string][]string{
					"": {"foo"},
				}}, ""},
		{"empty key", []string{":foo"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"": {"foo"},
				}}, ""},
		{"one key", []string{"one:foo"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
				}}, ""},
		{"two keys", []string{"one:foo,two:bar"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
					"two": {"bar"},
				}}, ""},
		{"two keys with space", []string{"one:foo, two:bar"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo"},
					"two": {"bar"},
				}}, ""},
		{"two keys, multiple items in one key", []string{"one: foo, two:bar, one:baz"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"one": {"foo", "baz"},
					"two": {"bar"},
				}}, ""},
		{"three keys, multiple items in one key", []string{"a:hello,b:again,c:world,b:beautiful"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"a": {"hello"},
					"b": {"again", "beautiful"},
					"c": {"world"},
				}}, ""},
		{"three keys, multiple items in one key, multiple Set invocations", []string{"a:hello,b:again", "c:world", "b:beautiful"},
			NewColonSeparatedMultimapStringString(&nilMap),
			&ColonSeparatedMultimapStringString{
				initialized: true,
				Multimap: &map[string][]string{
					"a": {"hello"},
					"b": {"again", "beautiful"},
					"c": {"world"},
				}}, ""},
		{"missing value", []string{"a"},
			NewColonSeparatedMultimapStringString(&nilMap),
			nil,
			"malformed pair, expect string:string"},
		{"no target", []string{"a:foo"},
			NewColonSeparatedMultimapStringString(nil),
			nil,
			"no target (nil pointer to map[string][]string)"},
	}

	for _, c := range cases {
		nilMap = nil
		t.Run(c.desc, func(t *testing.T) {
			var err error
			for _, val := range c.vals {
				err = c.start.Set(val)
				if err != nil {
					break
				}
			}
			if c.err != "" {
				if err == nil || err.Error() != c.err {
					t.Fatalf("expect error %s but got %v", c.err, err)
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(c.expect, c.start) {
				t.Fatalf("expect %#v but got %#v", c.expect, c.start)
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
				if err := m.Set(val); err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
			str := m.String()
			if c.expect != str {
				t.Fatalf("expect %q but got %q", c.expect, str)
			}
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
			if result != c.expect {
				t.Fatalf("expect %t but got %t", c.expect, result)
			}
		})
	}
}
