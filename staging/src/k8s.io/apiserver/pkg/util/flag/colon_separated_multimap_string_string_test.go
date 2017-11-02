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
	cases := []struct {
		desc   string
		m      ColonSeparatedMultimapStringString
		expect string
	}{
		{"empty", ColonSeparatedMultimapStringString{}, ""},
		{"empty key", ColonSeparatedMultimapStringString{"": []string{"foo"}}, ":foo"},
		{"one key", ColonSeparatedMultimapStringString{"one": []string{"foo"}}, "one:foo"},
		{"two keys", ColonSeparatedMultimapStringString{"one": []string{"foo"}, "two": []string{"bar"}}, "one:foo,two:bar"},
		{"two keys, multiple items in one key", ColonSeparatedMultimapStringString{"one": []string{"foo", "baz"}, "two": []string{"bar"}}, "one:foo,one:baz,two:bar"},
		{"three keys, multiple items in one key", ColonSeparatedMultimapStringString{"a": []string{"hello"}, "b": []string{"again", "beautiful"}, "c": []string{"world"}}, "a:hello,b:again,b:beautiful,c:world"},
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
	cases := []struct {
		desc   string
		val    string
		expect ColonSeparatedMultimapStringString
		err    string
	}{
		{"empty", "", ColonSeparatedMultimapStringString{}, ""},
		{"empty key", ":foo", ColonSeparatedMultimapStringString{
			"": []string{"foo"},
		}, ""},
		{"one key", "one:foo", ColonSeparatedMultimapStringString{
			"one": []string{"foo"}}, ""},
		{"two keys", "one:foo,two:bar", ColonSeparatedMultimapStringString{
			"one": []string{"foo"},
			"two": []string{"bar"},
		}, ""},
		{"two keys with space", "one:foo, two:bar", ColonSeparatedMultimapStringString{
			"one": []string{"foo"},
			"two": []string{"bar"},
		}, ""},
		{"two keys, multiple items in one key", "one: foo, two:bar, one:baz", ColonSeparatedMultimapStringString{
			"one": []string{"foo", "baz"},
			"two": []string{"bar"},
		}, ""},
		{"three keys, multiple items in one key", "a:hello,b:again,c:world,b:beautiful", ColonSeparatedMultimapStringString{
			"a": []string{"hello"},
			"b": []string{"again", "beautiful"},
			"c": []string{"world"},
		}, ""},
		{"missing value", "one", ColonSeparatedMultimapStringString{}, "malformed pair, expect string:string"},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			// we initialize the map with a default key that should be cleared by Set (no test cases specify "default")
			m := ColonSeparatedMultimapStringString{"default": []string{}}
			err := m.Set(c.val)
			if c.err != "" {
				if err.Error() != c.err {
					t.Fatalf("expect error %s but got %v", c.err, err)
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(c.expect, m) {
				t.Fatalf("expect %#v but got %#v", c.expect, m)
			}
		})
	}
}

func TestRoundTripColonSeparatedMultimapStringString(t *testing.T) {
	cases := []struct {
		desc   string
		val    string
		expect string
	}{
		{"empty", "", ""},
		{"empty key", ":foo", ":foo"},
		{"one key", "one:foo", "one:foo"},
		{"two keys", "one:foo,two:bar", "one:foo,two:bar"},
		{"two keys, multiple items in one key", "one:foo, two:bar, one:baz", "one:foo,one:baz,two:bar"},
		{"three keys, multiple items in one key", "a:hello,b:again,c:world,b:beautiful", "a:hello,b:again,b:beautiful,c:world"},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			m := ColonSeparatedMultimapStringString{}
			if err := m.Set(c.val); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			str := m.String()
			if c.expect != str {
				t.Fatalf("expect %q but got %q", c.expect, str)
			}
		})
	}
}
