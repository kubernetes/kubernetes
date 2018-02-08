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

func TestStringMapStringString(t *testing.T) {
	var nilMap map[string]string
	cases := []struct {
		desc   string
		m      *MapStringString
		expect string
	}{
		{"nil", NewMapStringString(&nilMap), ""},
		{"empty", NewMapStringString(&map[string]string{}), ""},
		{"one key", NewMapStringString(&map[string]string{"one": "foo"}), "one=foo"},
		{"two keys", NewMapStringString(&map[string]string{"one": "foo", "two": "bar"}), "one=foo,two=bar"},
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

func TestSetMapStringString(t *testing.T) {
	var nilMap map[string]string
	cases := []struct {
		desc   string
		vals   []string
		start  *MapStringString
		expect *MapStringString
		err    string
	}{
		// we initialize the map with a default key that should be cleared by Set
		{"clears defaults", []string{""},
			NewMapStringString(&map[string]string{"default": ""}),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{},
			}, ""},
		// make sure we still allocate for "initialized" maps where Map was initially set to a nil map
		{"allocates map if currently nil", []string{""},
			&MapStringString{initialized: true, Map: &nilMap},
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{},
			}, ""},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{"empty", []string{""},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{},
			}, ""},
		{"one key", []string{"one=foo"},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{"one": "foo"},
			}, ""},
		{"two keys", []string{"one=foo,two=bar"},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{"one": "foo", "two": "bar"},
			}, ""},
		{"two keys, multiple Set invocations", []string{"one=foo", "two=bar"},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{"one": "foo", "two": "bar"},
			}, ""},
		{"two keys with space", []string{"one=foo, two=bar"},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{"one": "foo", "two": "bar"},
			}, ""},
		{"empty key", []string{"=foo"},
			NewMapStringString(&nilMap),
			&MapStringString{
				initialized: true,
				Map:         &map[string]string{"": "foo"},
			}, ""},
		{"missing value", []string{"one"},
			NewMapStringString(&nilMap),
			nil,
			"malformed pair, expect string=string"},
		{"no target", []string{"a:foo"},
			NewMapStringString(nil),
			nil,
			"no target (nil pointer to map[string]string)"},
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

func TestEmptyMapStringString(t *testing.T) {
	var nilMap map[string]string
	cases := []struct {
		desc   string
		val    *MapStringString
		expect bool
	}{
		{"nil", NewMapStringString(&nilMap), true},
		{"empty", NewMapStringString(&map[string]string{}), true},
		{"populated", NewMapStringString(&map[string]string{"foo": ""}), false},
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
