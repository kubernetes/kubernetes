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

func TestStringMapStringBool(t *testing.T) {
	var nilMap map[string]bool
	cases := []struct {
		desc   string
		m      *MapStringBool
		expect string
	}{
		{"nil", NewMapStringBool(&nilMap), ""},
		{"empty", NewMapStringBool(&map[string]bool{}), ""},
		{"one key", NewMapStringBool(&map[string]bool{"one": true}), "one=true"},
		{"two keys", NewMapStringBool(&map[string]bool{"one": true, "two": false}), "one=true,two=false"},
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

func TestSetMapStringBool(t *testing.T) {
	var nilMap map[string]bool
	cases := []struct {
		desc   string
		vals   []string
		start  *MapStringBool
		expect *MapStringBool
		err    string
	}{
		// we initialize the map with a default key that should be cleared by Set
		{"clears defaults", []string{""},
			NewMapStringBool(&map[string]bool{"default": true}),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			}, ""},
		// make sure we still allocate for "initialized" maps where Map was initially set to a nil map
		{"allocates map if currently nil", []string{""},
			&MapStringBool{initialized: true, Map: &nilMap},
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			}, ""},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{"empty", []string{""},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			}, ""},
		{"one key", []string{"one=true"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true},
			}, ""},
		{"two keys", []string{"one=true,two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			}, ""},
		{"two keys, multiple Set invocations", []string{"one=true", "two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			}, ""},
		{"two keys with space", []string{"one=true, two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			}, ""},
		{"empty key", []string{"=true"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"": true},
			}, ""},
		{"missing value", []string{"one"},
			NewMapStringBool(&nilMap),
			nil,
			"malformed pair, expect string=bool"},
		{"non-boolean value", []string{"one=foo"},
			NewMapStringBool(&nilMap),
			nil,
			`invalid value of one: foo, err: strconv.ParseBool: parsing "foo": invalid syntax`},
		{"no target", []string{"one=true"},
			NewMapStringBool(nil),
			nil,
			"no target (nil pointer to map[string]bool)"},
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

func TestEmptyMapStringBool(t *testing.T) {
	var nilMap map[string]bool
	cases := []struct {
		desc   string
		val    *MapStringBool
		expect bool
	}{
		{"nil", NewMapStringBool(&nilMap), true},
		{"empty", NewMapStringBool(&map[string]bool{}), true},
		{"populated", NewMapStringBool(&map[string]bool{"foo": true}), false},
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
