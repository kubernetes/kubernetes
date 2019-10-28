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
			assert.Equalf(t, c.expect, c.m.String(), "expect %q but got %q", c.expect, c.m.String())
		})
	}
}

func TestSetMapStringBool(t *testing.T) {
	var nilMap map[string]bool
	cases := []struct {
		desc           string
		vals           []string
		start          *MapStringBool
		expect         *MapStringBool
		err            string
		expectedToFail bool
	}{
		// we initialize the map with a default key that should be cleared by Set
		{
			"clears defaults",
			[]string{""},
			NewMapStringBool(&map[string]bool{"default": true}),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			},
			"",
			false,
		},
		// make sure we still allocate for "initialized" maps where Map was initially set to a nil map
		{
			"allocates map if currently nil",
			[]string{""},
			&MapStringBool{initialized: true, Map: &nilMap},
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			},
			"",
			false,
		},
		// for most cases, we just reuse nilMap, which should be allocated by Set, and is reset before each test case
		{
			"empty",
			[]string{""},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{},
			},
			"",
			false,
		},
		{
			"one key",
			[]string{"one=true"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true},
			},
			"",
			false,
		},
		{
			"two keys",
			[]string{"one=true,two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			},
			"",
			false,
		},
		{
			"two keys, multiple Set invocations",
			[]string{"one=true", "two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			},
			"",
			false,
		},
		{
			"two keys with space",
			[]string{"one=true, two=false"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"one": true, "two": false},
			},
			"",
			false,
		},
		{
			"empty key",
			[]string{"=true"},
			NewMapStringBool(&nilMap),
			&MapStringBool{
				initialized: true,
				Map:         &map[string]bool{"": true},
			},
			"",
			false,
		},
		{
			"missing value",
			[]string{"one"},
			NewMapStringBool(&nilMap),
			nil,
			"malformed pair, expect string=bool",
			true,
		},
		{
			"non-boolean value",
			[]string{"one=foo"},
			NewMapStringBool(&nilMap),
			nil,
			`invalid value of one: foo, err: strconv.ParseBool: parsing "foo": invalid syntax`,
			true,
		},
		{
			"no target",
			[]string{"one=true"},
			NewMapStringBool(nil),
			nil,
			"no target (nil pointer to map[string]bool)",
			true,
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
				assert.Equalf(t, c.err, err.Error(), "expect error %s but got %v", c.err, err)
			} else {
				assert.Nil(t, err, "unexpected error: %v", err)
				assert.Equalf(t, c.expect, c.start, "expect %#v but got %#v", c.expect, c.start)
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
			assert.Equalf(t, c.expect, c.val.Empty(), "expect %t but got %t", c.expect, c.val.Empty())
		})
	}
}
