/*
Copyright 2014 Google Inc. All rights reserved.

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

package merge

import (
	"testing"
)

func TestMergeJSON(t *testing.T) {
	tests := []struct {
		target   string
		patch    string
		expected string
	}{
		{target: `{}`, patch: `{}`, expected: `{}`},
		{target: `{"a":"b","c":{"d":"e","f":"g"}}`, patch: `{}`, expected: `{"a":"b","c":{"d":"e","f":"g"}}`},
		//update
		{target: `{"a":"b","c":"d"}`, patch: `{"a":"z"}`, expected: `{"a":"z","c":"d"}`},
		//remove key
		{target: `{"f":"g"}`, patch: `{"f":null}`, expected: `{}`},
		//inner update
		{target: `{"c":{"d":"e","f":"g"}}`, patch: `{"c":{"f":"z"}}`, expected: `{"c":{"d":"e","f":"z"}}`},
		//inner remove
		{target: `{"c":{"d":"e","f":"g"}}`, patch: `{"c":{"f":null}}`, expected: `{"c":{"d":"e"}}`},
		//complex update and remove
		{target: `{"a":"b","c":{"d":"e","f":"g"}}`, patch: `{"a":"z","c":{"f":null}}`, expected: `{"a":"z","c":{"d":"e"}}`},
		// test cases from https://tools.ietf.org/html/rfc7386#appendix-A slightly adapted to correspond to go's
		// encoding/json conventions
		{target: `{"a":"b"}`, patch: `{"a":"c"}`, expected: `{"a":"c"}`},
		{target: `{"a":"b"}`, patch: `{"b":"c"}`, expected: `{"a":"b","b":"c"}`},
		{target: `{"a":"b"}`, patch: `{"a":null}`, expected: `{}`},
		{target: `{"a":"b","b":"c"}`, patch: `{"a":null}`, expected: `{"b":"c"}`},
		{target: `{"a":["b"]}`, patch: `{"a":"c"}`, expected: `{"a":"c"}`},
		{target: `{"a":"c"}`, patch: `{"a":["b"]}`, expected: `{"a":["b"]}`},
		{target: `{"a":{"b": "c"}}`, patch: `{"a": {"b": "d","c": null}}`, expected: `{"a":{"b":"d","c":null}}`},
		{target: `{"a":[{"b":"c"}]}`, patch: `{"a":[1]}`, expected: `{"a":[1]}`},
		{target: `["a","b"]`, patch: `["c","d"]`, expected: `["c","d"]`},
		{target: `{"a":"b"}`, patch: `["c"]`, expected: `["c"]`},
		{target: `{"a":"foo"}`, patch: `null`, expected: `null`},
		{target: `{"a":"foo"}`, patch: `"bar"`, expected: `"bar"`},
		{target: `{"e":null}`, patch: `{"a":1}`, expected: `{"a":1,"e":null}`},
		{target: `[1,2]`, patch: `{"a":"b","c":null}`, expected: `{"a":"b","c":null}`},
		{target: `{}`, patch: `{"a":{"bb":{"ccc":null}}}`, expected: `{"a":{"bb":{"ccc":null}}}`},
	}
	for i, test := range tests {
		out, err := MergeJSON([]byte(test.target), []byte(test.patch))
		if err != nil {
			t.Errorf("case %v, unexpected error: %v", i, err)
		}
		if string(out) != test.expected {
			t.Errorf("case %v, expected:\n%v\nsaw:\n%v\n", i, test.expected, string(out))
		}
	}
}
