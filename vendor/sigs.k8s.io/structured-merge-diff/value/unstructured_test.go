/*
Copyright 2018 The Kubernetes Authors.

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

package value

import (
	"fmt"
	"reflect"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestUnstructuredYAML(t *testing.T) {
	objects := []string{
		`{}`,
		// Valid yaml that isn't parsed right due to our use of MapSlice:
		// `[{}]`,
		// These two are also valid, and they do parse, but I'm not sure
		// they construct the right object:
		// `[]`,
		// `["a",{},"b",null]`,
		`foo: bar`,
		`foo:
  - bar
  - baz
qux: [1, 2]`,
		`1.5`,
		`true`,
		`"foo"`,
		`false`,
		`a:
  a: null
  b: null
  c: null
  d: null
z:
  d: null
  c: null
  b: null
  a: null
`,
		`foo:
  baz:
    bar:
      qux: [true, false, 1, "1"]
`,
		// TODO: I'd like to test random objects.
	}

	for i := range objects {
		b := []byte(objects[i])
		t.Run(fmt.Sprintf("unstructured-ordered-%v", i), func(t *testing.T) {
			t.Parallel()
			runUnstructuredTestOrderedYAML(t, b)
		})
		t.Run(fmt.Sprintf("unstructured-unordered-%v", i), func(t *testing.T) {
			t.Parallel()
			runUnstructuredTestUnorderedYAML(t, b)
		})
	}
}

func runUnstructuredTestOrderedYAML(t *testing.T, input []byte) {
	var decoded interface{}
	// this enables order sensitivity; note the yaml package is broken
	// for e.g. documents that have root-level arrays.
	var ms yaml.MapSlice
	if err := yaml.Unmarshal(input, &ms); err == nil {
		decoded = ms
	} else if err := yaml.Unmarshal(input, &decoded); err != nil {
		t.Fatalf("failed to decode (%v):\n%s", err, input)
	}

	v, err := FromUnstructured(decoded)
	if err != nil {
		t.Fatalf("failed to interpret (%v):\n%s", err, input)
	}

	dcheck, _ := yaml.Marshal(decoded)

	encoded := v.ToUnstructured(true)
	echeck, err := yaml.Marshal(encoded)
	if err != nil {
		t.Fatalf("unstructured rendered an unencodable output: %v", err)
	}

	if string(dcheck) != string(echeck) {
		t.Fatalf("From/To were not inverse.\n\ndecoded: %#v\n\nencoded: %#v\n\ndecoded:\n%s\n\nencoded:\n%s", decoded, encoded, dcheck, echeck)
	}

	echeck2, err := v.ToYAML()
	if err != nil {
		t.Fatalf("ToYAML gave different result: %v", err)
	}
	if string(echeck) != string(echeck2) {
		t.Errorf("ToYAML gave different result:\n%v", echeck2)
	}
}

func runUnstructuredTestUnorderedYAML(t *testing.T, input []byte) {
	var decoded interface{}
	err := yaml.Unmarshal(input, &decoded)
	if err != nil {
		t.Fatalf("failed to decode (%v):\n%s", err, input)
	}

	v, err := FromUnstructured(decoded)
	if err != nil {
		t.Fatalf("failed to interpret (%v):\n%s", err, input)
	}

	dcheck, _ := yaml.Marshal(decoded)

	encoded := v.ToUnstructured(false)
	echeck, err := yaml.Marshal(encoded)
	if err != nil {
		t.Fatalf("unstructured rendered an unencodable output: %v", err)
	}

	if string(dcheck) != string(echeck) {
		t.Fatalf("From/To were not inverse.\n\ndecoded: %#v\n\nencoded: %#v\n\ndecoded:\n%s\n\nencoded:\n%s", decoded, encoded, dcheck, echeck)
	}
}

func TestRoundTrip(t *testing.T) {
	i := map[string]interface{}{
		"foo": map[string]interface{}{
			"bar": map[string]interface{}{
				"qux": []interface{}{true, false, int64(1), float64(1.1), nil, "1"},
			},
		},
	}
	v, err := FromUnstructured(i)
	if err != nil {
		t.Fatalf("failed to interpret (%v):\n%s", err, i)
	}
	o := v.ToUnstructured(false)
	if !reflect.DeepEqual(i, o) {
		t.Fatalf("Failed to round-trip.\ninput: %#v\noutput: %#v", i, o)
	}
}

func TestToFromJSON(t *testing.T) {
	js := []string{
		"null",
		"1",
		"1.2",
		`"something"`,
		`[1,2,null,"something"]`,
		`[]`,
		`{}`,
		`{"a":[null,1.2],"b":"something"}`,
	}

	for i, j := range js {
		t.Run(fmt.Sprintf("Test %d", i), func(t *testing.T) {
			v, err := FromJSON([]byte(j))
			if err != nil {
				t.Fatalf("failed to parse json: %v", err)
			}
			o, err := v.ToJSON()
			if err != nil {
				t.Fatalf("failed to marshal into json: %v", err)
			}
			if !reflect.DeepEqual(j, string(o)) {
				t.Fatalf("Failed to round-trip.\ninput: %#v\noutput: %#v", j, string(o))
			}
		})
		t.Run(fmt.Sprintf("Fast %d", i), func(t *testing.T) {
			v, err := FromJSONFast([]byte(j))
			if err != nil {
				t.Fatalf("failed to parse json: %v", err)
			}
			o, err := v.ToJSONFast()
			if err != nil {
				t.Fatalf("failed to marshal into json: %v", err)
			}
			if !reflect.DeepEqual(j, string(o)) {
				t.Fatalf("Failed to round-trip.\ninput: %#v\noutput: %#v", j, string(o))
			}
		})
	}
}

func TestJSONParseError(t *testing.T) {
	js := []string{
		"invalid json",
	}

	for _, j := range js {
		t.Run(fmt.Sprintf("%q", j), func(t *testing.T) {
			v, err := FromJSON([]byte(j))
			if err == nil {
				t.Fatalf("wanted error but got: %#v", v)
			}
		})
		t.Run(fmt.Sprintf("fast-%q", j), func(t *testing.T) {
			v, err := FromJSONFast([]byte(j))
			if err == nil {
				t.Fatalf("wanted error but got: %#v", v)
			}
		})
	}
}
