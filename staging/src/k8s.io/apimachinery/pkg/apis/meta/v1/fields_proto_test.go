/*
Copyright 2019 The Kubernetes Authors.

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

package v1

import (
	"encoding/json"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"
)

type FieldsHolder struct {
	F Fields `json:"f"`
}

func TestFieldsMarshalYAML(t *testing.T) {
	cases := []struct {
		input  Fields
		result string
	}{
		{Fields{}, "f: null\n"},
		{Fields{Raw: []byte(`{"f:metadata":{"f:name":{}}}`)}, "f:\n  f:metadata:\n    f:name: {}\n"},
	}

	for _, c := range cases {
		input := FieldsHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestFieldsUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input  string
		result Fields
	}{
		{"f: null\n", Fields{}},
		{"f:\n  f:metadata:\n    f:name: {}\n", Fields{Raw: []byte(`{"f:metadata":{"f:name":{}}}`)}},
	}

	for _, c := range cases {
		var result FieldsHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.F, c.result) {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestFieldsMarshalJSON(t *testing.T) {
	cases := []struct {
		input  Fields
		result string
	}{
		{Fields{}, `{"f":null}`},
		{Fields{Raw: []byte(`{"f:metadata":{"f:name":{}}}`)}, `{"f":{"f:metadata":{"f:name":{}}}}`},
	}

	for _, c := range cases {
		input := FieldsHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal input: '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input: '%v': expected %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestFieldsUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result Fields
	}{
		{`{"f":null}`, Fields{}},
		{`{"f":{"f:metadata":{"f:name":{}}}}`, Fields{Raw: []byte(`{"f:metadata":{"f:name":{}}}`)}},
	}

	for _, c := range cases {
		var result FieldsHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result.F, c.result) {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestFieldsProtoRoundtrip(t *testing.T) {
	cases := []struct {
		input Fields
	}{
		{Fields{}},
		{Fields{Raw: []byte(`{"f:metadata":{"f:name":{}}}`)}},
	}

	for _, c := range cases {
		input := c.input
		data, err := input.Marshal()
		if err != nil {
			t.Fatalf("Failed to marshal input: '%v': %v", input, err)
		}
		fields := Fields{}
		if err := fields.Unmarshal(data); err != nil {
			t.Fatalf("Failed to unmarshal output: '%v': %v", input, err)
		}
		if !reflect.DeepEqual(input, fields) {
			t.Errorf("Marshal->Unmarshal is not idempotent: '%v' vs '%v'", input, fields)
		}
	}
}

func TestFieldsProtoConvert(t *testing.T) {
	cases := []struct {
		input  ProtoFields
		result Fields
	}{
		{
			input: ProtoFields{
				Map: map[string]ProtoFields{
					"f:metadata": {
						Map: map[string]ProtoFields{
							"f:name": {},
						},
					},
				},
			},
			result: Fields{
				Raw: []byte(`{"f:metadata":{"f:name":null}}`),
			},
		},
	}

	for _, c := range cases {
		input := c.input
		data, err := input.Marshal()
		if err != nil {
			t.Fatalf("Failed to marshal input: '%v': %v", input, err)
		}
		fields := Fields{}
		if err := fields.Unmarshal(data); err != nil {
			t.Fatalf("Failed to unmarshal output: '%v': %v", input, err)
		}
		if !reflect.DeepEqual(c.result, fields) {
			t.Errorf("Marshal->Unmarshal doesn't understand aplha api format: '%v' vs '%v'", c.result, fields)
		}
	}
}
