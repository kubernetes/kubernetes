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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
)

type JSONSchemaPropsOrBoolHolder struct {
	JSPoB          JSONSchemaPropsOrBool  `json:"val1"`
	JSPoBOmitEmpty *JSONSchemaPropsOrBool `json:"val2,omitempty"`
}

func TestJSONSchemaPropsOrBoolUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result JSONSchemaPropsOrBoolHolder
	}{
		{`{}`, JSONSchemaPropsOrBoolHolder{}},

		{`{"val1": {}}`, JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{}}}},
		{`{"val1": {"type":"string"}}`, JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{Type: "string"}}}},
		{`{"val1": false}`, JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{}}},
		{`{"val1": true}`, JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true}}},

		{`{"val2": {}}`, JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{}}}},
		{`{"val2": {"type":"string"}}`, JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{Type: "string"}}}},
		{`{"val2": false}`, JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{}}},
		{`{"val2": true}`, JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true}}},
	}

	for _, c := range cases {
		var result JSONSchemaPropsOrBoolHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result, c.result) {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestJSONSchemaPropsOrBoolUnmarshalCBOR(t *testing.T) {
	cases := []struct {
		name          string
		input         string
		expected      JSONSchemaPropsOrBoolHolder
		expectTypeErr bool
	}{
		{
			name:     "empty parent",
			input:    "\xA0",
			expected: JSONSchemaPropsOrBoolHolder{},
		},
		{
			name:     "empty",
			input:    "\xA1\x44val1\xA0",
			expected: JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{}}},
		},
		{
			name:     "with schema props",
			input:    "\xA1\x44val1\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{Type: "string"}}},
		},
		{
			name:     "bool false",
			input:    "\xA1\x44val1\xF4",
			expected: JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{}},
		},
		{
			name:     "bool true",
			input:    "\xA1\x44val1\xF5",
			expected: JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true}},
		},
		{
			name:     "omitempty empty",
			input:    "\xA1\x44val2\xA0",
			expected: JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{}}},
		},
		{
			name:     "omitempty with schema props",
			input:    "\xA1\x44val2\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true, Schema: &JSONSchemaProps{Type: "string"}}},
		},
		{
			name:     "omitempty bool false",
			input:    "\xA1\x44val2\xF4",
			expected: JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{}},
		},
		{
			name:     "omitempty bool true",
			input:    "\xA1\x44val2\xF5",
			expected: JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true}},
		},
		{
			name:          "wrong type",
			input:         "\xA1\x44val1\x18\x2A",
			expectTypeErr: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var result JSONSchemaPropsOrBoolHolder

			err := cbor.Unmarshal([]byte(c.input), &result)
			if c.expectTypeErr {
				if err == nil || !strings.Contains(err.Error(), "boolean or JSON schema expected") {
					t.Fatalf("Expected error 'boolean or JSON schema expected', got %v", err)
				}
			} else {
				if (err != nil) != c.expectTypeErr {
					t.Errorf("Unmarshal error = %v, wantErr %v", err, c.expectTypeErr)
				}
			}
			if !reflect.DeepEqual(result, c.expected) {
				t.Errorf("Failed to unmarshal input %X", c.input)
				t.Logf(cmp.Diff(c.expected, result))
			}
		})
	}
}

func TestJSONSchemaPropsOrBoolMarshalJSON(t *testing.T) {
	cases := []struct {
		input  JSONSchemaPropsOrBoolHolder
		result string
	}{
		{JSONSchemaPropsOrBoolHolder{}, `{"val1":false}`},

		{JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{}}}, `{"val1":{}}`},
		{JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{Type: "string"}}}, `{"val1":{"type":"string"}}`},
		{JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{}}, `{"val1":false}`},
		{JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true}}, `{"val1":true}`},

		{JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{}}}, `{"val1":false,"val2":{}}`},
		{JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{Type: "string"}}}, `{"val1":false,"val2":{"type":"string"}}`},
		{JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{}}, `{"val1":false,"val2":false}`},
		{JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true}}, `{"val1":false,"val2":true}`},
	}

	for _, c := range cases {
		result, err := json.Marshal(&c.input)
		if err != nil {
			t.Errorf("Unexpected error marshaling input '%v': %v", c.input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %q, got %q", c.input, c.result, string(result))
		}
	}
}

func TestJSONSchemaPropsOrBoolMarshalCBOR(t *testing.T) {
	cases := []struct {
		name     string
		input    JSONSchemaPropsOrBoolHolder
		expected string
	}{
		{
			name:     "empty parent",
			input:    JSONSchemaPropsOrBoolHolder{},
			expected: "\xA1\x44val1\xF4",
		},
		{
			name:     "empty",
			input:    JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{}}},
			expected: "\xA1\x44val1\xA0",
		},
		{
			name:     "with schema props",
			input:    JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{Type: "string"}}},
			expected: "\xA1\x44val1\xA1\x44type\x46string",
		},
		{
			name:     "bool false",
			input:    JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{}},
			expected: "\xA1\x44val1\xF4",
		},
		{
			name:     "bool true",
			input:    JSONSchemaPropsOrBoolHolder{JSPoB: JSONSchemaPropsOrBool{Allows: true}},
			expected: "\xA1\x44val1\xF5",
		},
		{
			name:     "omitempty empty",
			input:    JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{}}},
			expected: "\xA2\x44val1\xF4\x44val2\xA0",
		},
		{
			name:     "omitempty with schema props",
			input:    JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Schema: &JSONSchemaProps{Type: "string"}}},
			expected: "\xA2\x44val1\xF4\x44val2\xA1\x44type\x46string",
		},
		{
			name:     "omitempty bool false",
			input:    JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{}},
			expected: "\xA2\x44val1\xF4\x44val2\xF4",
		},
		{
			name:     "omitempty bool true",
			input:    JSONSchemaPropsOrBoolHolder{JSPoBOmitEmpty: &JSONSchemaPropsOrBool{Allows: true}},
			expected: "\xA2\x44val1\xF4\x44val2\xF5",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result, err := cbor.Marshal(&c.input)
			if err != nil {
				t.Errorf("Unexpected error marshaling input '%v': %v", c.input, err)
			}
			if string(result) != c.expected {
				t.Errorf("Failed to marshal input")
				t.Logf("input   : %v", c.input)
				t.Logf("expected: %X", c.expected)
				t.Logf("actual  : %X", result)
			}
		})
	}
}

type JSONSchemaPropsOrArrayHolder struct {
	JSPoA          JSONSchemaPropsOrArray  `json:"val1"`
	JSPoAOmitEmpty *JSONSchemaPropsOrArray `json:"val2,omitempty"`
}

func TestJSONSchemaPropsOrArrayUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result JSONSchemaPropsOrArrayHolder
	}{
		{`{}`, JSONSchemaPropsOrArrayHolder{}},

		{`{"val1": {}}`, JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}}},
		{`{"val1": {"type":"string"}}`, JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}}},
		{`{"val1": [{}]}`, JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}}},
		{`{"val1": [{},{"type":"string"}]}`, JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}}},

		{`{"val2": {}}`, JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}}},
		{`{"val2": {"type":"string"}}`, JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}}},
		{`{"val2": [{}]}`, JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}}},
		{`{"val2": [{},{"type":"string"}]}`, JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}}},
	}

	for _, c := range cases {
		var result JSONSchemaPropsOrArrayHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result, c.result) {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func TestJSONSchemaPropsOrArrayUnmarshalCBOR(t *testing.T) {
	cases := []struct {
		name          string
		input         string
		expected      JSONSchemaPropsOrArrayHolder
		expectTypeErr bool
	}{
		{
			name:     "empty parent",
			input:    "\xA0",
			expected: JSONSchemaPropsOrArrayHolder{},
		},
		{
			name:     "props zero value",
			input:    "\xA1\x44val1\xA0",
			expected: JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}},
		},
		{
			name:     "props",
			input:    "\xA1\x44val1\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}},
		},
		{
			name:     "array props zero value",
			input:    "\xA1\x44val1\x81\xA0",
			expected: JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}},
		},
		{
			name:     "array props zero value and props",
			input:    "\xA1\x44val1\x82\xA0\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}},
		},
		{
			name:     "omitempty props zero value",
			input:    "\xA1\x44val2\xA0",
			expected: JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}},
		},
		{
			name:     "omitempty props",
			input:    "\xA1\x44val2\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}},
		},
		{
			name:     "omitempty array props zero value",
			input:    "\xA1\x44val2\x81\xA0",
			expected: JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}},
		},
		{
			name:     "omitempty array props zero value and props",
			input:    "\xA1\x44val2\x82\xA0\xA1\x44type\x46string",
			expected: JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}},
		},
		{
			name:          "wrong type",
			input:         "\xA1\x44val1\x18\x2A",
			expectTypeErr: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var result JSONSchemaPropsOrArrayHolder
			err := cbor.Unmarshal([]byte(c.input), &result)
			if c.expectTypeErr {
				if err == nil || !strings.Contains(err.Error(), "JSON schema or array of JSON schema expected") {
					t.Fatalf("Expected error 'JSON schema or array of JSON schema expected', got %v", err)
				}
			} else {
				if (err != nil) != c.expectTypeErr {
					t.Errorf("Unmarshal error = %v, wantErr %v", err, c.expectTypeErr)
				}
			}

			if !reflect.DeepEqual(result, c.expected) {
				t.Errorf("Failed to unmarshal input %X", c.input)
				t.Logf(cmp.Diff(c.expected, result))
			}
		})
	}
}

func TestJSONSchemaPropsOrArrayMarshalJSON(t *testing.T) {
	cases := []struct {
		input  JSONSchemaPropsOrArrayHolder
		result string
	}{
		{JSONSchemaPropsOrArrayHolder{}, `{"val1":null}`},

		{JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}}, `{"val1":{}}`},
		{JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}}, `{"val1":{"type":"string"}}`},
		{JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}}, `{"val1":[{}]}`},
		{JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}}, `{"val1":[{},{"type":"string"}]}`},

		{JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{}}, `{"val1":null,"val2":null}`},
		{JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}}, `{"val1":null,"val2":{}}`},
		{JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}}, `{"val1":null,"val2":{"type":"string"}}`},
		{JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}}, `{"val1":null,"val2":[{}]}`},
		{JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}}, `{"val1":null,"val2":[{},{"type":"string"}]}`},
	}

	for i, c := range cases {
		result, err := json.Marshal(&c.input)
		if err != nil {
			t.Errorf("%d: Unexpected error marshaling input '%v': %v", i, c.input, err)
		}
		if string(result) != c.result {
			t.Errorf("%d: Failed to marshal input '%v': expected: %q, got %q", i, c.input, c.result, string(result))
		}
	}
}

func TestJSONSchemaPropsOrArrayMarshalCBOR(t *testing.T) {
	cases := []struct {
		name     string
		input    JSONSchemaPropsOrArrayHolder
		expected string
	}{
		{
			name:     "null",
			input:    JSONSchemaPropsOrArrayHolder{},
			expected: "\xA1\x44val1\xF6",
		},
		{
			name:     "empty props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}},
			expected: "\xA1\x44val1\xA0",
		},
		{
			name:     "props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}},
			expected: "\xA1\x44val1\xA1\x44type\x46string",
		},
		{
			name:     "array with empty props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}},
			expected: "\xA1\x44val1\x81\xA0",
		},
		{
			name:     "array with empty props and props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoA: JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}},
			expected: "\xA1\x44val1\x82\xA0\xA1\x44type\x46string",
		},
		{
			name:     "omitempty null",
			input:    JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{}},
			expected: "\xA2\x44val1\xF6\x44val2\xF6",
		},
		{
			name:     "omitempty empty props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{}}},
			expected: "\xA2\x44val1\xF6\x44val2\xA0",
		},
		{
			name:     "omitempty props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{Schema: &JSONSchemaProps{Type: "string"}}},
			expected: "\xA2\x44val1\xF6\x44val2\xA1\x44type\x46string",
		},
		{
			name:     "omitempty array with empty props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}}}},
			expected: "\xA2\x44val1\xF6\x44val2\x81\xA0",
		},
		{
			name:     "omitempty array with empty props and props",
			input:    JSONSchemaPropsOrArrayHolder{JSPoAOmitEmpty: &JSONSchemaPropsOrArray{JSONSchemas: []JSONSchemaProps{{}, {Type: "string"}}}},
			expected: "\xA2\x44val1\xF6\x44val2\x82\xA0\xA1\x44type\x46string",
		},
	}

	for i, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			result, err := cbor.Marshal(&c.input)
			if err != nil {
				t.Errorf("%d: Unexpected error marshaling input '%v': %v", i, c.input, err)
			}
			if string(result) != c.expected {
				t.Errorf("Failed to marshal input")
				t.Logf("input   : %v", c.input)
				t.Logf("expected: %X", c.expected)
				t.Logf("actual  : %X", result)
			}
		})
	}
}

type JSONSchemaPropsOrStringArrayHolder struct {
	JSPoSA          JSONSchemaPropsOrStringArray  `json:"val1"`
	JSPoSAOmitEmpty *JSONSchemaPropsOrStringArray `json:"val2,omitempty"`
}

func TestJSONSchemaPropsOrStringArrayMarshalJSON(t *testing.T) {
	testCases := []struct {
		name     string
		input    JSONSchemaPropsOrStringArrayHolder
		expected string
	}{
		{
			name:     "empty parent",
			input:    JSONSchemaPropsOrStringArrayHolder{},
			expected: `{"val1":null}`,
		},
		{
			name: "empty props",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{},
				},
			},
			expected: `{"val1":{}}`,
		},
		{
			name: "props value",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{
						Type: "string",
					},
				},
			},
			expected: `{"val1":{"type":"string"}}`},
		{
			name: "empty array",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Property: []string{},
				},
			},
			expected: `{"val1":null}`},
		{
			name: "array value",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Property: []string{"string"},
				},
			},
			expected: `{"val1":["string"]}`},
		{
			name: "omitempty empty",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{},
			},
			expected: `{"val1":null,"val2":null}`},
		{
			name: "omitempty empty props",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{},
				},
			},
			expected: `{"val1":null,"val2":{}}`},
		{
			name:     "omitempty props value",
			input:    JSONSchemaPropsOrStringArrayHolder{JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{Schema: &JSONSchemaProps{Type: "string"}}},
			expected: `{"val1":null,"val2":{"type":"string"}}`},
		{
			name: "omitempty empty array",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Property: []string{},
				},
			},
			expected: `{"val1":null,"val2":null}`,
		},
		{
			name: "omitempty array value",
			input: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Property: []string{"string"},
				},
			},
			expected: `{"val1":null,"val2":["string"]}`,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := json.Marshal(&tc.input)
			if err != nil {
				t.Errorf("%d: Unexpected error marshaling input '%v': %v", i, tc.input, err)
			}
			if string(result) != tc.expected {
				t.Errorf("%d: Failed to marshal input '%v': expected: %q, got %q", i, tc.input, tc.expected, string(result))
			}
		})
	}
}

func TestJSONSchemaPropsOrStringArrayUnmarshalJSON(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected JSONSchemaPropsOrStringArrayHolder
	}{
		{
			name:     "empty parent",
			input:    `{}`,
			expected: JSONSchemaPropsOrStringArrayHolder{},
		},
		{
			name:  "empty props value",
			input: `{"val1": {}}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{},
				},
			},
		},
		{
			name:  "props value",
			input: `{"val1": {"type":"string"}}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{Type: "string"},
				},
			},
		},
		{
			name:  "empty array value",
			input: `{"val1": []}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Property: []string{},
				},
			},
		},
		{
			name:  "array value",
			input: `{"val1": ["string"]}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSA: JSONSchemaPropsOrStringArray{
					Property: []string{"string"},
				},
			},
		},
		{
			name:  "omitempty empty",
			input: `{"val2": {}}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{},
				},
			},
		},
		{
			name:  "omitempty props value",
			input: `{"val2": {"type":"string"}}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Schema: &JSONSchemaProps{Type: "string"},
				},
			},
		},
		{
			name:  "omitempty empty array",
			input: `{"val2": []}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Property: []string{},
				},
			},
		},
		{
			name:  "omitempty array value",
			input: `{"val2": ["string"]}`,
			expected: JSONSchemaPropsOrStringArrayHolder{
				JSPoSAOmitEmpty: &JSONSchemaPropsOrStringArray{
					Property: []string{"string"},
				},
			},
		},
		{
			name:     "omitempty null values",
			input:    `{"val1":null,"val2":null}`,
			expected: JSONSchemaPropsOrStringArrayHolder{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var result JSONSchemaPropsOrStringArrayHolder
			if err := json.Unmarshal([]byte(tc.input), &result); err != nil {
				t.Errorf("Failed to unmarshal input '%v': %v", tc.input, err)
			}
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", tc.input, tc.expected, result)
				t.Log(cmp.Diff(tc.expected, result))
			}
		})
	}
}
