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

package arraystr

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"
)

func TestFromStringArray(t *testing.T) {
	a := FromStringArray([]string{"foo", "bar"})
	if a.Type != Array || !reflect.DeepEqual(a.ArrayVal, []string{"foo", "bar"}) {
		t.Errorf(`Expected ArrayVal=[]{"foo", "bar"}, got %+v`, a)
	}
}

func TestFromStringArrayOne(t *testing.T) {
	a := FromStringArray([]string{"foo"})
	if a.Type != Array || !reflect.DeepEqual(a.ArrayVal, []string{"foo"}) {
		t.Errorf(`Expected ArrayVal=[]{"foo"}, got %+v`, a)
	}
}

func TestFromStringArrayEmpty(t *testing.T) {
	a := FromStringArray([]string{})
	if a.Type != String || a.StrVal != "" {
		t.Errorf(`Expected StrVal=\"\", got %+v`, a)
	}
}

func TestFromString(t *testing.T) {
	s := FromString("76")
	if s.Type != String || s.StrVal != "76" {
		t.Errorf("Expected StrVal=\"76\", got %+v", s)
	}
}

func TestFromStringEmpty(t *testing.T) {
	s := FromString("")
	if s.Type != String || s.StrVal != "" {
		t.Errorf("Expected StrVal=\"\", got %+v", s)
	}
}

type StringArrayOrStringHolder struct {
	SAOrS          StringArrayOrString  `json:"val1"`
	SAOrSOmitEmpty *StringArrayOrString `json:"val2,omitempty"`
}

func TestStringArrayOrStringUnmarshalJSON(t *testing.T) {
	zero := FromString("")
	cases := []struct {
		input  string
		result StringArrayOrStringHolder
	}{
		{`{"val1": ""}`, StringArrayOrStringHolder{SAOrS: FromString("")}},
		{`{"val1": []}`, StringArrayOrStringHolder{SAOrS: FromString("")}},
		{`{"val1": [""]}`, StringArrayOrStringHolder{SAOrS: FromStringArray([]string{""})}},
		{`{"val1": ["foo"]}`, StringArrayOrStringHolder{SAOrS: FromStringArray([]string{"foo"})}},
		{`{"val1": ["foo", "bar"]}`, StringArrayOrStringHolder{SAOrS: FromStringArray([]string{"foo", "bar"})}},
		{`{"val1": "foo"}`, StringArrayOrStringHolder{SAOrS: FromString("foo")}},

		{`{"val2": ""}`, StringArrayOrStringHolder{SAOrSOmitEmpty: &zero}},
		{`{"val2": []}`, StringArrayOrStringHolder{SAOrSOmitEmpty: &zero}}, // this is special, but expected!
		{`{"val2": [""]}`, StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{""})}},
		{`{"val2": ["foo"]}`, StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{"foo"})}},
		{`{"val2": ["foo", "bar"]}`, StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{"foo", "bar"})}},
		{`{"val2": "foo"}`, StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringPtr("foo")}},
	}

	for _, c := range cases {
		var result StringArrayOrStringHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if !reflect.DeepEqual(result, c.result) {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
		}
	}
}

func fromStringArrayPtr(val []string) *StringArrayOrString {
	x := FromStringArray(val)
	if x.IsZero() {
		return nil
	}
	return &x
}

// FromString creates an StringArrayOrString object with a string value.
func fromStringPtr(val string) *StringArrayOrString {
	x := FromString(val)
	if x.IsZero() {
		return nil
	}
	return &x
}

func TestStringArrayOrStringMarshalJSON(t *testing.T) {
	cases := []struct {
		input  StringArrayOrStringHolder
		result string
	}{
		{StringArrayOrStringHolder{SAOrS: FromStringArray([]string{})}, `{"val1":""}`},
		{StringArrayOrStringHolder{SAOrS: FromStringArray([]string{""})}, `{"val1":[""]}`},
		{StringArrayOrStringHolder{SAOrS: FromStringArray([]string{"foo"})}, `{"val1":["foo"]}`},
		{StringArrayOrStringHolder{SAOrS: FromStringArray([]string{"foo", "bar"})}, `{"val1":["foo","bar"]}`},
		{StringArrayOrStringHolder{SAOrS: FromStringArray([]string{"foo"})}, `{"val1":["foo"]}`},
		{StringArrayOrStringHolder{SAOrS: FromString("foo")}, `{"val1":"foo"}`},
		{StringArrayOrStringHolder{SAOrS: FromString("")}, `{"val1":""}`},

		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{})}, `{"val1":""}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{""})}, `{"val1":"","val2":[""]}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{"foo"})}, `{"val1":"","val2":["foo"]}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{"foo", "bar"})}, `{"val1":"","val2":["foo","bar"]}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringArrayPtr([]string{"foo"})}, `{"val1":"","val2":["foo"]}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringPtr("foo")}, `{"val1":"","val2":"foo"}`},
		{StringArrayOrStringHolder{SAOrSOmitEmpty: fromStringPtr("")}, `{"val1":""}`},
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

func TestStringArrayOrStringMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input StringArrayOrString
	}{
		{FromStringArray([]string{"foo"})},
		{FromString("foo")},
	}

	for _, c := range cases {
		input := StringArrayOrStringHolder{SAOrS: c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("1: Unexpected error marshaling input: '%v': %v", input, err)
		}

		var result StringArrayOrStringHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("2: Failed to unmarshal '%+v': %v", string(jsonMarshalled), err)
		}

		if !reflect.DeepEqual(input, result) {
			t.Errorf("3: Failed to marshal input '%+v': got %+v", input, result)
		}
	}
}
