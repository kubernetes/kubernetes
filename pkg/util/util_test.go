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

package util

import (
	"encoding/json"
	"testing"

	"gopkg.in/v1/yaml"
)

type FakeJSONBase struct {
	ID string
}
type FakePod struct {
	FakeJSONBase `json:",inline" yaml:",inline"`
	Labels       map[string]string
	Int          int
	Str          string
}

func TestMakeJSONString(t *testing.T) {
	pod := FakePod{
		FakeJSONBase: FakeJSONBase{ID: "foo"},
		Labels: map[string]string{
			"foo": "bar",
			"baz": "blah",
		},
		Int: -6,
		Str: "a string",
	}

	body := MakeJSONString(pod)

	expectedBody, err := json.Marshal(pod)
	expectNoError(t, err)
	if string(expectedBody) != body {
		t.Errorf("JSON doesn't match.  Expected %s, saw %s", expectedBody, body)
	}
}

func TestHandleCrash(t *testing.T) {
	count := 0
	expect := 10
	for i := 0; i < expect; i = i + 1 {
		defer HandleCrash()
		if i%2 == 0 {
			panic("Test Panic")
		}
		count = count + 1
	}
	if count != expect {
		t.Errorf("Expected %d iterations, found %d", expect, count)
	}
}

type IntOrStringHolder struct {
	IOrS IntOrString `json:"val" yaml:"val"`
}

func TestIntOrStringUnmarshalYAML(t *testing.T) {
	{
		yaml_code_int := "val: 123\n"

		var result IntOrStringHolder
		if err := yaml.Unmarshal([]byte(yaml_code_int), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS.Kind != IntstrInt || result.IOrS.IntVal != 123 {
			t.Errorf("Failed to unmarshal int-typed IntOrString: %v", result)
		}
	}

	{
		yaml_code_str := "val: \"123\"\n"

		var result IntOrStringHolder
		if err := yaml.Unmarshal([]byte(yaml_code_str), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS.Kind != IntstrString || result.IOrS.StrVal != "123" {
			t.Errorf("Failed to unmarshal string-typed IntOrString: %v", result)
		}
	}
}

func TestIntOrStringMarshalYAML(t *testing.T) {
	{
		input := IntOrStringHolder{
			IOrS: IntOrString{
				Kind:   IntstrInt,
				IntVal: 123,
			},
		}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != "val: 123\n" {
			t.Errorf("Failed to marshal int-typed IntOrString: %q", string(result))
		}
	}

	{
		input := IntOrStringHolder{
			IOrS: IntOrString{
				Kind:   IntstrString,
				StrVal: "123",
			},
		}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != "val: \"123\"\n" {
			t.Errorf("Failed to marshal string-typed IntOrString: %q", string(result))
		}
	}
}

func TestIntOrStringUnmarshalJSON(t *testing.T) {
	{
		json_code_int := "{\"val\": 123}"

		var result IntOrStringHolder
		if err := json.Unmarshal([]byte(json_code_int), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS.Kind != IntstrInt || result.IOrS.IntVal != 123 {
			t.Errorf("Failed to unmarshal int-typed IntOrString: %v", result)
		}
	}

	{
		json_code_str := "{\"val\": \"123\"}"

		var result IntOrStringHolder
		if err := json.Unmarshal([]byte(json_code_str), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS.Kind != IntstrString || result.IOrS.StrVal != "123" {
			t.Errorf("Failed to unmarshal string-typed IntOrString: %v", result)
		}
	}
}

func TestIntOrStringMarshalJSON(t *testing.T) {
	{
		input := IntOrStringHolder{
			IOrS: IntOrString{
				Kind:   IntstrInt,
				IntVal: 123,
			},
		}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != "{\"val\":123}" {
			t.Errorf("Failed to marshal int-typed IntOrString: %q", string(result))
		}
	}

	{
		input := IntOrStringHolder{
			IOrS: IntOrString{
				Kind:   IntstrString,
				StrVal: "123",
			},
		}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != "{\"val\":\"123\"}" {
			t.Errorf("Failed to marshal string-typed IntOrString: %q", string(result))
		}
	}
}
