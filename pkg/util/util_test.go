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
	cases := []struct {
		input  string
		result IntOrString
	}{
		{"val: 123\n", IntOrString{Kind: IntstrInt, IntVal: 123}},
		{"val: \"123\"\n", IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		var result IntOrStringHolder
		if err := yaml.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal IntOrString: got %+v", result)
		}
	}
}

func TestIntOrStringMarshalYAML(t *testing.T) {
	cases := []struct {
		input  IntOrString
		result string
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}, "val: 123\n"},
		{IntOrString{Kind: IntstrString, StrVal: "123"}, "val: \"123\"\n"},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		result, err := yaml.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal IntOrString: got %q", string(result))
		}
	}
}

func TestIntOrStringUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result IntOrString
	}{
		{"{\"val\": 123}", IntOrString{Kind: IntstrInt, IntVal: 123}},
		{"{\"val\": \"123\"}", IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		var result IntOrStringHolder
		if err := json.Unmarshal([]byte(c.input), &result); err != nil {
			t.Errorf("Failed to unmarshal: %v", err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal IntOrString: got %+v", result)
		}
	}
}

func TestIntOrStringMarshalJSON(t *testing.T) {
	cases := []struct {
		input  IntOrString
		result string
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}, "{\"val\":123}"},
		{IntOrString{Kind: IntstrString, StrVal: "123"}, "{\"val\":\"123\"}"},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		result, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("Failed to marshal: %v", err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal IntOrString: got %q", string(result))
		}
	}
}
