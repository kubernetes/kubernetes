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
	"reflect"
	"testing"

	"gopkg.in/v1/yaml"
)

type FakeTypeMeta struct {
	ID string
}
type FakePod struct {
	FakeTypeMeta `json:",inline" yaml:",inline"`
	Labels       map[string]string
	Int          int
	Str          string
}

func TestEncodeJSON(t *testing.T) {
	pod := FakePod{
		FakeTypeMeta: FakeTypeMeta{ID: "foo"},
		Labels: map[string]string{
			"foo": "bar",
			"baz": "blah",
		},
		Int: -6,
		Str: "a string",
	}

	body := EncodeJSON(pod)

	expectedBody, err := json.Marshal(pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

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

func TestNewIntOrStringFromInt(t *testing.T) {
	i := NewIntOrStringFromInt(93)
	if i.Kind != IntstrInt || i.IntVal != 93 {
		t.Errorf("Expected IntVal=93, got %+v", i)
	}
}

func TestNewIntOrStringFromString(t *testing.T) {
	i := NewIntOrStringFromString("76")
	if i.Kind != IntstrString || i.StrVal != "76" {
		t.Errorf("Expected StrVal=\"76\", got %+v", i)
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
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected: %+v, got %+v", c.input, c.result, result)
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
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %+v, got %q", input, c.result, string(result))
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
			t.Errorf("Failed to unmarshal input '%v': %v", c.input, err)
		}
		if result.IOrS != c.result {
			t.Errorf("Failed to unmarshal input '%v': expected %+v, got %+v", c.input, c.result, result)
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
			t.Errorf("Failed to marshal input '%v': %v", input, err)
		}
		if string(result) != c.result {
			t.Errorf("Failed to marshal input '%v': expected: %+v, got %q", input, c.result, string(result))
		}
	}
}

func TestIntOrStringMarshalJSONUnmarshalYAML(t *testing.T) {
	cases := []struct {
		input IntOrString
	}{
		{IntOrString{Kind: IntstrInt, IntVal: 123}},
		{IntOrString{Kind: IntstrString, StrVal: "123"}},
	}

	for _, c := range cases {
		input := IntOrStringHolder{c.input}
		jsonMarshalled, err := json.Marshal(&input)
		if err != nil {
			t.Errorf("1: Failed to marshal input: '%v': %v", input, err)
		}

		var result IntOrStringHolder
		err = yaml.Unmarshal(jsonMarshalled, &result)
		if err != nil {
			t.Errorf("2: Failed to unmarshal '%+v': %v", string(jsonMarshalled), err)
		}

		if !reflect.DeepEqual(input, result) {
			t.Errorf("3: Failed to marshal input '%+v': got %+v", input, result)
		}
	}
}

func TestStringDiff(t *testing.T) {
	diff := StringDiff("aaabb", "aaacc")
	expect := "aaa\n\nA: bb\n\nB: cc\n\n"
	if diff != expect {
		t.Errorf("diff returned %v", diff)
	}
}

func TestCompileRegex(t *testing.T) {
	uncompiledRegexes := []string{"endsWithMe$", "^startingWithMe"}
	regexes, err := CompileRegexps(uncompiledRegexes)

	if err != nil {
		t.Errorf("Failed to compile legal regexes: '%v': %v", uncompiledRegexes, err)
	}
	if len(regexes) != len(uncompiledRegexes) {
		t.Errorf("Wrong number of regexes returned: '%v': %v", uncompiledRegexes, regexes)
	}

	if !regexes[0].MatchString("Something that endsWithMe") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if regexes[0].MatchString("Something that doesn't endsWithMe.") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[0], regexes[0])
	}
	if !regexes[1].MatchString("startingWithMe is very important") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
	if regexes[1].MatchString("not startingWithMe should fail") {
		t.Errorf("Wrong regex returned: '%v': %v", uncompiledRegexes[1], regexes[1])
	}
}
