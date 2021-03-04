/*
Copyright 2014 The Kubernetes Authors.

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

package intstr

import (
	"encoding/json"
	"reflect"
	"testing"

	"sigs.k8s.io/yaml"
)

func TestFromInt(t *testing.T) {
	i := FromInt(93)
	if i.Type != Int || i.IntVal != 93 {
		t.Errorf("Expected IntVal=93, got %+v", i)
	}
}

func TestFromString(t *testing.T) {
	i := FromString("76")
	if i.Type != String || i.StrVal != "76" {
		t.Errorf("Expected StrVal=\"76\", got %+v", i)
	}
}

type IntOrStringHolder struct {
	IOrS IntOrString `json:"val"`
}

func TestIntOrStringUnmarshalJSON(t *testing.T) {
	cases := []struct {
		input  string
		result IntOrString
	}{
		{"{\"val\": 123}", FromInt(123)},
		{"{\"val\": \"123\"}", FromString("123")},
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
		{FromInt(123), "{\"val\":123}"},
		{FromString("123"), "{\"val\":\"123\"}"},
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
		{FromInt(123)},
		{FromString("123")},
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

func TestGetIntFromIntOrString(t *testing.T) {
	tests := []struct {
		input      IntOrString
		expectErr  bool
		expectVal  int
		expectPerc bool
	}{
		{
			input:      FromInt(200),
			expectErr:  false,
			expectVal:  200,
			expectPerc: false,
		},
		{
			input:      FromString("200"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("30%0"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("40%"),
			expectErr:  false,
			expectVal:  40,
			expectPerc: true,
		},
		{
			input:      FromString("%"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("a%"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("a"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("40#"),
			expectErr:  true,
			expectPerc: false,
		},
		{
			input:      FromString("40%%"),
			expectErr:  true,
			expectPerc: false,
		},
	}
	for _, test := range tests {
		t.Run("", func(t *testing.T) {
			value, isPercent, err := getIntOrPercentValueSafely(&test.input)
			if test.expectVal != value {
				t.Fatalf("expected value does not match, expected: %d, got: %d", test.expectVal, value)
			}
			if test.expectPerc != isPercent {
				t.Fatalf("expected percent does not match, expected: %t, got: %t", test.expectPerc, isPercent)
			}
			if test.expectErr != (err != nil) {
				t.Fatalf("expected error does not match, expected error: %v, got: %v", test.expectErr, err)
			}
		})
	}

}

func TestGetIntFromIntOrPercent(t *testing.T) {
	tests := []struct {
		input     IntOrString
		total     int
		roundUp   bool
		expectErr bool
		expectVal int
	}{
		{
			input:     FromInt(123),
			expectErr: false,
			expectVal: 123,
		},
		{
			input:     FromString("90%"),
			total:     100,
			roundUp:   true,
			expectErr: false,
			expectVal: 90,
		},
		{
			input:     FromString("90%"),
			total:     95,
			roundUp:   true,
			expectErr: false,
			expectVal: 86,
		},
		{
			input:     FromString("90%"),
			total:     95,
			roundUp:   false,
			expectErr: false,
			expectVal: 85,
		},
		{
			input:     FromString("%"),
			expectErr: true,
		},
		{
			input:     FromString("90#"),
			expectErr: true,
		},
		{
			input:     FromString("#%"),
			expectErr: true,
		},
		{
			input:     FromString("90"),
			expectErr: true,
		},
	}

	for i, test := range tests {
		t.Logf("test case %d", i)
		value, err := GetScaledValueFromIntOrPercent(&test.input, test.total, test.roundUp)
		if test.expectErr && err == nil {
			t.Errorf("expected error, but got none")
			continue
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected err: %v", err)
			continue
		}
		if test.expectVal != value {
			t.Errorf("expected %v, but got %v", test.expectVal, value)
		}
	}
}

func TestGetValueFromIntOrPercentNil(t *testing.T) {
	_, err := GetScaledValueFromIntOrPercent(nil, 0, false)
	if err == nil {
		t.Errorf("expected error got none")
	}
}
