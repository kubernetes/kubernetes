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

package kubectl

import (
	"reflect"
	"testing"

	"github.com/spf13/cobra"
)

type TestStruct struct {
	val int
}

func TestIsZero(t *testing.T) {
	tests := []struct {
		val        interface{}
		expectZero bool
	}{
		{"", true},
		{nil, true},
		{0, true},
		{TestStruct{}, true},
		{"foo", false},
		{1, false},
		{TestStruct{val: 2}, false},
	}

	for _, test := range tests {
		output := IsZero(test.val)
		if output != test.expectZero {
			t.Errorf("expected: %v, saw %v", test.expectZero, output)
		}
	}
}

func TestValidateParams(t *testing.T) {
	tests := []struct {
		paramSpec []GeneratorParam
		params    map[string]interface{}
		valid     bool
	}{
		{
			paramSpec: []GeneratorParam{},
			params:    map[string]interface{}{},
			valid:     true,
		},
		{
			paramSpec: []GeneratorParam{
				{Name: "foo"},
			},
			params: map[string]interface{}{},
			valid:  true,
		},
		{
			paramSpec: []GeneratorParam{
				{Name: "foo", Required: true},
			},
			params: map[string]interface{}{
				"foo": "bar",
			},
			valid: true,
		},
		{
			paramSpec: []GeneratorParam{
				{Name: "foo", Required: true},
			},
			params: map[string]interface{}{
				"baz": "blah",
				"foo": "bar",
			},
			valid: true,
		},
		{
			paramSpec: []GeneratorParam{
				{Name: "foo", Required: true},
				{Name: "baz", Required: true},
			},
			params: map[string]interface{}{
				"baz": "blah",
				"foo": "bar",
			},
			valid: true,
		},
		{
			paramSpec: []GeneratorParam{
				{Name: "foo", Required: true},
				{Name: "baz", Required: true},
			},
			params: map[string]interface{}{
				"foo": "bar",
			},
			valid: false,
		},
	}
	for _, test := range tests {
		err := ValidateParams(test.paramSpec, test.params)
		if test.valid && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !test.valid && err == nil {
			t.Errorf("unexpected non-error")
		}
	}
}

func TestMakeParams(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.Flags().String("foo", "bar", "")
	cmd.Flags().String("baz", "", "")
	cmd.Flags().Set("baz", "blah")

	paramSpec := []GeneratorParam{
		{Name: "foo", Required: true},
		{Name: "baz", Required: true},
	}
	expected := map[string]interface{}{
		"foo": "bar",
		"baz": "blah",
	}
	params := MakeParams(cmd, paramSpec)
	if !reflect.DeepEqual(params, expected) {
		t.Errorf("\nexpected:\n%v\nsaw:\n%v", expected, params)
	}
}

func TestGetBool(t *testing.T) {
	testCases := []struct {
		name         string
		parameters   map[string]string
		key          string
		defaultValue bool
		expected     bool
		expectError  bool
	}{
		{
			name: "found key in parameters, default value is different from key value",
			parameters: map[string]string{
				"foo": "false",
			},
			key:          "foo",
			defaultValue: false,
			expected:     false,
			expectError:  false,
		},
		{
			name: "found key in parameters, default value is same with key value",
			parameters: map[string]string{
				"foo": "true",
			},
			key:          "foo",
			defaultValue: true,
			expected:     true,
			expectError:  false,
		},
		{
			name: "key not found in parameters, default value is true",
			parameters: map[string]string{
				"foo": "true",
				"far": "false",
			},
			key:          "bar",
			defaultValue: true,
			expected:     true,
			expectError:  false,
		},
		{
			name: "key not found in parameters, default value is false",
			parameters: map[string]string{
				"foo": "true",
				"far": "false",
			},
			key:          "bar",
			defaultValue: false,
			expected:     false,
			expectError:  false,
		},
		{
			name:         "parameters is empty",
			parameters:   map[string]string{},
			key:          "foo",
			defaultValue: true,
			expected:     true,
			expectError:  false,
		},
		{
			name: "parameters key is not a valid bool value",
			parameters: map[string]string{
				"foo": "error",
			},
			key:          "foo",
			defaultValue: true,
			expected:     false,
			expectError:  true,
		},
	}
	for _, test := range testCases {
		got, err := GetBool(test.parameters, test.key, test.defaultValue)
		if err != nil && test.expectError == false {
			t.Errorf("%s: unexpected error: %v", test.name, err)
		}
		if err == nil && test.expectError == true {
			t.Errorf("%s: expect error, got nil", test.name)
		}
		if got != test.expected {
			t.Errorf("%s: expect %v, got %v", test.name, test.expected, got)
		}
	}
}

func TestMakeParseLabels(t *testing.T) {
	successCases := []struct {
		labels   map[string]string
		expected map[string]string
	}{
		{
			labels: map[string]string{
				"foo": "false",
			},
			expected: map[string]string{
				"foo": "false",
			},
		},
		{
			labels: map[string]string{
				"foo": "true",
				"bar": "123",
			},
			expected: map[string]string{
				"foo": "true",
				"bar": "123",
			},
		},
	}
	for _, test := range successCases {
		labelString := MakeLabels(test.labels)
		got, err := ParseLabels(labelString)
		if err != nil {
			t.Errorf("unexpected error :%v", err)
		}
		if !reflect.DeepEqual(test.expected, got) {
			t.Errorf("\nexpected:\n%v\ngot:\n%v", test.expected, got)
		}
	}

	errorCases := []struct {
		name   string
		labels interface{}
	}{
		{
			name:   "non-string",
			labels: 123,
		},
		{
			name:   "empty string",
			labels: "",
		},
		{
			name:   "error format",
			labels: "abc=456;bcd=789",
		},
		{
			name:   "error format",
			labels: "abc=456.bcd=789",
		},
		{
			name:   "error format",
			labels: "abc,789",
		},
		{
			name:   "error format",
			labels: "abc",
		},
		{
			name:   "error format",
			labels: "=abc",
		},
	}
	for _, test := range errorCases {
		_, err := ParseLabels(test.labels)
		if err == nil {
			t.Errorf("labels %s expect error, reason: %s, got nil", test.labels, test.name)
		}
	}
}

func TestMakeParseProtocols(t *testing.T) {
	successCases := []struct {
		protocols map[string]string
		expected  map[string]string
	}{
		{
			protocols: map[string]string{
				"101": "TCP",
			},
			expected: map[string]string{
				"101": "TCP",
			},
		},
		{
			protocols: map[string]string{
				"102": "UDP",
				"101": "TCP",
			},
			expected: map[string]string{
				"102": "UDP",
				"101": "TCP",
			},
		},
	}
	for _, test := range successCases {
		protocolString := MakeProtocols(test.protocols)
		got, err := ParseProtocols(protocolString)
		if err != nil {
			t.Errorf("unexpected error :%v", err)
		}
		if !reflect.DeepEqual(test.expected, got) {
			t.Errorf("\nexpected:\n%v\ngot:\n%v", test.expected, got)
		}
	}

	errorCases := []struct {
		name      string
		protocols interface{}
	}{
		{
			name:      "non-string",
			protocols: 123,
		},
		{
			name:      "empty string",
			protocols: "",
		},
		{
			name:      "error format",
			protocols: "123/TCP;456/UDP",
		},
		{
			name:      "error format",
			protocols: "123/TCP.456/UDP",
		},
		{
			name:      "error format",
			protocols: "123=456",
		},
		{
			name:      "error format",
			protocols: "123",
		},
		{
			name:      "error format",
			protocols: "123=",
		},
		{
			name:      "error format",
			protocols: "=TCP",
		},
	}
	for _, test := range errorCases {
		_, err := ParseProtocols(test.protocols)
		if err == nil {
			t.Errorf("protocols %s expect error, reason: %s, got nil", test.protocols, test.name)
		}
	}
}
