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

package generate

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

type TestStruct struct {
	val int
}

func TestIsZero(t *testing.T) {
	tests := []struct {
		name       string
		val        interface{}
		expectZero bool
	}{
		{
			name:       "test1",
			val:        "",
			expectZero: true,
		},
		{
			name:       "test2",
			val:        nil,
			expectZero: true,
		},
		{
			name:       "test3",
			val:        0,
			expectZero: true,
		},
		{
			name:       "test4",
			val:        TestStruct{},
			expectZero: true,
		},
		{
			name:       "test5",
			val:        "foo",
			expectZero: false,
		},
		{
			name:       "test6",
			val:        1,
			expectZero: false,
		},
		{
			name:       "test7",
			val:        TestStruct{val: 2},
			expectZero: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := IsZero(tt.val)
			if output != tt.expectZero {
				t.Errorf("expected: %v, saw %v", tt.expectZero, output)
			}
		})
	}
}

func TestValidateParams(t *testing.T) {
	tests := []struct {
		name      string
		paramSpec []GeneratorParam
		params    map[string]interface{}
		valid     bool
	}{
		{
			name:      "test1",
			paramSpec: []GeneratorParam{},
			params:    map[string]interface{}{},
			valid:     true,
		},
		{
			name: "test2",
			paramSpec: []GeneratorParam{
				{Name: "foo"},
			},
			params: map[string]interface{}{},
			valid:  true,
		},
		{
			name: "test3",
			paramSpec: []GeneratorParam{
				{Name: "foo", Required: true},
			},
			params: map[string]interface{}{
				"foo": "bar",
			},
			valid: true,
		},
		{
			name: "test4",
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
			name: "test5",
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
			name: "test6",
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
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateParams(tt.paramSpec, tt.params)
			if tt.valid && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !tt.valid && err == nil {
				t.Errorf("unexpected non-error")
			}
		})
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
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetBool(tt.parameters, tt.key, tt.defaultValue)
			if err != nil && !tt.expectError {
				t.Errorf("%s: unexpected error: %v", tt.name, err)
			}
			if err == nil && tt.expectError {
				t.Errorf("%s: expect error, got nil", tt.name)
			}
			if got != tt.expected {
				t.Errorf("%s: expect %v, got %v", tt.name, tt.expected, got)
			}
		})
	}
}

func makeLabels(labels map[string]string) string {
	out := []string{}
	for key, value := range labels {
		out = append(out, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(out, ",")
}

func TestMakeParseLabels(t *testing.T) {
	successCases := []struct {
		name     string
		labels   map[string]string
		expected map[string]string
	}{
		{
			name: "test1",
			labels: map[string]string{
				"foo": "false",
			},
			expected: map[string]string{
				"foo": "false",
			},
		},
		{
			name: "test2",
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
	for _, tt := range successCases {
		t.Run(tt.name, func(t *testing.T) {
			labelString := makeLabels(tt.labels)
			got, err := ParseLabels(labelString)
			if err != nil {
				t.Errorf("unexpected error :%v", err)
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("\nexpected:\n%v\ngot:\n%v", tt.expected, got)
			}
		})
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
		name      string
		protocols map[string]string
		expected  map[string]string
	}{
		{
			name: "test1",
			protocols: map[string]string{
				"101": "TCP",
			},
			expected: map[string]string{
				"101": "TCP",
			},
		},
		{
			name: "test2",
			protocols: map[string]string{
				"102": "UDP",
				"101": "TCP",
				"103": "SCTP",
			},
			expected: map[string]string{
				"102": "UDP",
				"101": "TCP",
				"103": "SCTP",
			},
		},
	}
	for _, tt := range successCases {
		t.Run(tt.name, func(t *testing.T) {
			protocolString := MakeProtocols(tt.protocols)
			got, err := ParseProtocols(protocolString)
			if err != nil {
				t.Errorf("unexpected error :%v", err)
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("\nexpected:\n%v\ngot:\n%v", tt.expected, got)
			}
		})
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
