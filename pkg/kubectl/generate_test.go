/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
