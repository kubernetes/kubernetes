/*
Copyright 2023 The Kubernetes Authors.

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

package kubeadm

import (
	"reflect"
	"testing"
)

func TestGetArgValue(t *testing.T) {
	var tests = []struct {
		testName      string
		args          []Arg
		name          string
		expectedValue string
		startIdx      int
		expectedIdx   int
	}{
		{
			testName:      "argument exists with non-empty value",
			args:          []Arg{{Name: "a", Value: "a1"}, {Name: "b", Value: "b1"}, {Name: "c", Value: "c1"}},
			name:          "b",
			expectedValue: "b1",
			expectedIdx:   1,
			startIdx:      -1,
		},
		{
			testName:      "argument exists with non-empty value (offset index)",
			args:          []Arg{{Name: "a", Value: "a1"}, {Name: "b", Value: "b1"}, {Name: "c", Value: "c1"}},
			name:          "a",
			expectedValue: "a1",
			expectedIdx:   0,
			startIdx:      0,
		},
		{
			testName:      "argument exists with empty value",
			args:          []Arg{{Name: "foo1", Value: ""}, {Name: "foo2", Value: ""}},
			name:          "foo2",
			expectedValue: "",
			expectedIdx:   1,
			startIdx:      -1,
		},
		{
			testName:      "argument does not exists",
			args:          []Arg{{Name: "foo", Value: "bar"}},
			name:          "z",
			expectedValue: "",
			expectedIdx:   -1,
			startIdx:      -1,
		},
	}

	for _, rt := range tests {
		t.Run(rt.testName, func(t *testing.T) {
			value, idx := GetArgValue(rt.args, rt.name, rt.startIdx)
			if idx != rt.expectedIdx {
				t.Errorf("expected index: %v, got: %v", rt.expectedIdx, idx)
			}
			if value != rt.expectedValue {
				t.Errorf("expected value: %s, got: %s", rt.expectedValue, value)
			}
		})
	}
}

func TestSetArgValues(t *testing.T) {
	var tests = []struct {
		testName     string
		args         []Arg
		name         string
		value        string
		nArgs        int
		expectedArgs []Arg
	}{
		{
			testName:     "update 1 argument",
			args:         []Arg{{Name: "foo", Value: "bar1"}, {Name: "foo", Value: "bar2"}},
			name:         "foo",
			value:        "zz",
			nArgs:        1,
			expectedArgs: []Arg{{Name: "foo", Value: "bar1"}, {Name: "foo", Value: "zz"}},
		},
		{
			testName:     "update all arguments",
			args:         []Arg{{Name: "foo", Value: "bar1"}, {Name: "foo", Value: "bar2"}},
			name:         "foo",
			value:        "zz",
			nArgs:        -1,
			expectedArgs: []Arg{{Name: "foo", Value: "zz"}, {Name: "foo", Value: "zz"}},
		},
		{
			testName:     "add new argument",
			args:         []Arg{{Name: "foo", Value: "bar1"}, {Name: "foo", Value: "bar2"}},
			name:         "z",
			value:        "zz",
			nArgs:        -1,
			expectedArgs: []Arg{{Name: "foo", Value: "bar1"}, {Name: "foo", Value: "bar2"}, {Name: "z", Value: "zz"}},
		},
	}

	for _, rt := range tests {
		t.Run(rt.testName, func(t *testing.T) {
			args := SetArgValues(rt.args, rt.name, rt.value, rt.nArgs)
			if !reflect.DeepEqual(args, rt.expectedArgs) {
				t.Errorf("expected args: %#v, got: %#v", rt.expectedArgs, args)
			}
		})
	}
}
