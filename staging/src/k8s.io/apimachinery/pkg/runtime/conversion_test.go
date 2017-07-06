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

package runtime_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
)

func TestStringMapConversion(t *testing.T) {
	internalGV := schema.GroupVersion{Group: "test.group", Version: runtime.APIVersionInternal}
	externalGV := schema.GroupVersion{Group: "test.group", Version: "external"}

	scheme := runtime.NewScheme()
	scheme.Log(t)
	scheme.AddKnownTypeWithName(internalGV.WithKind("Complex"), &runtimetesting.InternalComplex{})
	scheme.AddKnownTypeWithName(externalGV.WithKind("Complex"), &runtimetesting.ExternalComplex{})

	testCases := map[string]struct {
		input    map[string][]string
		errFn    func(error) bool
		expected runtime.Object
	}{
		"ignores omitempty": {
			input: map[string][]string{
				"String":    {"not_used"},
				"string":    {"value"},
				"int":       {"1"},
				"Integer64": {"2"},
			},
			expected: &runtimetesting.ExternalComplex{String: "value", Integer: 1},
		},
		"returns error on bad int": {
			input: map[string][]string{
				"int": {"a"},
			},
			errFn:    func(err error) bool { return err != nil },
			expected: &runtimetesting.ExternalComplex{},
		},
		"parses int64": {
			input: map[string][]string{
				"Int64": {"-1"},
			},
			expected: &runtimetesting.ExternalComplex{Int64: -1},
		},
		"returns error on bad int64": {
			input: map[string][]string{
				"Int64": {"a"},
			},
			errFn:    func(err error) bool { return err != nil },
			expected: &runtimetesting.ExternalComplex{},
		},
		"parses boolean true": {
			input: map[string][]string{
				"bool": {"true"},
			},
			expected: &runtimetesting.ExternalComplex{Bool: true},
		},
		"parses boolean any value": {
			input: map[string][]string{
				"bool": {"foo"},
			},
			expected: &runtimetesting.ExternalComplex{Bool: true},
		},
		"parses boolean false": {
			input: map[string][]string{
				"bool": {"false"},
			},
			expected: &runtimetesting.ExternalComplex{Bool: false},
		},
		"parses boolean empty value": {
			input: map[string][]string{
				"bool": {""},
			},
			expected: &runtimetesting.ExternalComplex{Bool: true},
		},
		"parses boolean no value": {
			input: map[string][]string{
				"bool": {},
			},
			expected: &runtimetesting.ExternalComplex{Bool: false},
		},
	}

	for k, tc := range testCases {
		out := &runtimetesting.ExternalComplex{}
		if err := scheme.Convert(&tc.input, out, nil); (tc.errFn == nil && err != nil) || (tc.errFn != nil && !tc.errFn(err)) {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		} else if err != nil {
			continue
		}
		if !reflect.DeepEqual(out, tc.expected) {
			t.Errorf("%s: unexpected output: %#v", k, out)
		}
	}
}
