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

package group

import (
	"strings"
	"testing"

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMustRunAsOptions(t *testing.T) {
	tests := map[string]struct {
		ranges []policy.IDRange
		pass   bool
	}{
		"empty": {
			ranges: []policy.IDRange{},
		},
		"ranges": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 1},
			},
			pass: true,
		},
	}

	for k, v := range tests {
		_, err := NewMustRunAs(v.ranges)
		if v.pass && err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		if !v.pass && err == nil {
			t.Errorf("expected error for %s but got none", k)
		}
	}
}

func TestGenerate(t *testing.T) {
	tests := map[string]struct {
		ranges   []policy.IDRange
		expected []int64
	}{
		"multi value": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 2},
			},
			expected: []int64{1},
		},
		"single value": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 1},
			},
			expected: []int64{1},
		},
		"multi range": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 1},
				{Min: 2, Max: 500},
			},
			expected: []int64{1},
		},
	}

	for k, v := range tests {
		s, err := NewMustRunAs(v.ranges)
		if err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		actual, err := s.Generate(nil)
		if err != nil {
			t.Errorf("unexpected error for %s: %v", k, err)
		}
		if len(actual) != len(v.expected) {
			t.Errorf("unexpected generated values.  Expected %v, got %v", v.expected, actual)
			continue
		}
		if len(actual) > 0 && len(v.expected) > 0 {
			if actual[0] != v.expected[0] {
				t.Errorf("unexpected generated values.  Expected %v, got %v", v.expected, actual)
			}
		}

		single, err := s.GenerateSingle(nil)
		if err != nil {
			t.Errorf("unexpected error for %s: %v", k, err)
		}
		if single == nil {
			t.Errorf("unexpected nil generated value for %s: %v", k, single)
		}
		if *single != v.expected[0] {
			t.Errorf("unexpected generated single value.  Expected %v, got %v", v.expected, actual)
		}
	}
}

func TestValidate(t *testing.T) {
	tests := map[string]struct {
		ranges        []policy.IDRange
		groups        []int64
		expectedError string
	}{
		"nil security context": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
			expectedError: "unable to validate empty groups against required ranges",
		},
		"empty groups": {
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
			expectedError: "unable to validate empty groups against required ranges",
		},
		"not in range": {
			groups: []int64{5},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
				{Min: 4, Max: 4},
			},
			expectedError: "group 5 must be in the ranges: [{1 3} {4 4}]",
		},
		"in range 1": {
			groups: []int64{2},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"in range boundary min": {
			groups: []int64{1},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"in range boundary max": {
			groups: []int64{3},
			ranges: []policy.IDRange{
				{Min: 1, Max: 3},
			},
		},
		"singular range": {
			groups: []int64{4},
			ranges: []policy.IDRange{
				{Min: 4, Max: 4},
			},
		},
	}

	for k, v := range tests {
		s, err := NewMustRunAs(v.ranges)
		if err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		errs := s.Validate(field.NewPath(""), nil, v.groups)
		if v.expectedError == "" && len(errs) > 0 {
			t.Errorf("unexpected errors for %s: %v", k, errs)
		}
		if v.expectedError != "" && len(errs) == 0 {
			t.Errorf("expected errors for %s but got: %v", k, errs)
		}
		if v.expectedError != "" && len(errs) > 0 && !strings.Contains(errs[0].Error(), v.expectedError) {
			t.Errorf("expected error for %s: %v, but got: %v", k, v.expectedError, errs[0])
		}
	}
}
