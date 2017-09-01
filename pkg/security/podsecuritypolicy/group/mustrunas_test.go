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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestMustRunAsOptions(t *testing.T) {
	tests := map[string]struct {
		ranges []extensions.GroupIDRange
		pass   bool
	}{
		"empty": {
			ranges: []extensions.GroupIDRange{},
		},
		"ranges": {
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 1},
			},
			pass: true,
		},
	}

	for k, v := range tests {
		_, err := NewMustRunAs(v.ranges, "")
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
		ranges   []extensions.GroupIDRange
		expected []int64
	}{
		"multi value": {
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 2},
			},
			expected: []int64{1},
		},
		"single value": {
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 1},
			},
			expected: []int64{1},
		},
		"multi range": {
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 1},
				{Min: 2, Max: 500},
			},
			expected: []int64{1},
		},
	}

	for k, v := range tests {
		s, err := NewMustRunAs(v.ranges, "")
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
	validPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
			},
		}
	}

	tests := map[string]struct {
		ranges []extensions.GroupIDRange
		pod    *api.Pod
		groups []int64
		pass   bool
	}{
		"nil security context": {
			pod: &api.Pod{},
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
			},
		},
		"empty groups": {
			pod: validPod(),
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
			},
		},
		"not in range": {
			pod:    validPod(),
			groups: []int64{5},
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
				{Min: 4, Max: 4},
			},
		},
		"in range 1": {
			pod:    validPod(),
			groups: []int64{2},
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
			},
			pass: true,
		},
		"in range boundry min": {
			pod:    validPod(),
			groups: []int64{1},
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
			},
			pass: true,
		},
		"in range boundry max": {
			pod:    validPod(),
			groups: []int64{3},
			ranges: []extensions.GroupIDRange{
				{Min: 1, Max: 3},
			},
			pass: true,
		},
		"singular range": {
			pod:    validPod(),
			groups: []int64{4},
			ranges: []extensions.GroupIDRange{
				{Min: 4, Max: 4},
			},
			pass: true,
		},
	}

	for k, v := range tests {
		s, err := NewMustRunAs(v.ranges, "")
		if err != nil {
			t.Errorf("error creating strategy for %s: %v", k, err)
		}
		errs := s.Validate(v.pod, v.groups)
		if v.pass && len(errs) > 0 {
			t.Errorf("unexpected errors for %s: %v", k, errs)
		}
		if !v.pass && len(errs) == 0 {
			t.Errorf("expected no errors for %s but got: %v", k, errs)
		}
	}
}
