/*
Copyright 2016 The Kubernetes Authors.

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

package extensions

import (
	"reflect"
	"testing"
)

func TestPodAnnotationsFromSysctls(t *testing.T) {
	type Test struct {
		sysctls       []string
		expectedValue string
	}
	for _, test := range []Test{
		{sysctls: []string{"a.b"}, expectedValue: "a.b"},
		{sysctls: []string{"a.b", "c.d"}, expectedValue: "a.b,c.d"},
		{sysctls: []string{"a.b", "a.b"}, expectedValue: "a.b,a.b"},
		{sysctls: []string{}, expectedValue: ""},
		{sysctls: nil, expectedValue: ""},
	} {
		a := PodAnnotationsFromSysctls(test.sysctls)
		if a != test.expectedValue {
			t.Errorf("wrong value for %v: got=%q wanted=%q", test.sysctls, a, test.expectedValue)
		}
	}
}

func TestSysctlsFromPodSecurityPolicyAnnotation(t *testing.T) {
	type Test struct {
		expectedValue []string
		annotation    string
	}
	for _, test := range []Test{
		{annotation: "a.b", expectedValue: []string{"a.b"}},
		{annotation: "a.b,c.d", expectedValue: []string{"a.b", "c.d"}},
		{annotation: "a.b,a.b", expectedValue: []string{"a.b", "a.b"}},
		{annotation: "", expectedValue: []string{}},
	} {
		sysctls, err := SysctlsFromPodSecurityPolicyAnnotation(test.annotation)
		if err != nil {
			t.Errorf("error for %q: %v", test.annotation, err)
		}
		if !reflect.DeepEqual(sysctls, test.expectedValue) {
			t.Errorf("wrong value for %q: got=%v wanted=%v", test.annotation, sysctls, test.expectedValue)
		}
	}
}
