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

package openapi

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	openapitesting "k8s.io/apiserver/pkg/endpoints/openapi/testing"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func assertEqual(t *testing.T, expected, actual interface{}) {
	var equal bool
	if expected == nil || actual == nil {
		equal = expected == actual
	} else {
		equal = reflect.DeepEqual(expected, actual)
	}
	if !equal {
		t.Errorf("%v != %v", expected, actual)
	}
}

func TestGetDefinitionName(t *testing.T) {
	testType := openapitesting.TestType{}
	// in production, the name is stripped of ".*vendor/" prefix before passed
	// to GetDefinitionName, so here typePkgName does not have the
	// "k8s.io/kubernetes/vendor" prefix.
	typePkgName := "k8s.io/apiserver/pkg/endpoints/openapi/testing.TestType"
	typeFriendlyName := "io.k8s.apiserver.pkg.endpoints.openapi.testing.TestType"
	s := runtime.NewScheme()
	s.AddKnownTypeWithName(testType.GroupVersionKind(), &testType)
	namer := NewDefinitionNamer(s)
	n, e := namer.GetDefinitionName(typePkgName)
	assertEqual(t, typeFriendlyName, n)
	assertEqual(t, []interface{}{
		map[string]interface{}{
			"group":   "test",
			"version": "v1",
			"kind":    "TestType",
		},
	}, e["x-kubernetes-group-version-kind"])
	n, e2 := namer.GetDefinitionName("test.com/another.Type")
	assertEqual(t, "com.test.another.Type", n)
	assertEqual(t, e2, spec.Extensions(nil))
}

func TestToValidOperationID(t *testing.T) {
	scenarios := []struct {
		s                     string
		capitalizeFirstLetter bool
		expectedResult        string
	}{
		{
			s:                     "test_operation",
			capitalizeFirstLetter: true,
			expectedResult:        "Test_operation",
		},
		{
			s:                     "test operation& test",
			capitalizeFirstLetter: true,
			expectedResult:        "TestOperationTest",
		},
		{
			s:                     "test78operation",
			capitalizeFirstLetter: false,
			expectedResult:        "test78operation",
		},
	}
	for _, tt := range scenarios {
		result := ToValidOperationID(tt.s, tt.capitalizeFirstLetter)
		if result != tt.expectedResult {
			t.Errorf("expected result: %s, got: %s", tt.expectedResult, result)
		}
	}
}
