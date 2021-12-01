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

	authenticationv1 "k8s.io/api/authentication/v1"
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

// Tests the functionality of the OpenAPI v3 GetDefinitionNamev3 function of
// the definition namer.
func TestGetDefinitionNamev3(t *testing.T) {
	testType := openapitesting.TestType{}

	// Setup scheme
	s := runtime.NewScheme()
	s.AddKnownTypeWithName(testType.GroupVersionKind(), &testType)
	authenticationv1.AddToScheme(s)

	// Test a type known statically from this package
	namer := NewDefinitionNamer(s)
	n, e := namer.GetDefinitionNameV3("k8s.io/apiserver/pkg/endpoints/openapi/testing.TestType")
	assertEqual(t, "test.v1.TestType", n)
	assertEqual(t, []interface{}{
		map[string]interface{}{
			"group":   "test",
			"version": "v1",
			"kind":    "TestType",
		},
	}, e["x-kubernetes-group-version-kind"])

	// Test a type not known statically like a CRD
	n, e2 := namer.GetDefinitionNameV3("test.com/another.Type")
	assertEqual(t, "com.test.another.Type", n)
	assertEqual(t, e2, spec.Extensions(nil))

	// Test a type known statically from k8s.io
	n, e3 := namer.GetDefinitionNameV3("k8s.io/api/authentication/v1.TokenRequest")
	assertEqual(t, "io.k8s.authentication.v1.TokenRequest", n)
	assertEqual(t, []interface{}{map[string]interface{}{
		"group":   authenticationv1.SchemeGroupVersion.Group,
		"version": authenticationv1.SchemeGroupVersion.Version,
		"kind":    reflect.TypeOf(&authenticationv1.TokenRequest{}).Elem().Name(),
	}}, e3["x-kubernetes-group-version-kind"])
}
