/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"reflect"
	"testing"
)

func TestCanonicalName(t *testing.T) {

	var tests = []struct {
		input    string
		expected string
	}{
		{"k8s.io/api/core/v1.Pod", "io.k8s.api.core.v1.Pod"},
		{"k8s.io/api/networking/v1/NetworkPolicy", "io.k8s.api.networking.v1.NetworkPolicy"},
		{"k8s.io/api/apps/v1beta2.Scale", "io.k8s.api.apps.v1beta2.Scale"},
		{"servicecatalog.k8s.io/foo/bar/v1alpha1.Baz", "io.k8s.servicecatalog.foo.bar.v1alpha1.Baz"},
	}
	for _, test := range tests {
		if got := ToRESTFriendlyName(test.input); got != test.expected {
			t.Errorf("ToRESTFriendlyName(%q) = %v", test.input, got)
		}
	}
}

type TestType struct{}

func TestGetCanonicalTypeName(t *testing.T) {

	var tests = []struct {
		input    interface{}
		expected string
	}{
		{TestType{}, "k8s.io/kube-openapi/pkg/util.TestType"},
		{&TestType{}, "k8s.io/kube-openapi/pkg/util.TestType"},
	}
	for _, test := range tests {
		if got := GetCanonicalTypeName(test.input); got != test.expected {
			t.Errorf("GetCanonicalTypeName(%q) = %v", reflect.TypeOf(test.input), got)
		}
	}
}
