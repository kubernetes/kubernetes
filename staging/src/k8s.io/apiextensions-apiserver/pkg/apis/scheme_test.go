/*
Copyright The Kubernetes Authors.

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

package api

import (
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	typesv1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1"
	typesv1beta1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

// TestAddToScheme pins down what each AddToScheme registers since the types
// moved to k8s.io/apiextensions: both packages register the types and their
// defaulting functions, only the apiextensions-apiserver packages add the
// conversions to the internal version.
func TestAddToScheme(t *testing.T) {
	tests := []struct {
		name            string
		addToScheme     func(*runtime.Scheme) error
		obj             runtime.Object
		wantConversions bool
	}{
		{name: "apiextensions-apiserver v1", addToScheme: apiextensionsv1.AddToScheme, obj: &typesv1.CustomResourceDefinition{}, wantConversions: true},
		{name: "apiextensions-apiserver v1beta1", addToScheme: apiextensionsv1beta1.AddToScheme, obj: &typesv1beta1.CustomResourceDefinition{}, wantConversions: true},
		{name: "apiextensions v1", addToScheme: typesv1.AddToScheme, obj: &typesv1.CustomResourceDefinition{}},
		{name: "apiextensions v1beta1", addToScheme: typesv1beta1.AddToScheme, obj: &typesv1beta1.CustomResourceDefinition{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			if err := tc.addToScheme(scheme); err != nil {
				t.Fatal(err)
			}

			scheme.Default(tc.obj)
			var defaulted bool
			switch crd := tc.obj.(type) {
			case *typesv1.CustomResourceDefinition:
				defaulted = crd.Spec.Conversion != nil && crd.Spec.Conversion.Strategy == typesv1.NoneConverter
			case *typesv1beta1.CustomResourceDefinition:
				defaulted = crd.Spec.Conversion != nil && crd.Spec.Conversion.Strategy == typesv1beta1.NoneConverter
			}
			if !defaulted {
				t.Errorf("spec.conversion was not defaulted: %#v", tc.obj)
			}

			err := scheme.Convert(tc.obj, &apiextensions.CustomResourceDefinition{}, nil)
			if tc.wantConversions && err != nil {
				t.Errorf("conversion to the internal version failed: %v", err)
			}
			if !tc.wantConversions && err == nil {
				t.Error("conversion to the internal version is registered, want it only in k8s.io/apiextensions-apiserver")
			}
		})
	}
}
