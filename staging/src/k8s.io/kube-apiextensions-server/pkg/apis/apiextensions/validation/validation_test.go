/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
)

type validationMatch struct {
	path      *field.Path
	errorType field.ErrorType
}

func required(path *field.Path) validationMatch {
	return validationMatch{path: path, errorType: field.ErrorTypeRequired}
}
func invalid(path *field.Path) validationMatch {
	return validationMatch{path: path, errorType: field.ErrorTypeInvalid}
}

func (v validationMatch) matches(err *field.Error) bool {
	return err.Type == v.errorType && err.Field == v.path.String()
}

func TestValidateCustomResourceDefinition(t *testing.T) {
	tests := []struct {
		name     string
		resource *apiextensions.CustomResourceDefinition
		errors   []validationMatch
	}{
		{
			name: "mismatched name",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.not.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural: "plural",
					},
				},
			},
			errors: []validationMatch{
				invalid(field.NewPath("metadata", "name")),
			},
		},
		{
			name: "missing values",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
			},
			errors: []validationMatch{
				required(field.NewPath("spec", "group")),
				required(field.NewPath("spec", "version")),
				{path: field.NewPath("spec", "scope"), errorType: field.ErrorTypeNotSupported},
				required(field.NewPath("spec", "names", "plural")),
				required(field.NewPath("spec", "names", "singular")),
				required(field.NewPath("spec", "names", "kind")),
				required(field.NewPath("spec", "names", "listKind")),
			},
		},
		{
			name: "bad names 01",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group",
					Version: "ve()*rsion",
					Scope:   apiextensions.ResourceScope("foo"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "pl()*ural",
						Singular: "value()*a",
						Kind:     "value()*a",
						ListKind: "value()*a",
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "pl()*ural",
						Singular: "value()*a",
						Kind:     "value()*a",
						ListKind: "value()*a",
					},
				},
			},
			errors: []validationMatch{
				invalid(field.NewPath("spec", "group")),
				invalid(field.NewPath("spec", "version")),
				{path: field.NewPath("spec", "scope"), errorType: field.ErrorTypeNotSupported},
				invalid(field.NewPath("spec", "names", "plural")),
				invalid(field.NewPath("spec", "names", "singular")),
				invalid(field.NewPath("spec", "names", "kind")),
				invalid(field.NewPath("spec", "names", "listKind")),
				invalid(field.NewPath("status", "acceptedNames", "plural")),
				invalid(field.NewPath("status", "acceptedNames", "singular")),
				invalid(field.NewPath("status", "acceptedNames", "kind")),
				invalid(field.NewPath("status", "acceptedNames", "listKind")),
			},
		},
		{
			name: "bad names 02",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.c(*&om",
					Version: "version",
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "matching",
						ListKind: "matching",
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "matching",
						ListKind: "matching",
					},
				},
			},
			errors: []validationMatch{
				invalid(field.NewPath("spec", "group")),
				invalid(field.NewPath("spec", "names", "listKind")),
				invalid(field.NewPath("status", "acceptedNames", "listKind")),
			},
		},
	}

	for _, tc := range tests {
		errs := ValidateCustomResourceDefinition(tc.resource)

		for _, expectedError := range tc.errors {
			found := false
			for _, err := range errs {
				if expectedError.matches(err) {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("%s: expected %v at %v, got %v", tc.name, expectedError.errorType, expectedError.path.String(), errs)
			}
		}
	}
}
