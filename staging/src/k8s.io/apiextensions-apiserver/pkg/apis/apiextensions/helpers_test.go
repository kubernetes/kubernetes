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

package apiextensions

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCRDHasFinalizer(t *testing.T) {
	tests := []struct {
		name             string
		crd              *CustomResourceDefinition
		finalizerToCheck string

		expected bool
	}{
		{
			name: "missing",
			crd: &CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         false,
		},
		{
			name: "present",
			crd: &CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it", "it"}},
			},
			finalizerToCheck: "it",
			expected:         true,
		},
	}
	for _, tc := range tests {
		actual := CRDHasFinalizer(tc.crd, tc.finalizerToCheck)
		if tc.expected != actual {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expected, actual)
		}
	}
}

func TestCRDRemoveFinalizer(t *testing.T) {
	tests := []struct {
		name             string
		crd              *CustomResourceDefinition
		finalizerToCheck string

		expected []string
	}{
		{
			name: "missing",
			crd: &CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it"}},
			},
			finalizerToCheck: "it",
			expected:         []string{"not-it"},
		},
		{
			name: "present",
			crd: &CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Finalizers: []string{"not-it", "it"}},
			},
			finalizerToCheck: "it",
			expected:         []string{"not-it"},
		},
	}
	for _, tc := range tests {
		CRDRemoveFinalizer(tc.crd, tc.finalizerToCheck)
		if !reflect.DeepEqual(tc.expected, tc.crd.Finalizers) {
			t.Errorf("%v expected %v, got %v", tc.name, tc.expected, tc.crd.Finalizers)
		}
	}
}
