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

package validation

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestValidateStorageClass(t *testing.T) {
	successCases := []storage.StorageClass{
		{
			// empty parameters
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo-provisioner",
			Parameters:  map[string]string{},
		},
		{
			// nil parameters
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo-provisioner",
		},
		{
			// some parameters
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo-provisioner",
			Parameters: map[string]string{
				"kubernetes.io/foo-parameter": "free/form/string",
				"foo-parameter":               "free-form-string",
				"foo-parameter2":              "{\"embedded\": \"json\", \"with\": {\"structures\":\"inside\"}}",
			},
		},
	}

	// Success cases are expected to pass validation.
	for k, v := range successCases {
		if errs := ValidateStorageClass(&v); len(errs) != 0 {
			t.Errorf("Expected success for %d, got %v", k, errs)
		}
	}

	// generate a map longer than maxProvisionerParameterSize
	longParameters := make(map[string]string)
	totalSize := 0
	for totalSize < maxProvisionerParameterSize {
		k := fmt.Sprintf("param/%d", totalSize)
		v := fmt.Sprintf("value-%d", totalSize)
		longParameters[k] = v
		totalSize = totalSize + len(k) + len(v)
	}

	errorCases := map[string]storage.StorageClass{
		"namespace is present": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Provisioner: "kubernetes.io/foo-provisioner",
		},
		"invalid provisioner": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/invalid/provisioner",
		},
		"invalid empty parameter name": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo",
			Parameters: map[string]string{
				"": "value",
			},
		},
		"provisioner: Required value": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "",
		},
		"too long parameters": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo",
			Parameters:  longParameters,
		},
	}

	// Error cases are not expected to pass validation.
	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(&storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}
