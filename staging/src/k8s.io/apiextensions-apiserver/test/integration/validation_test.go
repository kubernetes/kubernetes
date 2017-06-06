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

package integration

import (
	"strings"
	"testing"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestForProperValidationErrors(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)

	tests := []struct {
		name          string
		instanceFn    func() *unstructured.Unstructured
		expectedError string
	}{
		{
			name: "bad version",
			instanceFn: func() *unstructured.Unstructured {
				instance := testserver.NewNoxuInstance(ns, "foo")
				instance.Object["apiVersion"] = "mygroup.example.com/v2"
				return instance
			},
			expectedError: "the API version in the data (mygroup.example.com/v2) does not match the expected API version (mygroup.example.com/v1beta1)",
		},
		{
			name: "bad kind",
			instanceFn: func() *unstructured.Unstructured {
				instance := testserver.NewNoxuInstance(ns, "foo")
				instance.Object["kind"] = "SomethingElse"
				return instance
			},
			expectedError: `SomethingElse.mygroup.example.com "foo" is invalid: kind: Invalid value: "SomethingElse": must be WishIHadChosenNoxu`,
		},
	}

	for _, tc := range tests {
		_, err := noxuResourceClient.Create(tc.instanceFn())
		if err == nil {
			t.Errorf("%v: expected %v", tc.name, tc.expectedError)
			continue
		}
		// this only works when status errors contain the expect kind and version, so this effectively tests serializations too
		if !strings.Contains(err.Error(), tc.expectedError) {
			t.Errorf("%v: expected %v, got %v", tc.name, tc.expectedError, err)
			continue
		}
	}
}
