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
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
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

func newNoxuValidationCRD(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
			},
			Scope: apiextensionsv1beta1.NamespaceScoped,
			Validation: &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
					Required: []string{"alpha", "beta"},
					AdditionalProperties: &apiextensionsv1beta1.JSONSchemaPropsOrBool{
						Allows: true,
					},
					Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
						"alpha": {
							Description: "Alpha is an alphanumeric string with underscores",
							Type:        "string",
							Pattern:     "^[a-zA-Z0-9_]*$",
						},
						"beta": {
							Description: "Minimum value of beta is 10",
							Type:        "number",
							Minimum:     float64Ptr(10),
						},
						"gamma": {
							Description: "Gamma is restricted to foo, bar and baz",
							Type:        "string",
							Enum: []apiextensionsv1beta1.JSON{
								{
									Raw: []byte(`"foo"`),
								},
								{
									Raw: []byte(`"bar"`),
								},
								{
									Raw: []byte(`"baz"`),
								},
							},
						},
						"delta": {
							Description: "Delta is a string with a maximum length of 5 or a number with a minimum value of 0",
							AnyOf: []apiextensionsv1beta1.JSONSchemaProps{
								{
									Type:      "string",
									MaxLength: int64Ptr(5),
								},
								{
									Type:    "number",
									Minimum: float64Ptr(0),
								},
							},
						},
					},
				},
			},
		},
	}
}

func newNoxuValidationInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "WishIHadChosenNoxu",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"alpha": "foo_123",
			"beta":  10,
			"gamma": "bar",
			"delta": "hello",
		},
	}
}

func TestCustomResourceValidation(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, newNoxuValidationInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}
}

func TestCustomResourceUpdateValidation(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, newNoxuValidationInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}

	gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// invalidate the instance
	gottenNoxuInstance.Object = map[string]interface{}{
		"apiVersion": "mygroup.example.com/v1beta1",
		"kind":       "WishIHadChosenNoxu",
		"metadata": map[string]interface{}{
			"namespace": "not-the-default",
			"name":      "foo",
		},
		"gamma": "bar",
		"delta": "hello",
	}

	_, err = noxuResourceClient.Update(gottenNoxuInstance)
	if err == nil {
		t.Fatalf("unexpected non-error: alpha and beta should be present while updating %v", gottenNoxuInstance)
	}
}

func TestCustomResourceValidationErrors(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
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
			name: "bad alpha",
			instanceFn: func() *unstructured.Unstructured {
				instance := newNoxuValidationInstance(ns, "foo")
				instance.Object["alpha"] = "foo_123!"
				return instance
			},
			expectedError: "alpha in body should match '^[a-zA-Z0-9_]*$'",
		},
		{
			name: "bad beta",
			instanceFn: func() *unstructured.Unstructured {
				instance := newNoxuValidationInstance(ns, "foo")
				instance.Object["beta"] = 5
				return instance
			},
			expectedError: "beta in body should be greater than or equal to 10",
		},
		{
			name: "bad gamma",
			instanceFn: func() *unstructured.Unstructured {
				instance := newNoxuValidationInstance(ns, "foo")
				instance.Object["gamma"] = "qux"
				return instance
			},
			expectedError: "gamma in body should be one of [foo bar baz]",
		},
		{
			name: "bad delta",
			instanceFn: func() *unstructured.Unstructured {
				instance := newNoxuValidationInstance(ns, "foo")
				instance.Object["delta"] = "foobarbaz"
				return instance
			},
			expectedError: "must validate at least one schema (anyOf)\ndelta in body should be at most 5 chars long",
		},
		{
			name: "absent alpha and beta",
			instanceFn: func() *unstructured.Unstructured {
				instance := newNoxuValidationInstance(ns, "foo")
				instance.Object = map[string]interface{}{
					"apiVersion": "mygroup.example.com/v1beta1",
					"kind":       "WishIHadChosenNoxu",
					"metadata": map[string]interface{}{
						"namespace": "not-the-default",
						"name":      "foo",
					},
					"gamma": "bar",
					"delta": "hello",
				}
				return instance
			},
			expectedError: ".alpha in body is required\n.beta in body is required",
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

func TestCRValidationOnCRDUpdate(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)

	// set stricter schema
	noxuDefinition.Spec.Validation.OpenAPIV3Schema.Required = []string{"alpha", "beta", "epsilon"}

	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)

	// CR is rejected
	_, err = instantiateCustomResource(t, newNoxuValidationInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err == nil {
		t.Fatalf("unexpected non-error: CR should be rejected")
	}

	gottenCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// update the CRD to a less stricter schema
	gottenCRD.Spec.Validation.OpenAPIV3Schema.Required = []string{"alpha", "beta"}
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD); err != nil {
		t.Fatal(err)
	}

	// CR is now accepted
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := noxuResourceClient.Create(newNoxuValidationInstance(ns, "foo"))
		if statusError, isStatus := err.(*apierrors.StatusError); isStatus {
			if strings.Contains(statusError.Error(), "is invalid") {
				return false, nil
			}
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestForbiddenFieldsInSchema(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition.Spec.Validation.OpenAPIV3Schema.AdditionalProperties.Allows = false

	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err == nil {
		t.Fatalf("unexpected non-error: additionalProperties cannot be set to false")
	}

	noxuDefinition.Spec.Validation.OpenAPIV3Schema.Properties["zeta"] = apiextensionsv1beta1.JSONSchemaProps{
		Type:        "array",
		UniqueItems: true,
	}
	noxuDefinition.Spec.Validation.OpenAPIV3Schema.AdditionalProperties.Allows = true

	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err == nil {
		t.Fatalf("unexpected non-error: uniqueItems cannot be set to true")
	}

	noxuDefinition.Spec.Validation.OpenAPIV3Schema.Properties["zeta"] = apiextensionsv1beta1.JSONSchemaProps{
		Type:        "array",
		UniqueItems: false,
	}

	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

}

func float64Ptr(f float64) *float64 {
	return &f
}

func int64Ptr(f int64) *int64 {
	return &f
}
