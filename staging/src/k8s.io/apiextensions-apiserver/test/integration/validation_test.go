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
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/util/yaml"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	clientschema "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

func TestForProperValidationErrors(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	tests := []struct {
		name          string
		instanceFn    func() *unstructured.Unstructured
		expectedError string
	}{
		{
			name: "bad version",
			instanceFn: func() *unstructured.Unstructured {
				instance := fixtures.NewVersionedNoxuInstance(ns, "foo", "v2")
				return instance
			},
			expectedError: "the API version in the data (mygroup.example.com/v2) does not match the expected API version (mygroup.example.com/v1beta1)",
		},
		{
			name: "bad kind",
			instanceFn: func() *unstructured.Unstructured {
				instance := fixtures.NewNoxuInstance(ns, "foo")
				instance.Object["kind"] = "SomethingElse"
				return instance
			},
			expectedError: `SomethingElse.mygroup.example.com "foo" is invalid: kind: Invalid value: "SomethingElse": must be WishIHadChosenNoxu`,
		},
	}

	for _, tc := range tests {
		_, err := noxuResourceClient.Create(context.TODO(), tc.instanceFn(), metav1.CreateOptions{})
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

func newNoxuValidationCRDs(scope apiextensionsv1beta1.ResourceScope) []*apiextensionsv1beta1.CustomResourceDefinition {
	validationSchema := &apiextensionsv1beta1.JSONSchemaProps{
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
	}
	validationSchemaWithDescription := validationSchema.DeepCopy()
	validationSchemaWithDescription.Description = "test"
	return []*apiextensionsv1beta1.CustomResourceDefinition{
		{
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
					OpenAPIV3Schema: validationSchema,
				},
				Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
					{
						Name:    "v1beta1",
						Served:  true,
						Storage: true,
					},
					{
						Name:    "v1",
						Served:  true,
						Storage: false,
					},
				},
			},
		},
		{
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
				Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
					{
						Name:    "v1beta1",
						Served:  true,
						Storage: true,
						Schema: &apiextensionsv1beta1.CustomResourceValidation{
							OpenAPIV3Schema: validationSchema,
						},
					},
					{
						Name:    "v1",
						Served:  true,
						Storage: false,
						Schema: &apiextensionsv1beta1.CustomResourceValidation{
							OpenAPIV3Schema: validationSchemaWithDescription,
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
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			instanceToCreate := newNoxuValidationInstance(ns, "foo")
			instanceToCreate.Object["apiVersion"] = fmt.Sprintf("%s/%s", noxuDefinition.Spec.Group, v.Name)
			_, err = instantiateVersionedCustomResource(t, instanceToCreate, noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}
			noxuResourceClient.Delete(context.TODO(), "foo", metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCustomResourceItemsValidation(t *testing.T) {
	tearDown, apiExtensionClient, client, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	// decode CRD manifest
	obj, _, err := clientschema.Codecs.UniversalDeserializer().Decode([]byte(fixtureItemsAndType), nil, nil)
	if err != nil {
		t.Fatalf("failed decoding of: %v\n\n%s", err, fixtureItemsAndType)
	}
	crd := obj.(*apiextensionsv1.CustomResourceDefinition)

	// create CRDs
	t.Logf("Creating CRD %s", crd.Name)
	if _, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, client); err != nil {
		t.Fatalf("unexpected create error: %v", err)
	}

	// create CR
	gvr := schema.GroupVersionResource{
		Group:    crd.Spec.Group,
		Version:  crd.Spec.Versions[0].Name,
		Resource: crd.Spec.Names.Plural,
	}
	u := unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": gvr.GroupVersion().String(),
		"kind":       crd.Spec.Names.Kind,
		"metadata": map[string]interface{}{
			"name": "foo",
		},
		"items-no-type": map[string]interface{}{
			"items": []interface{}{
				map[string]interface{}{},
			},
		},
		"items-items-no-type": map[string]interface{}{
			"items": []interface{}{
				[]interface{}{map[string]interface{}{}},
			},
		},
		"items-properties-items-no-type": map[string]interface{}{
			"items": []interface{}{
				map[string]interface{}{
					"items": []interface{}{
						map[string]interface{}{},
					},
				},
			},
		},
		"type-array-no-items": map[string]interface{}{
			"type": "array",
		},
		"items-and-type": map[string]interface{}{
			"items": []interface{}{map[string]interface{}{}},
			"type":  "array",
		},
		"issue-84880": map[string]interface{}{
			"volumes": []interface{}{
				map[string]interface{}{
					"downwardAPI": map[string]interface{}{
						"items": []interface{}{
							map[string]interface{}{
								"path": "annotations",
							},
						},
					},
				},
			},
		},
	}}
	_, err = client.Resource(gvr).Create(context.TODO(), &u, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

const fixtureItemsAndType = `
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: foos.tests.example.com
spec:
  group: tests.example.com
  version: v1beta1
  names:
    plural: foos
    singular: foo
    kind: Foo
    listKind: Foolist
  scope: Cluster
  versions:
  - name: v1beta1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          items-no-type:
            type: object
            properties:
              items:
                type: array
                items:
                  type: object
          items-items-no-type:
            type: object
            properties:
              items:
                type: array
                items:
                  type: array
                  items:
                    type: object
          items-properties-items-no-type:
            type: object
            properties:
              items:
                type: array
                items:
                  type: object
                  properties:
                    items:
                      type: array
                      items:
                        type: object
          type-array-no-items:
            type: object
            properties:
              type:
                type: string
          items-and-type:
            type: object
            properties:
              type:
                type: string
              items:
                type: array
                items:
                  type: object
          default-with-items-and-no-type:
            type: object
            properties:
              type:
                type: string
              items:
                type: array
                items:
                  type: object
            default: {"items": []}
          issue-84880:
            type: object
            properties:
              volumes:
                type: array
                items:
                  type: object
                  properties:
                    downwardAPI:
                      type: object
                      properties:
                        items:
                          items:
                            properties:
                              path:
                                type: string
                            required:
                            - path
                            type: object
                          type: array
`

func TestCustomResourceUpdateValidation(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			instanceToCreate := newNoxuValidationInstance(ns, "foo")
			instanceToCreate.Object["apiVersion"] = fmt.Sprintf("%s/%s", noxuDefinition.Spec.Group, v.Name)
			_, err = instantiateVersionedCustomResource(t, instanceToCreate, noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}

			gottenNoxuInstance, err := noxuResourceClient.Get(context.TODO(), "foo", metav1.GetOptions{})
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

			_, err = noxuResourceClient.Update(context.TODO(), gottenNoxuInstance, metav1.UpdateOptions{})
			if err == nil {
				t.Fatalf("unexpected non-error: alpha and beta should be present while updating %v", gottenNoxuInstance)
			}
			noxuResourceClient.Delete(context.TODO(), "foo", metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCustomResourceValidationErrors(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"

		tests := []struct {
			name           string
			instanceFn     func() *unstructured.Unstructured
			expectedErrors []string
		}{
			{
				name: "bad alpha",
				instanceFn: func() *unstructured.Unstructured {
					instance := newNoxuValidationInstance(ns, "foo")
					instance.Object["alpha"] = "foo_123!"
					return instance
				},
				expectedErrors: []string{"alpha in body should match '^[a-zA-Z0-9_]*$'"},
			},
			{
				name: "bad beta",
				instanceFn: func() *unstructured.Unstructured {
					instance := newNoxuValidationInstance(ns, "foo")
					instance.Object["beta"] = 5
					return instance
				},
				expectedErrors: []string{"beta in body should be greater than or equal to 10"},
			},
			{
				name: "bad gamma",
				instanceFn: func() *unstructured.Unstructured {
					instance := newNoxuValidationInstance(ns, "foo")
					instance.Object["gamma"] = "qux"
					return instance
				},
				expectedErrors: []string{`gamma: Unsupported value: "qux": supported values: "foo", "bar", "baz"`},
			},
			{
				name: "bad delta",
				instanceFn: func() *unstructured.Unstructured {
					instance := newNoxuValidationInstance(ns, "foo")
					instance.Object["delta"] = "foobarbaz"
					return instance
				},
				expectedErrors: []string{
					"must validate at least one schema (anyOf)",
					"delta in body should be at most 5 chars long",
				},
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
				expectedErrors: []string{"alpha: Required value", "beta: Required value"},
			},
		}

		for _, tc := range tests {
			for _, v := range noxuDefinition.Spec.Versions {
				noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
				instanceToCreate := tc.instanceFn()
				instanceToCreate.Object["apiVersion"] = fmt.Sprintf("%s/%s", noxuDefinition.Spec.Group, v.Name)
				_, err := noxuResourceClient.Create(context.TODO(), instanceToCreate, metav1.CreateOptions{})
				if err == nil {
					t.Errorf("%v: expected %v", tc.name, tc.expectedErrors)
					continue
				}
				// this only works when status errors contain the expect kind and version, so this effectively tests serializations too
				for _, expectedError := range tc.expectedErrors {
					if !strings.Contains(err.Error(), expectedError) {
						t.Errorf("%v: expected %v, got %v", tc.name, expectedError, err)
					}
				}
			}
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCRValidationOnCRDUpdate(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)
	for i, noxuDefinition := range noxuDefinitions {
		for _, v := range noxuDefinition.Spec.Versions {
			// Re-define the CRD to make sure we start with a clean CRD
			noxuDefinition := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)[i]
			validationSchema, err := getSchemaForVersion(noxuDefinition, v.Name)
			if err != nil {
				t.Fatal(err)
			}

			// set stricter schema
			validationSchema.OpenAPIV3Schema.Required = []string{"alpha", "beta", "epsilon"}

			noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err != nil {
				t.Fatal(err)
			}
			ns := "not-the-default"
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			instanceToCreate := newNoxuValidationInstance(ns, "foo")
			instanceToCreate.Object["apiVersion"] = fmt.Sprintf("%s/%s", noxuDefinition.Spec.Group, v.Name)

			// CR is rejected
			_, err = instantiateVersionedCustomResource(t, instanceToCreate, noxuResourceClient, noxuDefinition, v.Name)
			if err == nil {
				t.Fatalf("unexpected non-error: CR should be rejected")
			}

			// update the CRD to a less stricter schema
			_, err = UpdateCustomResourceDefinitionWithRetry(apiExtensionClient, "noxus.mygroup.example.com", func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
				validationSchema, err := getSchemaForVersion(crd, v.Name)
				if err != nil {
					t.Fatal(err)
				}
				validationSchema.OpenAPIV3Schema.Required = []string{"alpha", "beta"}
			})
			if err != nil {
				t.Fatal(err)
			}

			// CR is now accepted
			err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
				_, err := noxuResourceClient.Create(context.TODO(), instanceToCreate, metav1.CreateOptions{})
				if _, isStatus := err.(*apierrors.StatusError); isStatus {
					if apierrors.IsInvalid(err) {
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
			noxuResourceClient.Delete(context.TODO(), "foo", metav1.DeleteOptions{})
			if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestForbiddenFieldsInSchema(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)
	for i, noxuDefinition := range noxuDefinitions {
		for _, v := range noxuDefinition.Spec.Versions {
			// Re-define the CRD to make sure we start with a clean CRD
			noxuDefinition := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)[i]
			validationSchema, err := getSchemaForVersion(noxuDefinition, v.Name)
			if err != nil {
				t.Fatal(err)
			}
			validationSchema.OpenAPIV3Schema.AdditionalProperties.Allows = false

			_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err == nil {
				t.Fatalf("unexpected non-error: additionalProperties cannot be set to false")
			}

			validationSchema.OpenAPIV3Schema.Properties["zeta"] = apiextensionsv1beta1.JSONSchemaProps{
				Type:        "array",
				UniqueItems: true,
			}
			validationSchema.OpenAPIV3Schema.AdditionalProperties.Allows = true

			_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err == nil {
				t.Fatalf("unexpected non-error: uniqueItems cannot be set to true")
			}

			validationSchema.OpenAPIV3Schema.Ref = strPtr("#/definition/zeta")
			validationSchema.OpenAPIV3Schema.Properties["zeta"] = apiextensionsv1beta1.JSONSchemaProps{
				Type:        "array",
				UniqueItems: false,
			}

			_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err == nil {
				t.Fatal("unexpected non-error: $ref cannot be non-empty string")
			}

			validationSchema.OpenAPIV3Schema.Ref = nil

			noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err != nil {
				t.Fatal(err)
			}
			if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestNonStructuralSchemaConditionUpdate(t *testing.T) {
	tearDown, apiExtensionClient, _, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	manifest := `
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: foos.tests.example.com
spec:
  group: tests.example.com
  version: v1beta1
  names:
    plural: foos
    singular: foo
    kind: Foo
    listKind: Foolist
  scope: Namespaced
  validation:
    openAPIV3Schema:
      type: object
      properties:
        a: {}
  versions:
  - name: v1beta1
    served: true
    storage: true
`

	// decode CRD manifest
	obj, _, err := clientschema.Codecs.UniversalDeserializer().Decode([]byte(manifest), nil, nil)
	if err != nil {
		t.Fatalf("failed decoding of: %v\n\n%s", err, manifest)
	}
	crd := obj.(*apiextensionsv1beta1.CustomResourceDefinition)
	name := crd.Name

	// save schema for later
	origSchema := crd.Spec.Validation.OpenAPIV3Schema

	// create CRDs
	t.Logf("Creating CRD %s", crd.Name)
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(context.TODO(), crd, metav1.CreateOptions{}); err != nil {
		t.Fatalf("unexpected create error: %v", err)
	}

	// wait for condition with violations
	t.Log("Waiting for NonStructuralSchema condition")
	var cond *apiextensionsv1beta1.CustomResourceDefinitionCondition
	err = wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		obj, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		cond = findCRDCondition(obj, apiextensionsv1beta1.NonStructuralSchema)
		return cond != nil, nil
	})
	if err != nil {
		t.Fatalf("unexpected error waiting for NonStructuralSchema condition: %v", cond)
	}
	if v := "spec.versions[0].schema.openAPIV3Schema.properties[a].type: Required value: must not be empty for specified object fields"; !strings.Contains(cond.Message, v) {
		t.Fatalf("expected violation %q, but got: %v", v, cond.Message)
	}
	if v := "spec.preserveUnknownFields: Invalid value: true: must be false"; !strings.Contains(cond.Message, v) {
		t.Fatalf("expected violation %q, but got: %v", v, cond.Message)
	}

	// remove schema
	t.Log("Remove schema")
	for retry := 0; retry < 5; retry++ {
		// This patch fixes two fields to resolve
		// 1. property type validation error
		// 2. preserveUnknownFields validation error
		patch := []byte("[{\"op\":\"add\",\"path\":\"/spec/validation/openAPIV3Schema/properties/a/type\",\"value\":\"int\"}," +
			"{\"op\":\"replace\",\"path\":\"/spec/preserveUnknownFields\",\"value\":false}]")
		if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Patch(context.TODO(), name, types.JSONPatchType, patch, metav1.PatchOptions{}); apierrors.IsConflict(err) {
			continue
		}
		if err != nil {
			t.Fatalf("unexpected update error: %v", err)
		}
		break
	}
	if err != nil {
		t.Fatalf("unexpected update error: %v", err)
	}

	// wait for condition to go away
	t.Log("Wait for condition to disappear")
	err = wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		obj, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		cond = findCRDCondition(obj, apiextensionsv1beta1.NonStructuralSchema)
		return cond == nil, nil
	})
	if err != nil {
		t.Fatalf("unexpected error waiting for NonStructuralSchema condition: %v", cond)
	}

	// re-add schema
	t.Log("Re-add schema")
	for retry := 0; retry < 5; retry++ {
		crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("unexpected get error: %v", err)
		}
		crd.Spec.PreserveUnknownFields = nil
		crd.Spec.Validation = &apiextensionsv1beta1.CustomResourceValidation{OpenAPIV3Schema: origSchema}
		if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(context.TODO(), crd, metav1.UpdateOptions{}); apierrors.IsConflict(err) {
			continue
		}
		if err != nil {
			t.Fatalf("unexpected update error: %v", err)
		}
		break
	}
	if err != nil {
		t.Fatalf("unexpected update error: %v", err)
	}

	// wait for condition with violations
	t.Log("Wait for condition to reappear")
	err = wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		obj, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		cond = findCRDCondition(obj, apiextensionsv1beta1.NonStructuralSchema)
		return cond != nil, nil
	})
	if err != nil {
		t.Fatalf("unexpected error waiting for NonStructuralSchema condition: %v", cond)
	}
	if v := "spec.versions[0].schema.openAPIV3Schema.properties[a].type: Required value: must not be empty for specified object fields"; !strings.Contains(cond.Message, v) {
		t.Fatalf("expected violation %q, but got: %v", v, cond.Message)
	}
	if v := "spec.preserveUnknownFields: Invalid value: true: must be false"; !strings.Contains(cond.Message, v) {
		t.Fatalf("expected violation %q, but got: %v", v, cond.Message)
	}
}

func TestNonStructuralSchemaCondition(t *testing.T) {
	tearDown, apiExtensionClient, _, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	tmpl := `
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
spec:
  preserveUnknownFields: PRESERVE_UNKNOWN_FIELDS
  version: v1beta1
  names:
    plural: foos
    singular: foo
    kind: Foo
    listKind: Foolist
  scope: Namespaced
  validation: GLOBAL_SCHEMA
  versions:
  - name: v1beta1
    served: true
    storage: true
    schema: V1BETA1_SCHEMA
  - name: v1
    served: true
    schema: V1_SCHEMA
`

	type Test struct {
		desc                                  string
		preserveUnknownFields                 string
		globalSchema, v1Schema, v1beta1Schema string
		expectedCreateErrors                  []string
		unexpectedCreateErrors                []string
		expectedViolations                    []string
		unexpectedViolations                  []string
	}
	tests := []Test{
		{
			desc: "empty",
			expectedViolations: []string{
				"spec.preserveUnknownFields: Invalid value: true: must be false",
			},
		},
		{
			desc:                  "preserve unknown fields is false",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
`,
		},
		{
			desc: "int-or-string and preserve-unknown-fields true",
			globalSchema: `
x-kubernetes-preserve-unknown-fields: true
x-kubernetes-int-or-string: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.x-kubernetes-preserve-unknown-fields: Invalid value: true: must be false if x-kubernetes-int-or-string is true",
			},
		},
		{
			desc: "int-or-string and embedded-resource true",
			globalSchema: `
type: object
x-kubernetes-embedded-resource: true
x-kubernetes-int-or-string: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.x-kubernetes-embedded-resource: Invalid value: true: must be false if x-kubernetes-int-or-string is true",
			},
		},
		{
			desc: "embedded-resource without preserve-unknown-fields",
			globalSchema: `
type: object
x-kubernetes-embedded-resource: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.properties: Required value: must not be empty if x-kubernetes-embedded-resource is true without x-kubernetes-preserve-unknown-fields",
			},
		},
		{
			desc:                  "embedded-resource without preserve-unknown-fields, but properties",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
x-kubernetes-embedded-resource: true
properties:
 apiVersion:
   type: string
 kind:
   type: string
 metadata:
   type: object
`,
		},
		{
			desc:                  "embedded-resource with preserve-unknown-fields",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
x-kubernetes-embedded-resource: true
x-kubernetes-preserve-unknown-fields: true
`,
		},
		{
			desc: "embedded-resource with wrong type",
			globalSchema: `
type: array
x-kubernetes-embedded-resource: true
x-kubernetes-preserve-unknown-fields: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.type: Invalid value: \"array\": must be object if x-kubernetes-embedded-resource is true",
			},
		},
		{
			desc: "embedded-resource with empty type",
			globalSchema: `
type: ""
x-kubernetes-embedded-resource: true
x-kubernetes-preserve-unknown-fields: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.type: Required value: must be object if x-kubernetes-embedded-resource is true",
			},
		},
		{
			desc: "no top-level type",
			globalSchema: `
type: ""
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.type: Required value: must not be empty at the root",
			},
		},
		{
			desc: "non-object top-level type",
			globalSchema: `
type: "integer"
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.type: Invalid value: \"integer\": must be object at the root",
			},
		},
		{
			desc: "forbidden in nested value validation",
			globalSchema: `
type: object
properties:
 foo:
   type: string
not:
 type: string
 additionalProperties: true
 title: hello
 description: world
 nullable: true
allOf:
- properties:
   foo:
     type: string
     additionalProperties: true
     title: hello
     description: world
     nullable: true
anyOf:
- items:
   type: string
   additionalProperties: true
   title: hello
   description: world
   nullable: true
oneOf:
- properties:
   foo:
     type: string
     additionalProperties: true
     title: hello
     description: world
     nullable: true
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].items.type: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].items.additionalProperties: Forbidden: must be undefined to be structural",
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].items.title: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].items.description: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].items.nullable: Forbidden: must be false to be structural",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[foo].type: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[foo].additionalProperties: Forbidden: must be undefined to be structural",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[foo].title: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[foo].description: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[foo].nullable: Forbidden: must be false to be structural",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[foo].type: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[foo].additionalProperties: Forbidden: must be undefined to be structural",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[foo].title: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[foo].description: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[foo].nullable: Forbidden: must be false to be structural",
				"spec.versions[0].schema.openAPIV3Schema.not.type: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.not.additionalProperties: Forbidden: must be undefined to be structural",
				"spec.versions[0].schema.openAPIV3Schema.not.title: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.not.description: Forbidden: must be empty to be structural",
				"spec.versions[0].schema.openAPIV3Schema.not.nullable: Forbidden: must be false to be structural",
				"spec.versions[0].schema.openAPIV3Schema.items: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.anyOf[0].items",
			},
			unexpectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.not.default",
			},
		},
		{
			desc: "invalid regex pattern",
			globalSchema: `
type: object
properties:
 foo:
   type: string
   pattern: "+"
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[foo].pattern: Invalid value: \"+\": must be a valid regular expression, but isn't: error parsing regexp: missing argument to repetition operator: `+`",
			},
		},
		{
			desc: "forbidden vendor extensions in nested value validation",
			globalSchema: `
type: object
properties:
 int-or-string:
   x-kubernetes-int-or-string: true
 embedded-resource:
   type: object
   x-kubernetes-embedded-resource: true
   x-kubernetes-preserve-unknown-fields: true
not:
 properties:
   int-or-string:
     x-kubernetes-int-or-string: true
   embedded-resource:
     x-kubernetes-embedded-resource: true
     x-kubernetes-preserve-unknown-fields: true
allOf:
- properties:
   int-or-string:
     x-kubernetes-int-or-string: true
   embedded-resource:
     x-kubernetes-embedded-resource: true
     x-kubernetes-preserve-unknown-fields: true
anyOf:
- properties:
   int-or-string:
     x-kubernetes-int-or-string: true
   embedded-resource:
     x-kubernetes-embedded-resource: true
     x-kubernetes-preserve-unknown-fields: true
oneOf:
- properties:
   int-or-string:
     x-kubernetes-int-or-string: true
   embedded-resource:
     x-kubernetes-embedded-resource: true
     x-kubernetes-preserve-unknown-fields: true
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.allOf[0].properties[embedded-resource].x-kubernetes-preserve-unknown-fields: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.allOf[0].properties[embedded-resource].x-kubernetes-embedded-resource: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.allOf[0].properties[int-or-string].x-kubernetes-int-or-string: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.anyOf[0].properties[embedded-resource].x-kubernetes-preserve-unknown-fields: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.anyOf[0].properties[embedded-resource].x-kubernetes-embedded-resource: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.anyOf[0].properties[int-or-string].x-kubernetes-int-or-string: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.oneOf[0].properties[embedded-resource].x-kubernetes-preserve-unknown-fields: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.oneOf[0].properties[embedded-resource].x-kubernetes-embedded-resource: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.oneOf[0].properties[int-or-string].x-kubernetes-int-or-string: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.not.properties[embedded-resource].x-kubernetes-preserve-unknown-fields: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.not.properties[embedded-resource].x-kubernetes-embedded-resource: Forbidden: must be false to be structural",
				"spec.validation.openAPIV3Schema.not.properties[int-or-string].x-kubernetes-int-or-string: Forbidden: must be false to be structural",
			},
		},
		{
			desc: "missing types with extensions",
			globalSchema: `
properties:
 foo:
   properties:
     a: {}
 bar:
   items:
     additionalProperties:
       properties:
         a: {}
       items: {}
 abc:
   additionalProperties:
     properties:
       a:
         items:
           additionalProperties:
             items:
 json:
   x-kubernetes-preserve-unknown-fields: true
   properties:
     a: {}
 int-or-string:
   x-kubernetes-int-or-string: true
   properties:
     a: {}
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.properties[foo].properties[a].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[foo].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[int-or-string].properties[a].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[json].properties[a].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[abc].additionalProperties.properties[a].items.additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[abc].additionalProperties.properties[a].items.type: Required value: must not be empty for specified array items",
				"spec.validation.openAPIV3Schema.properties[abc].additionalProperties.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[abc].additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[abc].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[bar].items.additionalProperties.items.type: Required value: must not be empty for specified array items",
				"spec.validation.openAPIV3Schema.properties[bar].items.additionalProperties.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[bar].items.additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.properties[bar].items.type: Required value: must not be empty for specified array items",
				"spec.validation.openAPIV3Schema.properties[bar].type: Required value: must not be empty for specified object fields",
				"spec.validation.openAPIV3Schema.type: Required value: must not be empty at the root",
			},
		},
		{
			desc: "missing types without extensions",
			globalSchema: `
properties:
 foo:
   properties:
     a: {}
 bar:
   items:
     additionalProperties:
       properties:
         a: {}
       items: {}
 abc:
   additionalProperties:
     properties:
       a:
         items:
           additionalProperties:
             items:
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[foo].properties[a].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[foo].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[abc].additionalProperties.properties[a].items.additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[abc].additionalProperties.properties[a].items.type: Required value: must not be empty for specified array items",
				"spec.versions[0].schema.openAPIV3Schema.properties[abc].additionalProperties.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[abc].additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[abc].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[bar].items.additionalProperties.items.type: Required value: must not be empty for specified array items",
				"spec.versions[0].schema.openAPIV3Schema.properties[bar].items.additionalProperties.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[bar].items.additionalProperties.type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[bar].items.type: Required value: must not be empty for specified array items",
				"spec.versions[0].schema.openAPIV3Schema.properties[bar].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.type: Required value: must not be empty at the root",
			},
		},
		{
			desc: "int-or-string variants",
			globalSchema: `
type: object
properties:
 a:
   x-kubernetes-int-or-string: true
 b:
   x-kubernetes-int-or-string: true
   anyOf:
   - type: integer
   - type: string
   allOf:
   - pattern: abc
 c:
   x-kubernetes-int-or-string: true
   allOf:
   - anyOf:
     - type: integer
     - type: string
   - pattern: abc
   - pattern: abc
 d:
   x-kubernetes-int-or-string: true
   anyOf:
   - type: integer
   - type: string
     pattern: abc
 e:
   x-kubernetes-int-or-string: true
   allOf:
   - anyOf:
     - type: integer
     - type: string
       pattern: abc
   - pattern: abc
 f:
   x-kubernetes-int-or-string: true
   anyOf:
   - type: integer
   - type: string
   - pattern: abc
 g:
   x-kubernetes-int-or-string: true
   anyOf:
   - type: string
   - type: integer
`,
			expectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.properties[d].anyOf[0].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[d].anyOf[1].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[e].allOf[0].anyOf[0].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[e].allOf[0].anyOf[1].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[f].anyOf[0].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[f].anyOf[1].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[g].anyOf[0].type: Forbidden: must be empty to be structural",
				"spec.validation.openAPIV3Schema.properties[g].anyOf[1].type: Forbidden: must be empty to be structural",
			},
			unexpectedCreateErrors: []string{
				"spec.validation.openAPIV3Schema.properties[a]",
				"spec.validation.openAPIV3Schema.properties[b]",
				"spec.validation.openAPIV3Schema.properties[c]",
			},
		},
		{
			desc: "forbidden additionalProperties at the root",
			globalSchema: `
type: object
additionalProperties: false
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.additionalProperties: Forbidden: must not be used at the root",
			},
		},
		{
			desc: "structural incomplete",
			globalSchema: `
type: object
properties:
 b:
   type: object
   properties:
     b:
       type: array
 c:
   type: array
   items:
     type: object
 d:
   type: array
not:
 properties:
   a: {}
   b:
     not:
       properties:
         a: {}
         b:
           items: {}
   c:
     items:
       not:
         items:
           properties:
             a: {}
   d:
     items: {}
allOf:
- properties:
   e: {}
anyOf:
- properties:
   f: {}
oneOf:
- properties:
   g: {}
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[d].items: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[d].items",
				"spec.versions[0].schema.openAPIV3Schema.properties[a]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[a]",
				"spec.versions[0].schema.openAPIV3Schema.properties[b].properties[a]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[b].not.properties[a]",
				"spec.versions[0].schema.openAPIV3Schema.properties[b].properties[b].items: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[b].not.properties[b].items",
				"spec.versions[0].schema.openAPIV3Schema.properties[c].items.items: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[c].items.not.items",
				"spec.versions[0].schema.openAPIV3Schema.properties[e]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[e]",
				"spec.versions[0].schema.openAPIV3Schema.properties[f]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.anyOf[0].properties[f]",
				"spec.versions[0].schema.openAPIV3Schema.properties[g]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[g]",
			},
		},
		{
			desc:                  "structural complete",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
properties:
 a:
   type: string
 b:
   type: object
   properties:
     a:
       type: string
     b:
       type: array
       items:
         type: string
 c:
   type: array
   items:
     type: array
     items:
       type: object
       properties:
         a:
           type: string
 d:
   type: array
   items:
     type: string
 e:
   type: string
 f:
   type: string
 g:
   type: string
not:
 properties:
   a: {}
   b:
     not:
       properties:
         a: {}
         b:
           items: {}
   c:
     items:
       not:
         items:
           properties:
             a: {}
   d:
     items: {}
allOf:
- properties:
   e: {}
anyOf:
- properties:
   f: {}
oneOf:
- properties:
   g: {}
`,
		},
		{
			desc: "invalid v1beta1 schema",
			v1beta1Schema: `
type: object
properties:
 a: {}
not:
 properties:
   b: {}
`,
			v1Schema: `
type: object
properties:
 a:
   type: string
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[b]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[b]",
			},
		},
		{
			desc: "invalid v1beta1 and v1 schemas",
			v1beta1Schema: `
type: object
properties:
 a: {}
not:
 properties:
   b: {}
`,
			v1Schema: `
type: object
properties:
 c: {}
not:
 properties:
   d: {}
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[a].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[b]: Required value: because it is defined in spec.versions[0].schema.openAPIV3Schema.not.properties[b]",
				"spec.versions[1].schema.openAPIV3Schema.properties[c].type: Required value: must not be empty for specified object fields",
				"spec.versions[1].schema.openAPIV3Schema.properties[d]: Required value: because it is defined in spec.versions[1].schema.openAPIV3Schema.not.properties[d]",
			},
		},
		{
			desc: "metadata with non-properties",
			globalSchema: `
type: object
properties:
 metadata:
   minimum: 42.0
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[metadata]: Forbidden: must not specify anything other than name and generateName, but metadata is implicitly specified",
				"spec.versions[0].schema.openAPIV3Schema.properties[metadata].type: Required value: must not be empty for specified object fields",
			},
		},
		{
			desc: "metadata with other properties",
			globalSchema: `
type: object
properties:
 metadata:
   properties:
     name:
       pattern: "^[a-z]+$"
     labels:
       type: object
       maxLength: 4
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[metadata]: Forbidden: must not specify anything other than name and generateName, but metadata is implicitly specified",
				"spec.versions[0].schema.openAPIV3Schema.properties[metadata].type: Required value: must not be empty for specified object fields",
				"spec.versions[0].schema.openAPIV3Schema.properties[metadata].properties[name].type: Required value: must not be empty for specified object fields",
			},
		},
		{
			desc:                  "metadata with name property",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
properties:
 metadata:
   type: object
   properties:
     name:
       type: string
       pattern: "^[a-z]+$"
`,
		},
		{
			desc:                  "metadata with generateName property",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
properties:
 metadata:
   type: object
   properties:
     generateName:
       type: string
       pattern: "^[a-z]+$"
`,
		},
		{
			desc:                  "metadata with name and generateName property",
			preserveUnknownFields: "false",
			globalSchema: `
type: object
properties:
 metadata:
   type: object
   properties:
     name:
       type: string
       pattern: "^[a-z]+$"
     generateName:
       type: string
       pattern: "^[a-z]+$"
`,
		},
		{
			desc: "metadata under junctors",
			globalSchema: `
type: object
properties:
 metadata:
   type: object
   properties:
     name:
       type: string
       pattern: "^[a-z]+$"
allOf:
- properties:
   metadata: {}
anyOf:
- properties:
   metadata: {}
oneOf:
- properties:
   metadata: {}
not:
 properties:
   metadata: {}
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.anyOf[0].properties[metadata]: Forbidden: must not be specified in a nested context",
				"spec.versions[0].schema.openAPIV3Schema.allOf[0].properties[metadata]: Forbidden: must not be specified in a nested context",
				"spec.versions[0].schema.openAPIV3Schema.oneOf[0].properties[metadata]: Forbidden: must not be specified in a nested context",
				"spec.versions[0].schema.openAPIV3Schema.not.properties[metadata]: Forbidden: must not be specified in a nested context",
			},
		},
		{
			desc: "missing items for array",
			globalSchema: `
type: object
properties:
 slice:
   type: array
`,
			expectedViolations: []string{
				"spec.versions[0].schema.openAPIV3Schema.properties[slice].items: Required value: must be specified",
			},
		},
		{
			desc: "items slice",
			globalSchema: `
type: object
properties:
 slice:
   type: array
   items:
   - type: string
   - type: integer
`,
			expectedCreateErrors: []string{"spec.validation.openAPIV3Schema.properties[slice].items: Forbidden: items must be a schema object and not an array"},
		},
		{
			desc: "items slice in value validation",
			globalSchema: `
type: object
properties:
 slice:
   type: array
   items:
     type: string
   not:
     items:
     - type: string
`,
			expectedCreateErrors: []string{"spec.validation.openAPIV3Schema.properties[slice].not.items: Forbidden: items must be a schema object and not an array"},
		},
	}

	for i := range tests {
		tst := tests[i]
		t.Run(tst.desc, func(t *testing.T) {
			// plug in schemas
			manifest := strings.NewReplacer(
				"GLOBAL_SCHEMA", toValidationJSON(tst.globalSchema),
				"V1BETA1_SCHEMA", toValidationJSON(tst.v1beta1Schema),
				"V1_SCHEMA", toValidationJSON(tst.v1Schema),
				"PRESERVE_UNKNOWN_FIELDS", tst.preserveUnknownFields,
			).Replace(tmpl)

			// decode CRD manifest
			obj, _, err := clientschema.Codecs.UniversalDeserializer().Decode([]byte(manifest), nil, nil)
			if err != nil {
				t.Fatalf("failed decoding of: %v\n\n%s", err, manifest)
			}
			crd := obj.(*apiextensionsv1beta1.CustomResourceDefinition)
			crd.Spec.Group = fmt.Sprintf("tests-%d.apiextension.k8s.io", i)
			crd.Name = fmt.Sprintf("foos.%s", crd.Spec.Group)

			// create CRDs
			crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(context.TODO(), crd, metav1.CreateOptions{})
			if len(tst.expectedCreateErrors) > 0 && err == nil {
				t.Fatalf("expected create errors, got none")
			} else if len(tst.expectedCreateErrors) == 0 && err != nil {
				t.Fatalf("unexpected create error: %v", err)
			} else if err != nil {
				for _, expectedErr := range tst.expectedCreateErrors {
					if !strings.Contains(err.Error(), expectedErr) {
						t.Errorf("expected error containing '%s', got '%s'", expectedErr, err.Error())
					}
				}
				for _, unexpectedErr := range tst.unexpectedCreateErrors {
					if strings.Contains(err.Error(), unexpectedErr) {
						t.Errorf("unexpected error containing '%s': '%s'", unexpectedErr, err.Error())
					}
				}
			}
			if err != nil {
				return
			}

			if len(tst.expectedViolations) == 0 {
				// wait for condition to not appear
				var cond *apiextensionsv1beta1.CustomResourceDefinitionCondition
				err := wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
					obj, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
					if err != nil {
						return false, err
					}
					cond = findCRDCondition(obj, apiextensionsv1beta1.NonStructuralSchema)
					if cond == nil {
						return false, nil
					}
					return true, nil
				})
				if err != wait.ErrWaitTimeout {
					t.Fatalf("expected no NonStructuralSchema condition, but got one: %v", cond)
				}
				return
			}

			// wait for condition to appear with the given violations
			var cond *apiextensionsv1beta1.CustomResourceDefinitionCondition
			err = wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
				obj, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				cond = findCRDCondition(obj, apiextensionsv1beta1.NonStructuralSchema)
				if cond != nil {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Fatalf("unexpected error waiting for violations in NonStructuralSchema condition: %v", err)
			}

			// check that the condition looks good
			if cond.Reason != "Violations" {
				t.Errorf("expected reason Violations, got: %v", cond.Reason)
			}
			if cond.Status != apiextensionsv1beta1.ConditionTrue {
				t.Errorf("expected reason True, got: %v", cond.Status)
			}

			// check that we got all violations
			t.Logf("Got violations: %q", cond.Message)
			for _, v := range tst.expectedViolations {
				if strings.Index(cond.Message, v) == -1 {
					t.Errorf("expected violation %q, but didn't get it", v)
				}
			}
			for _, v := range tst.unexpectedViolations {
				if strings.Index(cond.Message, v) != -1 {
					t.Errorf("unexpected violation %q", v)
				}
			}
		})
	}
}

// findCRDCondition returns the condition you're looking for or nil.
func findCRDCondition(crd *apiextensionsv1beta1.CustomResourceDefinition, conditionType apiextensionsv1beta1.CustomResourceDefinitionConditionType) *apiextensionsv1beta1.CustomResourceDefinitionCondition {
	for i := range crd.Status.Conditions {
		if crd.Status.Conditions[i].Type == conditionType {
			return &crd.Status.Conditions[i]
		}
	}

	return nil
}

func toValidationJSON(yml string) string {
	if len(yml) == 0 {
		return "null"
	}
	bs, err := yaml.ToJSON([]byte(yml))
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("{\"openAPIV3Schema\": %s}", string(bs))
}

func float64Ptr(f float64) *float64 {
	return &f
}

func int64Ptr(f int64) *int64 {
	return &f
}

func strPtr(str string) *string {
	return &str
}
