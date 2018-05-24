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

package integration

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/dynamic"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/features"
)

func TestCustomResourceDefaulting(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceDefaulting, true)()

	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	crd := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("42")}}
	crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'default' field to be set")
	ns := "not-the-default"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, crd)
	obj, err := instantiateCustomResource(t, newNoxuValidationInstance(ns, "foo"), noxuResourceClient, crd)
	if err != nil {
		t.Fatalf("Unable to create noxu instance: %v", err)
	}

	t.Logf("CR created: %#v", obj.UnstructuredContent())

	got, found, err := unstructured.NestedFieldCopy(obj.UnstructuredContent(), "default")
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatalf("Expected 'default' field")
	}
	if gotInt64, foundInt64, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); err != nil {
		t.Fatal(err)
	} else if !foundInt64 || gotInt64 != 42 {
		t.Fatalf("Expected 'default' field with value 42, got: %v", got)
	}

	t.Logf("Getting CR and expect 'default' field to be set")
	obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, found, err = unstructured.NestedFieldCopy(obj.UnstructuredContent(), "default")
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatalf("Expected 'default' field")
	}
	if gotInt64, foundInt64, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); err != nil {
		t.Fatal(err)
	} else if !foundInt64 || gotInt64 != 42 {
		t.Fatalf("Expected 'default' field with value 42, got: %v", got)
	}

	t.Logf("Adding 'default2' to schema")
	crd, err = updateCustomResourceDefinitionWithRetry(apiExtensionsClient, "noxus.mygroup.example.com", func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
		crd.Spec.Validation.OpenAPIV3Schema.Properties["default2"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("42")}}
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Getting old CR again to check for new 'default2 field")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("  got: %#v", obj)
		if gotInt64, foundInt64, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); err != nil {
			return false, err
		} else if foundInt64 && gotInt64 == 42 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for 'default2' to appear: %v", err)
	}

	t.Logf("Removing both defaults from schema")
	crd, err = updateCustomResourceDefinitionWithRetry(apiExtensionsClient, "noxus.mygroup.example.com", func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
		delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default")
		delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default2")
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Getting old CR again, waiting for unpersisted 'default2' field to disappear, but not 'default'")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		latest, err := noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("  got: %#v", latest)
		if gotDefault, foundDefault, err := unstructured.NestedInt64(latest.UnstructuredContent(), "default"); err != nil {
			return false, err
		} else if !foundDefault {
			return false, fmt.Errorf("property 'default' is gone, but should have been persisted")
		} else if gotDefault != 42 {
			return false, fmt.Errorf("expect 'default' to be 42, got: %v", gotDefault)
		}
		if _, foundDefault2, err := unstructured.NestedInt64(latest.UnstructuredContent(), "default2"); err != nil {
			return false, err
		} else if foundDefault2 {
			return false, nil
		}

		// default2 should be gone, default should still exist
		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for 'default2' to disappear: %v", err)
	}

	t.Logf("Re-adding defaults to schema with different value 43")
	crd, err = updateCustomResourceDefinitionWithRetry(apiExtensionsClient, "noxus.mygroup.example.com", func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
		crd.Spec.Validation.OpenAPIV3Schema.Properties["default"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{Raw: []byte("43")}}
		crd.Spec.Validation.OpenAPIV3Schema.Properties["default2"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{Raw: []byte("43")}}
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Waiting for 'default2' field to appear on update with value 43, but 'default' stay with persisted value 42")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		latest, err := noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// remove to enforce defaulting
		unstructured.RemoveNestedField(latest.UnstructuredContent(), "default2")
		latest, err = noxuResourceClient.Update(latest, metav1.UpdateOptions{})
		if err != nil {
			return false, err
		}
		t.Logf("Updated to %#v", latest.UnstructuredContent())

		if gotDefault, foundDefault, err := unstructured.NestedInt64(latest.UnstructuredContent(), "default"); err != nil {
			return false, err
		} else if !foundDefault {
			return false, fmt.Errorf("property 'default' is gone, but should have been persisted")
		} else if gotDefault != 42 {
			return false, fmt.Errorf("property 'default' has not the persistet value 42, but got: %d", gotDefault)
		}
		if gotDefault2, foundDefault2, err := unstructured.NestedInt64(latest.UnstructuredContent(), "default2"); err != nil {
			return false, err
		} else if !foundDefault2 {
			return false, nil
		} else if gotDefault2 != 43 {
			return false, fmt.Errorf("expected 'default2' field value 43, but got %d", gotDefault2)
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for 'default2' to change value: %v", err)
	}

	t.Logf("Removing 'default2' again from schema, and add 'default3' as a trigger")
	crd, err = updateCustomResourceDefinitionWithRetry(apiExtensionsClient, "noxus.mygroup.example.com", func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
		delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default")
		delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default2")
		crd.Spec.Validation.OpenAPIV3Schema.Properties["default3"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("44")}}
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Waiting for 'default3' field to appear, but 'default' and 'default2' to still exist because they were persisted by update")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if _, foundDefault, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); err != nil {
			return false, err
		} else if !foundDefault {
			return false, fmt.Errorf("property 'default' is gone, but should have been persisted")
		}
		if _, foundDefault2, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); err != nil {
			return false, err
		} else if !foundDefault2 {
			return false, fmt.Errorf("property 'default2' is gone, but should have been persisted")
		}
		if _, foundDefault3, err := unstructured.NestedInt64(obj.UnstructuredContent(), "default3"); err != nil {
			return false, err
		} else if !foundDefault3 {
			return false, nil
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("Failed waiting for 'default3' to appear and 'default'+'default2' to still exist: %v", err)
	}
}

func TestStatusDefaulting(t *testing.T) {
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceDefaulting, true)()

	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionsClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// make sure we are not restricting fields to properties even in subschemas
	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition.Spec.Subresources = &apiextensionsv1beta1.CustomResourceSubresources{
		Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
	}
	noxuDefinition.Spec.Validation.OpenAPIV3Schema = &apiextensionsv1beta1.JSONSchemaProps{
		Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
			"status": {
				Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
					"num": {
						Type:    "integer",
						Default: &apiextensionsv1beta1.JSON{Raw: []byte("42")},
					},
				},
			},
		},
	}
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatalf("Unable to created crd %v: %v", noxuDefinition.Name, err)
	}

	// create the resource
	ns := "default"
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)
	foo := NewNoxuSubresourceInstance(ns, "foo")
	unstructured.RemoveNestedField(foo.Object, "status")
	createdNoxuInstance, err := instantiateCustomResource(t, foo, noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("Unable to create noxu instance: %v", err)
	}
	if status, found, err := unstructured.NestedFieldNoCopy(createdNoxuInstance.UnstructuredContent(), "status"); err != nil {
		t.Fatalf("Failed to get .status: %v", err)
	} else if found {
		t.Fatalf("Unexpected .status: %#v", status)
	}

	// update status
	unstructured.SetNestedField(createdNoxuInstance.Object, map[string]interface{}{}, "status")
	if createdNoxuInstance, err = noxuResourceClient.UpdateStatus(createdNoxuInstance, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update .status: %v", err)
	} else if statusNum, found, err := unstructured.NestedInt64(createdNoxuInstance.Object, "status", "num"); err != nil {
		t.Fatalf("Failed to get .status.num after update: %v", err)
	} else if !found {
		t.Fatalf("Expected .status.num to be set, but isn't")
	} else if statusNum != 42 {
		t.Fatalf("Expected .status.num to be 42, but it is: %v", statusNum)
	}
}
