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
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
)

func TestCustomResourceDefaulting(t *testing.T) {
	// enable alpha feature CustomResourceDefaulting
	if err := utilfeature.DefaultFeatureGate.Set("CustomResourceValidation=true,CustomResourceDefaulting=true"); err != nil {
		t.Errorf("failed to enable feature gate for CustomResourceDefaulting: %v", err)
	}

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	crd := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("42")}}
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(crd, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating CR and expect 'default' field to be set")
	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, crd)
	obj, err := instantiateCustomResource(t, newNoxuValidationInstance(ns, "foo"), noxuResourceClient, crd)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}

	t.Logf("CR created: %#v", obj.UnstructuredContent())

	got, found := unstructured.NestedFieldCopy(obj.UnstructuredContent(), "default")
	if !found {
		t.Fatalf("expected 'default' field")
	}
	if gotInt64, foundInt64 := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); !foundInt64 || gotInt64 != 42 {
		t.Fatalf("expected 'default' field with value 42, got: %v", got)
	}

	t.Logf("Adding 'default2' to schema")
	crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default2"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("42")}}
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd); err != nil {
		t.Fatal(err)
	}

	t.Logf("Getting old CR again to check for new 'default2 field")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if gotInt64, foundInt64 := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); foundInt64 && gotInt64 == 42 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("failed waiting for 'default2' to appear: %v", err)
	}

	t.Logf("Removing both defaults from schema")
	crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default")
	delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default2")
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd); err != nil {
		t.Fatal(err)
	}

	t.Logf("Getting old CR again, waiting for unpersisted 'default2' field to disappear, but not 'default'")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if _, foundDefault := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); !foundDefault {
			return false, fmt.Errorf("property 'default' is gone, but should have been persisted")
		}
		if _, foundDefault2 := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); foundDefault2 {
			return false, nil
		}

		// default2 should be gone, default should still exist
		return true, nil
	})
	if err != nil {
		t.Fatalf("failed waiting for 'default2' to disappear: %v", err)
	}

	t.Logf("Re-adding defaults to schema with different value 43")
	crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("43")}}
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default2"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("43")}}
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd); err != nil {
		t.Fatal(err)
	}

	t.Logf("Waiting for 'default2' field to appear on update with value 43, but 'default' stay with persisted value 42")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// remove to enforce defaulting
		unstructured.RemoveNestedField(obj.UnstructuredContent(), "default2")
		obj, err = noxuResourceClient.Update(obj)
		if err != nil {
			return false, err
		}
		if gotDefault, foundDefault := unstructured.NestedInt64(obj.UnstructuredContent(), "default"); !foundDefault {
			return false, fmt.Errorf("property 'default' is gone, but should have been persisted")
		} else if gotDefault != 42 {
			return false, fmt.Errorf("property 'default' has not the persistet value 42, but got: %d", gotDefault)
		}
		if gotDefault2, foundDefault2 := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); !foundDefault2 {
			return false, nil
		} else if gotDefault2 != 43 {
			return false, fmt.Errorf("expected 'default2' field value 43, but got %d", gotDefault2)
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("failed waiting for 'default2' to change value: %v", err)
	}

	t.Logf("Removing 'default2' again from schema, and add 'default3' as a trigger")
	crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default")
	delete(crd.Spec.Validation.OpenAPIV3Schema.Properties, "default2")
	crd.Spec.Validation.OpenAPIV3Schema.Properties["default3"] = apiextensionsv1beta1.JSONSchemaProps{Default: &apiextensionsv1beta1.JSON{[]byte("44")}}
	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd); err != nil {
		t.Fatal(err)
	}

	t.Logf("Waiting for 'default3' field to appear, but 'default2' to still exist because it was persisted by update")
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		obj, err = noxuResourceClient.Get(obj.GetName(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if _, foundDefault := unstructured.NestedInt64(obj.UnstructuredContent(), "default3"); !foundDefault {
			return false, nil
		}
		if _, foundDefault2 := unstructured.NestedInt64(obj.UnstructuredContent(), "default2"); !foundDefault2 {
			return false, fmt.Errorf("property 'default2' is gone, but should have been persisted")
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("failed waiting for 'default3' to appear and 'default2' to still exist: %v", err)
	}
}
