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
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	autoscaling "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

var labelSelectorPath = ".status.labelSelector"
var anotherLabelSelectorPath = ".status.anotherLabelSelector"

func NewNoxuSubresourcesCRDs(scope apiextensionsv1beta1.ResourceScope) []*apiextensionsv1beta1.CustomResourceDefinition {
	return []*apiextensionsv1beta1.CustomResourceDefinition{
		// CRD that uses top-level subresources
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
				Scope: scope,
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
				Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
					Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
					Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
						SpecReplicasPath:   ".spec.replicas",
						StatusReplicasPath: ".status.replicas",
						LabelSelectorPath:  &labelSelectorPath,
					},
				},
			},
		},
		// CRD that uses per-version subresources
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
				Scope: scope,
				Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
					{
						Name:    "v1beta1",
						Served:  true,
						Storage: true,
						Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
							Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
							Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
								SpecReplicasPath:   ".spec.replicas",
								StatusReplicasPath: ".status.replicas",
								LabelSelectorPath:  &labelSelectorPath,
							},
						},
					},
					{
						Name:    "v1",
						Served:  true,
						Storage: false,
						Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
							Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
							Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
								SpecReplicasPath:   ".spec.replicas",
								StatusReplicasPath: ".status.replicas",
								LabelSelectorPath:  &anotherLabelSelectorPath,
							},
						},
					},
				},
			},
		},
	}
}

func NewNoxuSubresourceInstance(namespace, name, version string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": fmt.Sprintf("mygroup.example.com/%s", version),
			"kind":       "WishIHadChosenNoxu",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"spec": map[string]interface{}{
				"num":      int64(10),
				"replicas": int64(3),
			},
			"status": map[string]interface{}{
				"replicas": int64(7),
			},
		},
	}
}

func TestStatusSubresource(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			_, err = instantiateVersionedCustomResource(t, NewNoxuSubresourceInstance(ns, "foo", v.Name), noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}
			gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			// status should not be set after creation
			if val, ok := gottenNoxuInstance.Object["status"]; ok {
				t.Fatalf("status should not be set after creation, got %v", val)
			}

			// .status.num = 20
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(20), "status", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// .spec.num = 20
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(20), "spec", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// UpdateStatus should not update spec.
			// Check that .spec.num = 10 and .status.num = 20
			updatedStatusInstance, err := noxuResourceClient.UpdateStatus(gottenNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unable to update status: %v", err)
			}

			specNum, found, err := unstructured.NestedInt64(updatedStatusInstance.Object, "spec", "num")
			if !found || err != nil {
				t.Fatalf("unable to get .spec.num")
			}
			if specNum != int64(10) {
				t.Fatalf(".spec.num: expected: %v, got: %v", int64(10), specNum)
			}

			statusNum, found, err := unstructured.NestedInt64(updatedStatusInstance.Object, "status", "num")
			if !found || err != nil {
				t.Fatalf("unable to get .status.num")
			}
			if statusNum != int64(20) {
				t.Fatalf(".status.num: expected: %v, got: %v", int64(20), statusNum)
			}

			gottenNoxuInstance, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// .status.num = 40
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(40), "status", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// .spec.num = 40
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(40), "spec", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Update should not update status.
			// Check that .spec.num = 40 and .status.num = 20
			updatedInstance, err := noxuResourceClient.Update(gottenNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unable to update instance: %v", err)
			}

			specNum, found, err = unstructured.NestedInt64(updatedInstance.Object, "spec", "num")
			if !found || err != nil {
				t.Fatalf("unable to get .spec.num")
			}
			if specNum != int64(40) {
				t.Fatalf(".spec.num: expected: %v, got: %v", int64(40), specNum)
			}

			statusNum, found, err = unstructured.NestedInt64(updatedInstance.Object, "status", "num")
			if !found || err != nil {
				t.Fatalf("unable to get .status.num")
			}
			if statusNum != int64(20) {
				t.Fatalf(".status.num: expected: %v, got: %v", int64(20), statusNum)
			}
			noxuResourceClient.Delete("foo", &metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestScaleSubresource(t *testing.T) {
	groupResource := schema.GroupResource{
		Group:    "mygroup.example.com",
		Resource: "noxus",
	}

	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		for _, v := range noxuDefinition.Spec.Versions {
			// Start with a new CRD, so that the object doesn't have resourceVersion
			noxuDefinition := noxuDefinition.DeepCopy()

			subresources, err := getSubresourcesForVersion(noxuDefinition, v.Name)
			if err != nil {
				t.Fatal(err)
			}
			// set invalid json path for specReplicasPath
			subresources.Scale.SpecReplicasPath = "foo,bar"
			_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err == nil {
				t.Fatalf("unexpected non-error: specReplicasPath should be a valid json path under .spec")
			}

			subresources.Scale.SpecReplicasPath = ".spec.replicas"
			noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
			if err != nil {
				t.Fatal(err)
			}

			ns := "not-the-default"
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			_, err = instantiateVersionedCustomResource(t, NewNoxuSubresourceInstance(ns, "foo", v.Name), noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}

			scaleClient, err := fixtures.CreateNewVersionedScaleClient(noxuDefinition, config, v.Name)
			if err != nil {
				t.Fatal(err)
			}

			// set .status.labelSelector = bar
			gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, "bar", strings.Split((*subresources.Scale.LabelSelectorPath)[1:], ".")...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, err = noxuResourceClient.UpdateStatus(gottenNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unable to update status: %v", err)
			}

			// get the scale object
			gottenScale, err := scaleClient.Scales("not-the-default").Get(groupResource, "foo")
			if err != nil {
				t.Fatal(err)
			}
			if gottenScale.Spec.Replicas != 3 {
				t.Fatalf("Scale.Spec.Replicas: expected: %v, got: %v", 3, gottenScale.Spec.Replicas)
			}
			if gottenScale.Status.Selector != "bar" {
				t.Fatalf("Scale.Status.Selector: expected: %v, got: %v", "bar", gottenScale.Status.Selector)
			}

			// check self link
			expectedSelfLink := fmt.Sprintf("/apis/mygroup.example.com/%s/namespaces/not-the-default/noxus/foo/scale", v.Name)
			if gottenScale.GetSelfLink() != expectedSelfLink {
				t.Fatalf("Scale.Metadata.SelfLink: expected: %v, got: %v", expectedSelfLink, gottenScale.GetSelfLink())
			}

			// update the scale object
			// check that spec is updated, but status is not
			gottenScale.Spec.Replicas = 5
			gottenScale.Status.Selector = "baz"
			updatedScale, err := scaleClient.Scales("not-the-default").Update(groupResource, gottenScale)
			if err != nil {
				t.Fatal(err)
			}
			if updatedScale.Spec.Replicas != 5 {
				t.Fatalf("replicas: expected: %v, got: %v", 5, updatedScale.Spec.Replicas)
			}
			if updatedScale.Status.Selector != "bar" {
				t.Fatalf("scale should not update status: expected %v, got: %v", "bar", updatedScale.Status.Selector)
			}

			// check that .spec.replicas = 5, but status is not updated
			updatedNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			specReplicas, found, err := unstructured.NestedInt64(updatedNoxuInstance.Object, "spec", "replicas")
			if !found || err != nil {
				t.Fatalf("unable to get .spec.replicas")
			}
			if specReplicas != 5 {
				t.Fatalf("replicas: expected: %v, got: %v", 5, specReplicas)
			}
			statusLabelSelector, found, err := unstructured.NestedString(updatedNoxuInstance.Object, strings.Split((*subresources.Scale.LabelSelectorPath)[1:], ".")...)
			if !found || err != nil {
				t.Fatalf("unable to get %s", *subresources.Scale.LabelSelectorPath)
			}
			if statusLabelSelector != "bar" {
				t.Fatalf("scale should not update status: expected %v, got: %v", "bar", statusLabelSelector)
			}

			// validate maximum value
			// set .spec.replicas = math.MaxInt64
			gottenNoxuInstance, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(math.MaxInt64), "spec", "replicas")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, err = noxuResourceClient.Update(gottenNoxuInstance, metav1.UpdateOptions{})
			if err == nil {
				t.Fatalf("unexpected non-error: .spec.replicas should be less than 2147483647")
			}
			noxuResourceClient.Delete("foo", &metav1.DeleteOptions{})
			if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestValidationSchemaWithStatus(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	// fields other than properties in root schema are not allowed
	noxuDefinition := newNoxuValidationCRDs(apiextensionsv1beta1.NamespaceScoped)[0]
	noxuDefinition.Spec.Subresources = &apiextensionsv1beta1.CustomResourceSubresources{
		Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
	}
	_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err == nil {
		t.Fatalf(`unexpected non-error, expected: must not have "additionalProperties" at the root of the schema if the status subresource is enabled`)
	}

	// make sure we are not restricting fields to properties even in subschemas
	noxuDefinition.Spec.Validation.OpenAPIV3Schema = &apiextensionsv1beta1.JSONSchemaProps{
		Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
			"spec": {
				Description: "Validation for spec",
				Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
					"replicas": {
						Type: "integer",
					},
				},
			},
		},
		Required:    []string{"spec"},
		Description: "This is a description at the root of the schema",
	}
	_, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatalf("unable to created crd %v: %v", noxuDefinition.Name, err)
	}
}

func TestValidateOnlyStatus(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	// UpdateStatus should validate only status
	// 1. create a crd with max value of .spec.num = 10 and .status.num = 10
	// 2. create a cr with .spec.num = 10 and .status.num = 10 (valid)
	// 3. update the spec of the cr with .spec.num = 15 (spec is invalid), expect no error
	// 4. update the spec of the cr with .spec.num = 15 (spec is invalid), expect error

	// max value of spec.num = 10 and status.num = 10
	schema := &apiextensionsv1beta1.JSONSchemaProps{
		Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
			"spec": {
				Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
					"num": {
						Type:    "integer",
						Maximum: float64Ptr(10),
					},
				},
			},
			"status": {
				Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
					"num": {
						Type:    "integer",
						Maximum: float64Ptr(10),
					},
				},
			},
		},
	}

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for i, noxuDefinition := range noxuDefinitions {
		if i == 0 {
			noxuDefinition.Spec.Validation = &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: schema,
			}
		} else {
			noxuDefinition.Spec.Versions[0].Schema = &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: schema,
			}
			schemaWithDescription := schema.DeepCopy()
			schemaWithDescription.Description = "test"
			noxuDefinition.Spec.Versions[1].Schema = &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: schemaWithDescription,
			}
		}

		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}
		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)

			// set .spec.num = 10 and .status.num = 10
			noxuInstance := NewNoxuSubresourceInstance(ns, "foo", v.Name)
			err = unstructured.SetNestedField(noxuInstance.Object, int64(10), "status", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			createdNoxuInstance, err := instantiateVersionedCustomResource(t, noxuInstance, noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}

			// update the spec with .spec.num = 15, expecting no error
			err = unstructured.SetNestedField(createdNoxuInstance.Object, int64(15), "spec", "num")
			if err != nil {
				t.Fatalf("unexpected error setting .spec.num: %v", err)
			}
			createdNoxuInstance, err = noxuResourceClient.UpdateStatus(createdNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// update with .status.num = 15, expecting an error
			err = unstructured.SetNestedField(createdNoxuInstance.Object, int64(15), "status", "num")
			if err != nil {
				t.Fatalf("unexpected error setting .status.num: %v", err)
			}
			_, err = noxuResourceClient.UpdateStatus(createdNoxuInstance, metav1.UpdateOptions{})
			if err == nil {
				t.Fatal("expected error, but got none")
			}
			statusError, isStatus := err.(*apierrors.StatusError)
			if !isStatus || statusError == nil {
				t.Fatalf("expected status error, got %T: %v", err, err)
			}
			if !strings.Contains(statusError.Error(), "Invalid value") {
				t.Fatalf("expected 'Invalid value' in error, got: %v", err)
			}
			noxuResourceClient.Delete("foo", &metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSubresourcesDiscovery(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		for _, v := range noxuDefinition.Spec.Versions {
			group := "mygroup.example.com"
			version := v.Name

			resources, err := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/" + version)
			if err != nil {
				t.Fatal(err)
			}

			if len(resources.APIResources) != 3 {
				t.Fatalf("Expected exactly the resources \"noxus\", \"noxus/status\" and \"noxus/scale\" in group version %v/%v via discovery, got: %v", group, version, resources.APIResources)
			}

			// check discovery info for status
			status := resources.APIResources[1]

			if status.Name != "noxus/status" {
				t.Fatalf("incorrect status via discovery: expected name: %v, got: %v", "noxus/status", status.Name)
			}

			if status.Namespaced != true {
				t.Fatalf("incorrect status via discovery: expected namespace: %v, got: %v", true, status.Namespaced)
			}

			if status.Kind != "WishIHadChosenNoxu" {
				t.Fatalf("incorrect status via discovery: expected kind: %v, got: %v", "WishIHadChosenNoxu", status.Kind)
			}

			expectedVerbs := []string{"get", "patch", "update"}
			sort.Strings(status.Verbs)
			if !reflect.DeepEqual([]string(status.Verbs), expectedVerbs) {
				t.Fatalf("incorrect status via discovery: expected: %v, got: %v", expectedVerbs, status.Verbs)
			}

			// check discovery info for scale
			scale := resources.APIResources[2]

			if scale.Group != autoscaling.GroupName {
				t.Fatalf("incorrect scale via discovery: expected group: %v, got: %v", autoscaling.GroupName, scale.Group)
			}

			if scale.Version != "v1" {
				t.Fatalf("incorrect scale via discovery: expected version: %v, got %v", "v1", scale.Version)
			}

			if scale.Name != "noxus/scale" {
				t.Fatalf("incorrect scale via discovery: expected name: %v, got: %v", "noxus/scale", scale.Name)
			}

			if scale.Namespaced != true {
				t.Fatalf("incorrect scale via discovery: expected namespace: %v, got: %v", true, scale.Namespaced)
			}

			if scale.Kind != "Scale" {
				t.Fatalf("incorrect scale via discovery: expected kind: %v, got: %v", "Scale", scale.Kind)
			}

			sort.Strings(scale.Verbs)
			if !reflect.DeepEqual([]string(scale.Verbs), expectedVerbs) {
				t.Fatalf("incorrect scale via discovery: expected: %v, got: %v", expectedVerbs, scale.Verbs)
			}
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGeneration(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
			_, err = instantiateVersionedCustomResource(t, NewNoxuSubresourceInstance(ns, "foo", v.Name), noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}

			// .metadata.generation = 1
			gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if gottenNoxuInstance.GetGeneration() != 1 {
				t.Fatalf(".metadata.generation should be 1 after creation")
			}

			// .status.num = 20
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(20), "status", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// UpdateStatus does not increment generation
			updatedStatusInstance, err := noxuResourceClient.UpdateStatus(gottenNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unable to update status: %v", err)
			}
			if updatedStatusInstance.GetGeneration() != 1 {
				t.Fatalf("updating status should not increment .metadata.generation: expected: %v, got: %v", 1, updatedStatusInstance.GetGeneration())
			}

			gottenNoxuInstance, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// .spec.num = 20
			err = unstructured.SetNestedField(gottenNoxuInstance.Object, int64(20), "spec", "num")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Update increments generation
			updatedInstance, err := noxuResourceClient.Update(gottenNoxuInstance, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("unable to update instance: %v", err)
			}
			if updatedInstance.GetGeneration() != 2 {
				t.Fatalf("updating spec should increment .metadata.generation: expected: %v, got: %v", 2, updatedStatusInstance.GetGeneration())
			}
			noxuResourceClient.Delete("foo", &metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSubresourcePatch(t *testing.T) {
	groupResource := schema.GroupResource{
		Group:    "mygroup.example.com",
		Resource: "noxus",
	}

	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinitions := NewNoxuSubresourcesCRDs(apiextensionsv1beta1.NamespaceScoped)
	for _, noxuDefinition := range noxuDefinitions {
		noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			t.Fatal(err)
		}

		ns := "not-the-default"
		for _, v := range noxuDefinition.Spec.Versions {
			noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)

			t.Logf("Creating foo")
			_, err = instantiateVersionedCustomResource(t, NewNoxuSubresourceInstance(ns, "foo", v.Name), noxuResourceClient, noxuDefinition, v.Name)
			if err != nil {
				t.Fatalf("unable to create noxu instance: %v", err)
			}

			scaleClient, err := fixtures.CreateNewVersionedScaleClient(noxuDefinition, config, v.Name)
			if err != nil {
				t.Fatal(err)
			}

			t.Logf("Patching .status.num to 999")
			patch := []byte(`{"spec": {"num":999}, "status": {"num":999}}`)
			patchedNoxuInstance, err := noxuResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{}, "status")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "status", "num") // .status.num should be 999
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 10, "spec", "num")    // .spec.num should remain 10
			rv, found, err := unstructured.NestedString(patchedNoxuInstance.UnstructuredContent(), "metadata", "resourceVersion")
			if err != nil {
				t.Fatal(err)
			}
			if !found {
				t.Fatalf("metadata.resourceVersion not found")
			}

			// this call waits for the resourceVersion to be reached in the cache before returning.
			// We need to do this because the patch gets its initial object from the storage, and the cache serves that.
			// If it is out of date, then our initial patch is applied to an old resource version, which conflicts
			// and then the updated object shows a conflicting diff, which permanently fails the patch.
			// This gives expected stability in the patch without retrying on an known number of conflicts below in the test.
			// See https://issue.k8s.io/42644
			_, err = noxuResourceClient.Get("foo", metav1.GetOptions{ResourceVersion: patchedNoxuInstance.GetResourceVersion()})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// no-op patch
			t.Logf("Patching .status.num again to 999")
			patchedNoxuInstance, err = noxuResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{}, "status")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// make sure no-op patch does not increment resourceVersion
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "status", "num")
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 10, "spec", "num")
			expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

			// empty patch
			t.Logf("Applying empty patch")
			patchedNoxuInstance, err = noxuResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`), metav1.PatchOptions{}, "status")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// an empty patch is a no-op patch. make sure it does not increment resourceVersion
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "status", "num")
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 10, "spec", "num")
			expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

			t.Logf("Patching .spec.replicas to 7")
			patch = []byte(`{"spec": {"replicas":7}, "status": {"replicas":7}}`)
			patchedNoxuInstance, err = noxuResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{}, "scale")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 7, "spec", "replicas")
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 0, "status", "replicas") // .status.replicas should remain 0
			rv, found, err = unstructured.NestedString(patchedNoxuInstance.UnstructuredContent(), "metadata", "resourceVersion")
			if err != nil {
				t.Fatal(err)
			}
			if !found {
				t.Fatalf("metadata.resourceVersion not found")
			}

			// this call waits for the resourceVersion to be reached in the cache before returning.
			// We need to do this because the patch gets its initial object from the storage, and the cache serves that.
			// If it is out of date, then our initial patch is applied to an old resource version, which conflicts
			// and then the updated object shows a conflicting diff, which permanently fails the patch.
			// This gives expected stability in the patch without retrying on an known number of conflicts below in the test.
			// See https://issue.k8s.io/42644
			_, err = noxuResourceClient.Get("foo", metav1.GetOptions{ResourceVersion: patchedNoxuInstance.GetResourceVersion()})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Scale.Spec.Replicas = 7 but Scale.Status.Replicas should remain 0
			gottenScale, err := scaleClient.Scales("not-the-default").Get(groupResource, "foo")
			if err != nil {
				t.Fatal(err)
			}
			if gottenScale.Spec.Replicas != 7 {
				t.Fatalf("Scale.Spec.Replicas: expected: %v, got: %v", 7, gottenScale.Spec.Replicas)
			}
			if gottenScale.Status.Replicas != 0 {
				t.Fatalf("Scale.Status.Replicas: expected: %v, got: %v", 0, gottenScale.Spec.Replicas)
			}

			// no-op patch
			t.Logf("Patching .spec.replicas again to 7")
			patchedNoxuInstance, err = noxuResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{}, "scale")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// make sure no-op patch does not increment resourceVersion
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 7, "spec", "replicas")
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 0, "status", "replicas")
			expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

			// empty patch
			t.Logf("Applying empty patch")
			patchedNoxuInstance, err = noxuResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`), metav1.PatchOptions{}, "scale")
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// an empty patch is a no-op patch. make sure it does not increment resourceVersion
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 7, "spec", "replicas")
			expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 0, "status", "replicas")
			expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

			// make sure strategic merge patch is not supported for both status and scale
			_, err = noxuResourceClient.Patch("foo", types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "status")
			if err == nil {
				t.Fatalf("unexpected non-error: strategic merge patch is not supported for custom resources")
			}

			_, err = noxuResourceClient.Patch("foo", types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "scale")
			if err == nil {
				t.Fatalf("unexpected non-error: strategic merge patch is not supported for custom resources")
			}
			noxuResourceClient.Delete("foo", &metav1.DeleteOptions{})
		}
		if err := fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient); err != nil {
			t.Fatal(err)
		}
	}
}
