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
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	autoscaling "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/client-go/dynamic"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
)

var labelSelectorPath = ".status.labelSelector"

func NewNoxuSubresourcesCRD(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
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
			Scope: scope,
			Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
				Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
				Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
					SpecReplicasPath:   ".spec.replicas",
					StatusReplicasPath: ".status.replicas",
					LabelSelectorPath:  &labelSelectorPath,
				},
			},
		},
	}
}

func NewNoxuSubresourceInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
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
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuStatusResourceClient := NewNamespacedCustomResourceStatusClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, NewNoxuSubresourceInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
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
	updatedStatusInstance, err := noxuStatusResourceClient.Update(gottenNoxuInstance)
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
	updatedInstance, err := noxuResourceClient.Update(gottenNoxuInstance)
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
}

func TestScaleSubresource(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	groupResource := schema.GroupResource{
		Group:    "mygroup.example.com",
		Resource: "noxus",
	}

	stopCh, config, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	clientPool := dynamic.NewDynamicClientPool(config)

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)

	// set invalid json path for specReplicasPath
	noxuDefinition.Spec.Subresources.Scale.SpecReplicasPath = "foo,bar"
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err == nil {
		t.Fatalf("unexpected non-error: specReplicasPath should be a valid json path under .spec")
	}

	noxuDefinition.Spec.Subresources.Scale.SpecReplicasPath = ".spec.replicas"
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuStatusResourceClient := NewNamespacedCustomResourceStatusClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, NewNoxuSubresourceInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}

	scaleClient, err := testserver.CreateNewScaleClient(noxuDefinition, config)
	if err != nil {
		t.Fatal(err)
	}

	// set .status.labelSelector = bar
	gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	err = unstructured.SetNestedField(gottenNoxuInstance.Object, "bar", "status", "labelSelector")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = noxuStatusResourceClient.Update(gottenNoxuInstance)
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
	expectedSelfLink := "/apis/mygroup.example.com/v1beta1/namespaces/not-the-default/noxus/foo/scale"
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
	statusLabelSelector, found, err := unstructured.NestedString(updatedNoxuInstance.Object, "status", "labelSelector")
	if !found || err != nil {
		t.Fatalf("unable to get .status.labelSelector")
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
	_, err = noxuResourceClient.Update(gottenNoxuInstance)
	if err == nil {
		t.Fatalf("unexpected non-error: .spec.replicas should be less than 2147483647")
	}
}

func TestValidationSchema(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	stopCh, config, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	clientPool := dynamic.NewDynamicClientPool(config)

	// fields other than properties in root schema are not allowed
	noxuDefinition := newNoxuValidationCRD(apiextensionsv1beta1.NamespaceScoped)
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err == nil {
		t.Fatalf("unexpected non-error: if subresources for custom resources are enabled, only properties can be used at the root of the schema")
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
	}
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatalf("unable to created crd %v: %v", noxuDefinition.Name, err)
	}
}

func TestValidateOnlyStatus(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	// UpdateStatus should validate only status
	// 1. create a crd with max value of .spec.num = 10 and .status.num = 10
	// 2. create a cr with .spec.num = 10 and .status.num = 10 (valid)
	// 3. update the crd so that max value of .spec.num = 5 and .status.num = 10
	// 4. update the status of the cr with .status.num = 5 (spec is invalid)
	// validation passes becauses spec is not validated

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

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition.Spec.Validation = &apiextensionsv1beta1.CustomResourceValidation{
		OpenAPIV3Schema: schema,
	}

	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}
	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuStatusResourceClient := NewNamespacedCustomResourceStatusClient(ns, noxuVersionClient, noxuDefinition)

	// set .spec.num = 10 and .status.num = 10
	noxuInstance := NewNoxuSubresourceInstance(ns, "foo")
	err = unstructured.SetNestedField(noxuInstance.Object, int64(10), "status", "num")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	createdNoxuInstance, err := instantiateCustomResource(t, noxuInstance, noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}

	gottenCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get("noxus.mygroup.example.com", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// update the crd so that max value of spec.num = 5 and status.num = 10
	gottenCRD.Spec.Validation.OpenAPIV3Schema = &apiextensionsv1beta1.JSONSchemaProps{
		Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
			"spec": {
				Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
					"num": {
						Type:    "integer",
						Maximum: float64Ptr(5),
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

	if _, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD); err != nil {
		t.Fatal(err)
	}

	// update the status with .status.num = 5
	err = unstructured.SetNestedField(createdNoxuInstance.Object, int64(5), "status", "num")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// cr is updated even though spec is invalid
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := noxuStatusResourceClient.Update(createdNoxuInstance)
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

func TestSubresourcesDiscovery(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	stopCh, config, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	clientPool := dynamic.NewDynamicClientPool(config)

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	group := "mygroup.example.com"
	version := "v1beta1"

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

func TestGeneration(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuStatusResourceClient := NewNamespacedCustomResourceStatusClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, NewNoxuSubresourceInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
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
	updatedStatusInstance, err := noxuStatusResourceClient.Update(gottenNoxuInstance)
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
	updatedInstance, err := noxuResourceClient.Update(gottenNoxuInstance)
	if err != nil {
		t.Fatalf("unable to update instance: %v", err)
	}
	if updatedInstance.GetGeneration() != 2 {
		t.Fatalf("updating spec should increment .metadata.generation: expected: %v, got: %v", 2, updatedStatusInstance.GetGeneration())
	}
}

func TestSubresourcePatch(t *testing.T) {
	// enable alpha feature CustomResourceSubresources
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CustomResourceSubresources, true)()

	groupResource := schema.GroupResource{
		Group:    "mygroup.example.com",
		Resource: "noxus",
	}

	stopCh, config, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	clientPool := dynamic.NewDynamicClientPool(config)

	noxuDefinition := NewNoxuSubresourcesCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	noxuStatusResourceClient := NewNamespacedCustomResourceStatusClient(ns, noxuVersionClient, noxuDefinition)
	noxuScaleResourceClient := NewNamespacedCustomResourceScaleClient(ns, noxuVersionClient, noxuDefinition)
	_, err = instantiateCustomResource(t, NewNoxuSubresourceInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu instance: %v", err)
	}

	scaleClient, err := testserver.CreateNewScaleClient(noxuDefinition, config)
	if err != nil {
		t.Fatal(err)
	}

	patch := []byte(`{"spec": {"num":999}, "status": {"num":999}}`)
	patchedNoxuInstance, err := noxuStatusResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// .spec.num should remain 10
	specNum, found, err := unstructured.NestedInt64(patchedNoxuInstance.Object, "spec", "num")
	if !found || err != nil {
		t.Fatalf("unable to get .spec.num")
	}
	if specNum != 10 {
		t.Fatalf(".spec.num: expected: %v, got: %v", 10, specNum)
	}

	// .status.num should be 999
	statusNum, found, err := unstructured.NestedInt64(patchedNoxuInstance.Object, "status", "num")
	if !found || err != nil {
		t.Fatalf("unable to get .status.num")
	}
	if statusNum != 999 {
		t.Fatalf(".status.num: expected: %v, got: %v", 999, statusNum)
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
	_, err = noxuStatusResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// empty patch
	_, err = noxuStatusResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	patch = []byte(`{"spec": {"replicas":7}, "status": {"replicas":7}}`)
	patchedNoxuInstance, err = noxuScaleResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
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

	// Scale.Spec.Replicas = 7 but Scale.Status.Replicas should remain 7
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
	_, err = noxuScaleResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// empty patch
	_, err = noxuScaleResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// make sure strategic merge patch is not supported for both status and scale
	_, err = noxuStatusResourceClient.Patch("foo", types.StrategicMergePatchType, patch)
	if err == nil {
		t.Fatalf("unexpected non-error: strategic merge patch is not supported for custom resources")
	}

	_, err = noxuScaleResourceClient.Patch("foo", types.StrategicMergePatchType, patch)
	if err == nil {
		t.Fatalf("unexpected non-error: strategic merge patch is not supported for custom resources")
	}
}
