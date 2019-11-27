/*
Copyright 2019 The Kubernetes Authors.

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

package etcd

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/typed/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/controller/finalizer"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	apiregistrationclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset/typed/apiregistration/v1"
)

// TestOverlappingBuiltInResources ensures the list of group-resources the custom resource finalizer should skip is up to date
func TestOverlappingBuiltInResources(t *testing.T) {
	// Verify built-in resources that overlap with computed CRD storage paths are listed in OverlappingBuiltInResources()
	detectedOverlappingResources := map[schema.GroupResource]bool{}
	for gvr, gvrData := range GetEtcdStorageData() {
		if !strings.HasSuffix(gvr.Group, ".k8s.io") {
			// only fully-qualified group names can exist as CRDs
			continue
		}
		if !strings.Contains(gvrData.ExpectedEtcdPath, "/"+gvr.Group+"/"+gvr.Resource+"/") {
			// CRDs persist in storage under .../<group>/<resource>/...
			continue
		}
		detectedOverlappingResources[gvr.GroupResource()] = true
	}

	for detected := range detectedOverlappingResources {
		if !finalizer.OverlappingBuiltInResources()[detected] {
			t.Errorf("built-in resource %#v would overlap with custom resource storage if a CRD was created for the same group/resource", detected)
			t.Errorf("add %#v to the OverlappingBuiltInResources() list to prevent deletion by the CRD finalizer", detected)
		}
	}
	for skip := range finalizer.OverlappingBuiltInResources() {
		if !detectedOverlappingResources[skip] {
			t.Errorf("resource %#v does not overlap with any built-in resources in storage, but is skipped for CRD finalization by OverlappingBuiltInResources()", skip)
			t.Errorf("remove %#v from OverlappingBuiltInResources() to ensure CRD finalization cleans up stored custom resources", skip)
		}
	}
}

// TestOverlappingCustomResourceAPIService ensures creating and deleting a custom resource overlapping with APIServices does not destroy APIService data
func TestOverlappingCustomResourceAPIService(t *testing.T) {
	master := StartRealMasterOrDie(t)
	defer master.Cleanup()

	apiServiceClient, err := apiregistrationclient.NewForConfig(master.Config)
	if err != nil {
		t.Fatal(err)
	}
	crdClient, err := crdclient.NewForConfig(master.Config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(master.Config)
	if err != nil {
		t.Fatal(err)
	}

	// Verify APIServices can be listed
	apiServices, err := apiServiceClient.APIServices().List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	apiServiceNames := sets.NewString()
	for _, s := range apiServices.Items {
		apiServiceNames.Insert(s.Name)
	}
	if len(apiServices.Items) == 0 {
		t.Fatal("expected APIService objects, got none")
	}

	// Create a CRD defining an overlapping apiregistration.k8s.io apiservices resource with an incompatible schema
	crdCRD, err := crdClient.CustomResourceDefinitions().Create(&apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "apiservices.apiregistration.k8s.io",
			Annotations: map[string]string{"api-approved.kubernetes.io": "unapproved, testing only"},
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "apiregistration.k8s.io",
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{Plural: "apiservices", Singular: "customapiservice", Kind: "CustomAPIService", ListKind: "CustomAPIServiceList"},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type:     "object",
							Required: []string{"foo"},
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"foo": {Type: "string"},
								"bar": {Type: "string", Default: &apiextensionsv1.JSON{Raw: []byte(`"default"`)}},
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Wait until it is established
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := crdClient.CustomResourceDefinitions().Get(crdCRD.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range crd.Status.Conditions {
			if condition.Status == apiextensionsv1.ConditionTrue && condition.Type == apiextensionsv1.Established {
				return true, nil
			}
		}
		conditionJSON, _ := json.Marshal(crd.Status.Conditions)
		t.Logf("waiting for establishment (conditions: %s)", string(conditionJSON))
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Make sure API requests are still handled by the built-in handler (and return built-in kinds)

	// Listing v1 succeeds
	v1DynamicList, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "apiregistration.k8s.io", Version: "v1", Resource: "apiservices"}).List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Result was served by built-in handler, not CR handler
	if _, hasDefaultedCRField := v1DynamicList.Items[0].Object["spec"].(map[string]interface{})["bar"]; hasDefaultedCRField {
		t.Fatalf("expected no CR defaulting, got %#v", v1DynamicList.Items[0].Object)
	}

	// Creating v1 succeeds (built-in validation, not CR validation)
	testAPIService, err := apiServiceClient.APIServices().Create(&apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:                "example.com",
			Version:              "v1",
			VersionPriority:      100,
			GroupPriorityMinimum: 100,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = apiServiceClient.APIServices().Delete(testAPIService.Name, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// discovery is handled by the built-in handler
	v1Resources, err := master.Client.Discovery().ServerResourcesForGroupVersion("apiregistration.k8s.io/v1")
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range v1Resources.APIResources {
		if r.Name == "apiservices" {
			if r.Kind != "APIService" {
				t.Errorf("expected kind=APIService in discovery, got %s", r.Kind)
			}
		}
	}
	v2Resources, err := master.Client.Discovery().ServerResourcesForGroupVersion("apiregistration.k8s.io/v2")
	if err == nil {
		t.Fatalf("expected error looking up apiregistration.k8s.io/v2 discovery, got %#v", v2Resources)
	}

	// Delete the overlapping CRD
	err = crdClient.CustomResourceDefinitions().Delete(crdCRD.Name, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Make sure the CRD deletion succeeds
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := crdClient.CustomResourceDefinitions().Get(crdCRD.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		conditionJSON, _ := json.Marshal(crd.Status.Conditions)
		t.Logf("waiting for deletion (conditions: %s)", string(conditionJSON))
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Make sure APIService objects are not removed
	time.Sleep(5 * time.Second)
	finalAPIServices, err := apiServiceClient.APIServices().List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(finalAPIServices.Items) != len(apiServices.Items) {
		t.Fatalf("expected %d APIService objects, got %d", len(apiServices.Items), len(finalAPIServices.Items))
	}
}

// TestOverlappingCustomResourceCustomResourceDefinition ensures creating and deleting a custom resource overlapping with CustomResourceDefinition does not destroy CustomResourceDefinition data
func TestOverlappingCustomResourceCustomResourceDefinition(t *testing.T) {
	master := StartRealMasterOrDie(t)
	defer master.Cleanup()

	crdClient, err := crdclient.NewForConfig(master.Config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(master.Config)
	if err != nil {
		t.Fatal(err)
	}

	// Verify CustomResourceDefinitions can be listed
	crds, err := crdClient.CustomResourceDefinitions().List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	crdNames := sets.NewString()
	for _, s := range crds.Items {
		crdNames.Insert(s.Name)
	}
	if len(crds.Items) == 0 {
		t.Fatal("expected CustomResourceDefinition objects, got none")
	}

	// Create a CRD defining an overlapping apiregistration.k8s.io apiservices resource with an incompatible schema
	crdCRD, err := crdClient.CustomResourceDefinitions().Create(&apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "customresourcedefinitions.apiextensions.k8s.io",
			Annotations: map[string]string{"api-approved.kubernetes.io": "unapproved, testing only"},
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "apiextensions.k8s.io",
			Scope: apiextensionsv1.ClusterScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "customresourcedefinitions",
				Singular: "customcustomresourcedefinition",
				Kind:     "CustomCustomResourceDefinition",
				ListKind: "CustomAPIServiceList",
			},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
							Type:     "object",
							Required: []string{"foo"},
							Properties: map[string]apiextensionsv1.JSONSchemaProps{
								"foo": {Type: "string"},
								"bar": {Type: "string", Default: &apiextensionsv1.JSON{Raw: []byte(`"default"`)}},
							},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Wait until it is established
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := crdClient.CustomResourceDefinitions().Get(crdCRD.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range crd.Status.Conditions {
			if condition.Status == apiextensionsv1.ConditionTrue && condition.Type == apiextensionsv1.Established {
				return true, nil
			}
		}
		conditionJSON, _ := json.Marshal(crd.Status.Conditions)
		t.Logf("waiting for establishment (conditions: %s)", string(conditionJSON))
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Make sure API requests are still handled by the built-in handler (and return built-in kinds)

	// Listing v1 succeeds
	v1DynamicList, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "v1", Resource: "customresourcedefinitions"}).List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Result was served by built-in handler, not CR handler
	if _, hasDefaultedCRField := v1DynamicList.Items[0].Object["spec"].(map[string]interface{})["bar"]; hasDefaultedCRField {
		t.Fatalf("expected no CR defaulting, got %#v", v1DynamicList.Items[0].Object)
	}

	// Updating v1 succeeds (built-in validation, not CR validation)
	_, err = crdClient.CustomResourceDefinitions().Patch(crdCRD.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"updated"}}}`))
	if err != nil {
		t.Fatal(err)
	}

	// discovery is handled by the built-in handler
	v1Resources, err := master.Client.Discovery().ServerResourcesForGroupVersion("apiextensions.k8s.io/v1")
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range v1Resources.APIResources {
		if r.Name == "customresourcedefinitions" {
			if r.Kind != "CustomResourceDefinition" {
				t.Errorf("expected kind=CustomResourceDefinition in discovery, got %s", r.Kind)
			}
		}
	}
	v2Resources, err := master.Client.Discovery().ServerResourcesForGroupVersion("apiextensions.k8s.io/v2")
	if err == nil {
		t.Fatalf("expected error looking up apiregistration.k8s.io/v2 discovery, got %#v", v2Resources)
	}

	// Delete the overlapping CRD
	err = crdClient.CustomResourceDefinitions().Delete(crdCRD.Name, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Make sure the CRD deletion succeeds
	if err := wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := crdClient.CustomResourceDefinitions().Get(crdCRD.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		conditionJSON, _ := json.Marshal(crd.Status.Conditions)
		t.Logf("waiting for deletion (conditions: %s)", string(conditionJSON))
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Make sure other CustomResourceDefinition objects are not removed
	time.Sleep(5 * time.Second)
	finalCRDs, err := crdClient.CustomResourceDefinitions().List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(finalCRDs.Items) != len(crds.Items) {
		t.Fatalf("expected %d APIService objects, got %d", len(crds.Items), len(finalCRDs.Items))
	}
}
