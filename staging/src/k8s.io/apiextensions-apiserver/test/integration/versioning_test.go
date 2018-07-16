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
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"

	"k8s.io/apiextensions-apiserver/test/integration/utils"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/dynamic"
)

func TestVersionedNamspacedScopedCRD(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	testSimpleCRUD(t, ns, noxuDefinition, dynamicClient)
}

func TestVersionedClusterScopedCRD(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	testSimpleCRUD(t, ns, noxuDefinition, dynamicClient)
}

func TestInvaludConversionForVersionedClusterScopedCRD(t *testing.T) {
	testInvalidConversion(t, "")
}

func TestInvaludConversionForVersionedNamespaceScopedCRD(t *testing.T) {
	testInvalidConversion(t, "not-default")
}

// testInvalidConversion tests the behaviour of CRD handler when the conversion strategy is invalid. This can happened
// when downgrafing a master from a version that has conversion strategies unknown to current API server.
func testInvalidConversion(t *testing.T, ns string) {

	scope := apiextensionsv1beta1.NamespaceScoped
	if ns == "" {
		scope = apiextensionsv1beta1.ClusterScoped
	}
	testVersion := "v1beta1"

	stopCh, apiExtensionClient, dynamicClient, store, err := testserver.StartDefaultServerWithClientsAndStorage()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)
	storeWithoutValidation := utils.NewStoreWrapperWithoutValidation(store)

	// Step 1: Creating a CRD with invalid conversion should fail
	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(scope)
	noxuDefinition.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{Strategy: "invalid_strategy"}
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err == nil {
		t.Errorf("validation should fail for %v", noxuDefinition)
	}
	if !strings.Contains(err.Error(), "invalid_strategy") {
		t.Fatalf("operation failed on something other than invalid conversion strategy: %s", err)
	}

	// Step 2: Create the invalid CRD using a store interface with no validation.
	// This will simulate a downgrade. Assuming the CRD is created by a future release with a future conversion
	// strategy that is unknown to this version of api server.
	internalCRD := &apiextensions.CustomResourceDefinition{}
	apiextensionsv1beta1.Convert_v1beta1_CustomResourceDefinition_To_apiextensions_CustomResourceDefinition(noxuDefinition, internalCRD, nil)

	_, err = storeWithoutValidation.Create(context.Background(), internalCRD, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	// wait for the handler to catch up
	if err = testserver.WaitForAllVersionsExistsInDiscovery(noxuDefinition, apiExtensionClient); err != nil {
		t.Fatal(err)
	}

	// Step 3: Create a CR should fail as the CRD has invalid conversion strategy
	noxuResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, testVersion)
	_, err = instantiateVersionedCustomResource(t, testserver.NewVersionedNoxuInstance(ns, "foo", testVersion), noxuResourceClient, noxuDefinition, testVersion)
	if err == nil {
		t.Errorf("operation should fail. the conversion strategy is unknown for %v", noxuDefinition)
	}
	if !strings.Contains(err.Error(), "invalid_strategy") {
		t.Fatalf("operation failed on something other than invalid conversion strategy: %s", err)
	}

	// Step 4: Any other operation should also fail. trying a get
	_, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err == nil {
		t.Errorf("operation should fail. the conversion strategy is unknown for %v", noxuDefinition)
	}
	if !strings.Contains(err.Error(), "invalid_strategy") {
		t.Fatalf("operation failed on something other than invalid conversion strategy: %s", err)
	}

	// Step 5: Update the CRD and correct conversion strategy
	gotNoxuDefinition, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, v1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gotNoxuDefinition.Spec.Conversion == nil {
		gotNoxuDefinition.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{}
	}
	gotNoxuDefinition.Spec.Conversion.Strategy = "no-op"
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gotNoxuDefinition)
	if err != nil {
		t.Fatal(err)
	}

	// Step 6: Create a normal CR should work now
	createdNoxuInstance, err := instantiateVersionedCustomResource(t, testserver.NewVersionedNoxuInstance(ns, "foo", testVersion), noxuResourceClient, noxuDefinition, testVersion)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	if e, a := noxuDefinition.Spec.Group+"/"+testVersion, createdNoxuInstance.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// Step 7: Test to see if there is access to that CR
	gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := testVersion, gottenNoxuInstance.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	// Step 8: Update the CRD and add back the unknown conversion strategy should fail
	gotNoxuDefinition, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, v1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gotNoxuDefinition.Spec.Conversion == nil {
		gotNoxuDefinition.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{}
	}
	gotNoxuDefinition.Spec.Conversion.Strategy = "invalid_strategy"
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gotNoxuDefinition)
	if err == nil {
		t.Errorf("noxuDefinition has invalid strategy and should fail in validation")
	}

	// Step 9: Update the CRD and add back the unknown conversion strategy, with disabled validation
	internalCRD = &apiextensions.CustomResourceDefinition{}
	apiextensionsv1beta1.Convert_v1beta1_CustomResourceDefinition_To_apiextensions_CustomResourceDefinition(gotNoxuDefinition, internalCRD, nil)
	_, _, err = storeWithoutValidation.Update(context.Background(), internalCRD.Name, rest.DefaultUpdatedObjectInfo(internalCRD), nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Step 10: Now all operations should fail again on the CR

	// wait for the handler to catch up
	err = wait.PollImmediate(50*time.Millisecond, 30*time.Second, func() (bool, error) {
		_, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
		return err != nil && strings.Contains(err.Error(), "invalid_strategy"), nil
	})
	_, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err == nil {
		t.Errorf("operation should fail on invalid conversion strategy")
	}
	if !strings.Contains(err.Error(), "invalid_strategy") {
		t.Fatalf("operation failed on something other than invalid conversion strategy: %s", err)
	}

	// Step 11: trying another operation
	_, err = instantiateVersionedCustomResource(t, testserver.NewVersionedNoxuInstance(ns, "bar", testVersion), noxuResourceClient, noxuDefinition, testVersion)
	if err == nil {
		t.Errorf("operation should fail on invalid conversion strategy")
	}
	if !strings.Contains(err.Error(), "invalid_strategy") {
		t.Fatalf("operation failed on something other than invalid conversion strategy: %s", err)
	}

	// Step 12: return the CRD to its normal state and everything should go back to normal
	gotNoxuDefinition, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, v1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gotNoxuDefinition.Spec.Conversion == nil {
		gotNoxuDefinition.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{}
	}
	gotNoxuDefinition.Spec.Conversion.Strategy = "no-op"
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gotNoxuDefinition)
	if err != nil {
		t.Fatal(err)
	}
	err = wait.PollImmediate(50*time.Millisecond, 30*time.Second, func() (bool, error) {
		_, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
		return err == nil, nil
	})
	_, err = noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

func TestDisabledConversionForVersionedClusterScopedCRD(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition.Spec.Conversion = &apiextensionsv1beta1.CustomResourceConversion{Strategy: "disable"}
	noxuDefinition, err = testserver.CreateNewCustomResourceDefinitionWatchUnsafe(noxuDefinition, apiExtensionClient)
	if err != nil {
		t.Fatal(err)
	}
	ns := ""
	testVersionedCRDwithDisabledConversion(t, ns, noxuDefinition, dynamicClient)
}

func testVersionedCRDwithDisabledConversion(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, dynamicClient dynamic.Interface) {
	noxuResourceClients := map[string]dynamic.ResourceInterface{}
	disabledVersions := map[string]bool{}
	storageVersion := ""
	for _, v := range noxuDefinition.Spec.Versions {
		disabledVersions[v.Name] = !v.Served
		if v.Storage {
			storageVersion = v.Name
		}
	}
	for _, v := range noxuDefinition.Spec.Versions {
		noxuResourceClients[v.Name] = newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)
	}

	createdNoxuInstance, err := instantiateVersionedCustomResource(t, testserver.NewVersionedNoxuInstance(ns, "foo", storageVersion), noxuResourceClients[storageVersion], noxuDefinition, storageVersion)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	if e, a := noxuDefinition.Spec.Group+"/"+storageVersion, createdNoxuInstance.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for version, noxuResourceClient := range noxuResourceClients {
		// Get test
		gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
		if err == nil && gottenNoxuInstance.GroupVersionKind().Kind == "Status" {
			err = fmt.Errorf("%s", gottenNoxuInstance.Object["status"].(string))
		}
		if disabledVersions[version] {
			if err == nil {
				t.Errorf("expected the get operation fail for disabled version %s", version)
			}
		} else if version != storageVersion {
			if err == nil {
				t.Errorf("expected the get operation fail as the request version %v is different than storageVersion %s, %v", version, storageVersion, gottenNoxuInstance)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}

			if e, a := version, gottenNoxuInstance.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}
		}
		// List test
		listWithItem, err := noxuResourceClient.List(metav1.ListOptions{})
		if err == nil && gottenNoxuInstance.GroupVersionKind().Kind == "Status" {
			err = fmt.Errorf("%s", gottenNoxuInstance.Object["status"].(string))
		}
		if disabledVersions[version] {
			if err == nil {
				t.Errorf("expected the list operation fail for disabled version %s", version)
			}
		} else if version != storageVersion {
			if err == nil {
				t.Errorf("expected the list operation fail as the request version %v is different than storageVersion %s", version, storageVersion)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			if e, a := 1, len(listWithItem.Items); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := version, listWithItem.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := version, listWithItem.Items[0].GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, got %v", e, a)
			}
		}
	}

	if err := noxuResourceClients[storageVersion].DeleteCollection(metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		t.Fatal(err)
	}
}

func TestStoragedVersionInNamespacedCRDStatus(t *testing.T) {
	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.NamespaceScoped)
	ns := "not-the-default"
	testStoragedVersionInCRDStatus(t, ns, noxuDefinition)
}

func TestStoragedVersionInClusterScopedCRDStatus(t *testing.T) {
	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)
	ns := ""
	testStoragedVersionInCRDStatus(t, ns, noxuDefinition)
}

func testStoragedVersionInCRDStatus(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition) {
	versionsV1Beta1Storage := []apiextensionsv1beta1.CustomResourceDefinitionVersion{
		{
			Name:    "v1beta1",
			Served:  true,
			Storage: true,
		},
		{
			Name:    "v1beta2",
			Served:  true,
			Storage: false,
		},
	}
	versionsV1Beta2Storage := []apiextensionsv1beta1.CustomResourceDefinitionVersion{
		{
			Name:    "v1beta1",
			Served:  true,
			Storage: false,
		},
		{
			Name:    "v1beta2",
			Served:  true,
			Storage: true,
		},
	}
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition.Spec.Versions = versionsV1Beta1Storage
	noxuDefinition, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	// The storage version list should be initilized to storage version
	crd, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := []string{"v1beta1"}, crd.Status.StoredVersions; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	// Changing CRD storage version should be reflected immediately
	crd.Spec.Versions = versionsV1Beta2Storage
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(crd)
	if err != nil {
		t.Fatal(err)
	}
	crd, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := []string{"v1beta1", "v1beta2"}, crd.Status.StoredVersions; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	err = testserver.DeleteCustomResourceDefinition(crd, apiExtensionClient)
	if err != nil {
		t.Fatal(err)
	}
}
