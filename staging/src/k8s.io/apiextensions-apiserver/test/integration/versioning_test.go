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
	"reflect"
	"testing"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
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

// TestConvertVersion goes through the versioning lifecycle by
// converting from v1beta1 to v1beta2 in the following way:
//
// 1. Create a CRD with version = v1beta1, ensure that v1beta1 appears in discovery
// and the API is usable by creating a CR with version = v1beta1.
// 2. Add a version = v1beta2 with served = true. Ensure that both v1beta1 and v1beta2 appear in discovery
// and both APIs are usable by checking that the old CR still has version = v1beta1 and creating a new CR with version = v1beta2.
// 3. Set v1beta2 with stored = true. Create a new CR and ensure that it can persist with version = v1beta2
// and update the CR with version = v1beta1 to v1beta2, so that all CRs now have version = v1beta2.
// 4. Set v1beta1 with served = false, ensure that it disappears from discovery
// and the API is no longer callable by attempting to create a CR with version = v1beta1.
// 5. Remove v1beta1 from crd.Status.StoredVersions.
// 6. Remove v1beta1 from crd.Spec.Versions and set crd.Spec.Version as v1beta2.
func TestConvertVersion(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	// create CRD with version = v1beta1
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	// ensure v1beta1 appears in discovery
	group := "mygroup.example.com"
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		_, err = apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta1")
		if err == nil {
			return true, nil
		}
		if errors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}); err != nil {
		t.Fatalf("failed to see v1beta1 in discovery: %v", err)
	}

	// create a CR and make sure it's version is v1beta1
	ns := "not-the-default"
	noxuV1beta1ResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)
	noxuInstanceToCreate := testserver.NewNoxuInstance(ns, "foo")
	createdFooInstance, err := noxuV1beta1ResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}
	if createdFooInstance.GetAPIVersion() != "mygroup.example.com/v1beta1" {
		t.Fatalf("invalid version: expected \"mygroup.example.com/v1beta1\", got %v", createdFooInstance.GetAPIVersion())
	}

	// add v1beta2 with served = true and storage = false
	gottenCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get CRD: %v", err)
	}
	v1beta2 := apiextensionsv1beta1.CustomResourceDefinitionVersion{
		Name:    "v1beta2",
		Served:  true,
		Storage: false,
	}
	gottenCRD.Spec.Versions = append(gottenCRD.Spec.Versions, v1beta2)
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		updatedCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD)
		if err != nil {
			return false, err
		}

		if len(updatedCRD.Spec.Versions) == 2 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("could not update CRD: %v", err)
	}

	// ensure both v1beta1 and v1beta2 appear in discovery
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		_, err1 := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta1")
		_, err2 := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta2")
		if err1 == nil && err2 == nil {
			return true, nil
		}
		if errors.IsNotFound(err1) || errors.IsNotFound(err2) {
			return false, nil
		}
		return false, fmt.Errorf("failed to see both v1beta1 and v1beta2 in discovery")
	}); err != nil {
		t.Fatalf("failed to see both v1beta1 and v1beta2 in discovery: %v", err)
	}

	// ensure that both v1beta1 and v1beta2 are usable by
	// ensuring that the old CR still has version = v1beta1
	// and a new CR can be created with version = v1beta2
	gottenFooInstance, err := noxuV1beta1ResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if gottenFooInstance.GetAPIVersion() != "mygroup.example.com/v1beta1" {
		t.Fatalf("invalid version: expected \"mygroup.example.com/v1beta1\", got %v", gottenFooInstance.GetAPIVersion())
	}

	noxuV1beta2ResourceClient := newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, "v1beta2")
	noxuInstanceToCreate = testserver.NewVersionedNoxuInstance(ns, "bar", "v1beta2")
	createdBarInstance, err := noxuV1beta2ResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}
	if createdBarInstance.GetAPIVersion() != "mygroup.example.com/v1beta2" {
		t.Fatalf("invalid version: expected \"mygroup.example.com/v1beta2\", got %v", createdBarInstance.GetAPIVersion())
	}

	// set v1beta2 with storage = true
	gottenCRD, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get CRD: %v", err)
	}
	gottenCRD.Spec.Versions[0].Storage = false
	gottenCRD.Spec.Versions[1].Storage = true
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		updatedCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD)
		if err != nil {
			return false, err
		}

		if !updatedCRD.Spec.Versions[0].Storage && updatedCRD.Spec.Versions[1].Storage {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("could not update CRD: %v", err)
	}

	// double check that both v1beta1 and v1beta2 still appear in discovery
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		_, err1 := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta1")
		_, err2 := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta2")
		if err1 == nil && err2 == nil {
			return true, nil
		}
		if errors.IsNotFound(err1) || errors.IsNotFound(err2) {
			return false, nil
		}
		return false, fmt.Errorf("failed to see both v1beta1 and v1beta2 in discovery")
	}); err != nil {
		t.Fatalf("failed to see both v1beta1 and v1beta2 in discovery: %v", err)
	}

	// create a new CR and ensure it persists with version = v1beta2
	noxuInstanceToCreate = testserver.NewVersionedNoxuInstance(ns, "baz", "v1beta2")
	createdBazInstance, err := noxuV1beta2ResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}
	if createdBazInstance.GetAPIVersion() != "mygroup.example.com/v1beta2" {
		t.Fatalf("invalid version: expected \"mygroup.example.com/v1beta2\", got %v", createdBazInstance.GetAPIVersion())
	}

	// update Foo from v1beta1 to v1beta2
	gottenFooInstance, err = noxuV1beta1ResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	gottenFooInstance.SetAPIVersion(group + "/v1beta2")
	updatedFooInstance, err := noxuV1beta2ResourceClient.Update(gottenFooInstance)
	if err != nil {
		t.Fatal(err)
	}
	if updatedFooInstance.GetAPIVersion() != "mygroup.example.com/v1beta2" {
		t.Fatalf("invalid version: expected \"mygroup.example.com/v1beta2\", got %v", updatedFooInstance.GetAPIVersion())
	}

	// set v1beta with served = false
	gottenCRD, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get CRD: %v", err)
	}
	gottenCRD.Spec.Versions[0].Served = false
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		updatedCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD)
		if err != nil {
			return false, err
		}

		if !updatedCRD.Spec.Versions[0].Served {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("could not update CRD: %v", err)
	}

	// ensure v1beta1 is removed from discovery
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		_, err = apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/v1beta1")
		if err == nil {
			return false, nil
		}
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	}); err != nil {
		t.Fatalf("failed to remove v1beta1 from discovery: %v", err)
	}

	// make sure the v1beta1 API is no longer callable by attempting to create a v1beta1 instance
	noxuInstanceToCreate = testserver.NewNoxuInstance(ns, "invalid-instance")
	_, err = noxuV1beta1ResourceClient.Create(noxuInstanceToCreate)
	if err == nil {
		t.Fatalf("instance should not be created: the v1beta1 API should no longer be available")
	}

	// remove v1beta1 from status.StoredVersions
	gottenCRD, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get CRD: %v", err)
	}
	gottenCRD.Status.StoredVersions = gottenCRD.Status.StoredVersions[1:]
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		updatedCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().UpdateStatus(gottenCRD)
		if err != nil {
			return false, err
		}

		if len(updatedCRD.Status.StoredVersions) == 1 && updatedCRD.Status.StoredVersions[0] == "v1beta2" {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("could not update CRD: %v", err)
	}

	// remove v1beta1 from spec.Versions
	gottenCRD, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxuDefinition.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get CRD: %v", err)
	}
	gottenCRD.Spec.Version = "v1beta2"
	gottenCRD.Spec.Versions = gottenCRD.Spec.Versions[1:]
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		updatedCRD, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Update(gottenCRD)
		if err != nil {
			return false, err
		}

		if len(updatedCRD.Spec.Versions) == 1 && updatedCRD.Spec.Version == "v1beta2" {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("could not update CRD: %v", err)
	}
}
