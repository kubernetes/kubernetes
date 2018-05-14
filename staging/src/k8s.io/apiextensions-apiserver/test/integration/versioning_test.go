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
	"reflect"
	"testing"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

func TestVersionedNamspacedScopedCRD(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.NamespaceScoped)
	err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	testSimpleVersionedCRUD(t, ns, noxuDefinition, dynamicClient)
}

func TestVersionedClusterScopedCRD(t *testing.T) {
	stopCh, apiExtensionClient, dynamicClient, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewMultipleVersionNoxuCRD(apiextensionsv1beta1.ClusterScoped)
	err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	testSimpleVersionedCRUD(t, ns, noxuDefinition, dynamicClient)
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
	err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	// The storage version list should be initilized to storage version
	crd, err := testserver.GetCustomResourceDefinition(noxuDefinition, apiExtensionClient)
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
	crd, err = testserver.GetCustomResourceDefinition(noxuDefinition, apiExtensionClient)
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

func testSimpleVersionedCRUD(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, dynamicClient dynamic.Interface) {
	noxuResourceClients := map[string]dynamic.ResourceInterface{}
	noxuWatchs := map[string]watch.Interface{}
	disbaledVersions := map[string]bool{}
	for _, v := range noxuDefinition.Spec.Versions {
		disbaledVersions[v.Name] = !v.Served
	}
	for _, v := range noxuDefinition.Spec.Versions {
		noxuResourceClients[v.Name] = NewNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)

		noxuWatch, err := noxuResourceClients[v.Name].Watch(metav1.ListOptions{})
		if disbaledVersions[v.Name] {
			if err == nil {
				t.Errorf("expected the watch creation fail for disabled version %s", v.Name)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			noxuWatchs[v.Name] = noxuWatch
		}
	}
	defer func() {
		for _, w := range noxuWatchs {
			w.Stop()
		}
	}()

	for version, noxuResourceClient := range noxuResourceClients {
		createdNoxuInstance, err := instantiateVersionedCustomResource(t, testserver.NewVersionedNoxuInstance(ns, "foo", version), noxuResourceClient, noxuDefinition, version)
		if disbaledVersions[version] {
			if err == nil {
				t.Errorf("expected the CR creation fail for disabled version %s", version)
			}
			continue
		}
		if err != nil {
			t.Fatalf("unable to create noxu Instance:%v", err)
		}
		if e, a := noxuDefinition.Spec.Group+"/"+version, createdNoxuInstance.GetAPIVersion(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		for watchVersion, noxuWatch := range noxuWatchs {
			select {
			case watchEvent := <-noxuWatch.ResultChan():
				if e, a := watch.Added, watchEvent.Type; e != a {
					t.Errorf("expected %v, got %v", e, a)
					break
				}
				createdObjectMeta, err := meta.Accessor(watchEvent.Object)
				if err != nil {
					t.Fatal(err)
				}
				// it should have a UUID
				if len(createdObjectMeta.GetUID()) == 0 {
					t.Errorf("missing uuid: %#v", watchEvent.Object)
				}
				if e, a := ns, createdObjectMeta.GetNamespace(); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				createdTypeMeta, err := meta.TypeAccessor(watchEvent.Object)
				if err != nil {
					t.Fatal(err)
				}
				if e, a := noxuDefinition.Spec.Group+"/"+watchVersion, createdTypeMeta.GetAPIVersion(); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				if e, a := noxuDefinition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
			case <-time.After(5 * time.Second):
				t.Errorf("missing watch event")
			}
		}

		// Check get for all versions
		for version2, noxuResourceClient2 := range noxuResourceClients {
			// Get test
			gottenNoxuInstance, err := noxuResourceClient2.Get("foo", metav1.GetOptions{})

			if disbaledVersions[version2] {
				if err == nil {
					t.Errorf("expected the get operation fail for disabled version %s", version2)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}

				if e, a := version2, gottenNoxuInstance.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
					t.Errorf("expected %v, got %v", e, a)
				}
			}

			// List test
			listWithItem, err := noxuResourceClient2.List(metav1.ListOptions{})
			if disbaledVersions[version2] {
				if err == nil {
					t.Errorf("expected the list operation fail for disabled version %s", version2)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if e, a := 1, len(listWithItem.Items); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				if e, a := version2, listWithItem.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
					t.Errorf("expected %v, got %v", e, a)
				}
				if e, a := version2, listWithItem.Items[0].GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
					t.Errorf("expected %v, got %v", e, a)
				}
			}
		}

		// Delete test
		if err := noxuResourceClient.Delete("foo", metav1.NewDeleteOptions(0)); err != nil {
			t.Fatal(err)
		}

		listWithoutItem, err := noxuResourceClient.List(metav1.ListOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if e, a := 0, len(listWithoutItem.Items); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

		for _, noxuWatch := range noxuWatchs {
			select {
			case watchEvent := <-noxuWatch.ResultChan():
				if e, a := watch.Deleted, watchEvent.Type; e != a {
					t.Errorf("expected %v, got %v", e, a)
					break
				}
				deletedObjectMeta, err := meta.Accessor(watchEvent.Object)
				if err != nil {
					t.Fatal(err)
				}
				// it should have a UUID
				createdObjectMeta, err := meta.Accessor(createdNoxuInstance)
				if err != nil {
					t.Fatal(err)
				}
				if e, a := createdObjectMeta.GetUID(), deletedObjectMeta.GetUID(); e != a {
					t.Errorf("expected %v, got %v", e, a)
				}

			case <-time.After(5 * time.Second):
				t.Errorf("missing watch event")
			}
		}

		// Delete test
		if err := noxuResourceClient.DeleteCollection(metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
			t.Fatal(err)
		}

	}
}
