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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

func TestServerUp(t *testing.T) {
	tearDown, _, _, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()
}

func TestNamespaceScopedCRUD(t *testing.T) {
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

	testSimpleCRUD(t, ns, noxuDefinition, dynamicClient)
	testFieldSelector(t, ns, noxuDefinition, dynamicClient)
}

func TestClusterScopedCRUD(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	testSimpleCRUD(t, ns, noxuDefinition, dynamicClient)
	testFieldSelector(t, ns, noxuDefinition, dynamicClient)
}

func testSimpleCRUD(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, dynamicClient dynamic.Interface) {
	noxuResourceClients := map[string]dynamic.ResourceInterface{}
	noxuWatchs := map[string]watch.Interface{}
	disabledVersions := map[string]bool{}
	for _, v := range noxuDefinition.Spec.Versions {
		disabledVersions[v.Name] = !v.Served
	}
	for _, v := range noxuDefinition.Spec.Versions {
		noxuResourceClients[v.Name] = newNamespacedCustomResourceVersionedClient(ns, dynamicClient, noxuDefinition, v.Name)

		noxuWatch, err := noxuResourceClients[v.Name].Watch(metav1.ListOptions{})
		if disabledVersions[v.Name] {
			if !errors.IsNotFound(err) {
				t.Errorf("expected the watch operation fail with NotFound for disabled version %s, got error: %v", v.Name, err)
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
		createdNoxuInstance, err := instantiateVersionedCustomResource(t, fixtures.NewVersionedNoxuInstance(ns, "foo", version), noxuResourceClient, noxuDefinition, version)
		if disabledVersions[version] {
			if !errors.IsNotFound(err) {
				t.Errorf("expected the CR creation fail with NotFound for disabled version %s, got error: %v", version, err)
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

			if disabledVersions[version2] {
				if !errors.IsNotFound(err) {
					t.Errorf("expected the get operation fail with NotFound for disabled version %s, got error: %v", version2, err)

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
			if disabledVersions[version2] {
				if !errors.IsNotFound(err) {
					t.Errorf("expected the list operation fail with NotFound for disabled version %s, got error: %v", version2, err)

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

		// Update test
		for version2, noxuResourceClient2 := range noxuResourceClients {
			var gottenNoxuInstance *unstructured.Unstructured
			if disabledVersions[version2] {
				gottenNoxuInstance = &unstructured.Unstructured{}
				gottenNoxuInstance.SetName("foo")
			} else {
				gottenNoxuInstance, err = noxuResourceClient2.Get("foo", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			gottenNoxuInstance.Object["updated"] = version2
			updatedNoxuInstance, err := noxuResourceClient2.Update(gottenNoxuInstance, metav1.UpdateOptions{})
			if disabledVersions[version2] {
				if !errors.IsNotFound(err) {
					t.Errorf("expected the update operation fail with NotFound for disabled version %s, got error: %v", version2, err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}

				if updated, ok := updatedNoxuInstance.Object["updated"]; !ok {
					t.Errorf("expected string 'updated' field")
				} else if updated, ok := updated.(string); !ok || updated != version2 {
					t.Errorf("expected string 'updated' field to equal %q, got %q of type %T", version2, updated, updated)
				}

				if e, a := version2, updatedNoxuInstance.GroupVersionKind().Version; !reflect.DeepEqual(e, a) {
					t.Errorf("expected %v, got %v", e, a)
				}

				for _, noxuWatch := range noxuWatchs {
					select {
					case watchEvent := <-noxuWatch.ResultChan():
						eventMetadata, err := meta.Accessor(watchEvent.Object)
						if err != nil {
							t.Fatal(err)
						}

						if watchEvent.Type != watch.Modified {
							t.Errorf("expected modified event, got %v", watchEvent.Type)
							break
						}

						// it should have a UUID
						createdMetadata, err := meta.Accessor(createdNoxuInstance)
						if err != nil {
							t.Fatal(err)
						}
						if e, a := createdMetadata.GetUID(), eventMetadata.GetUID(); e != a {
							t.Errorf("expected equal UID for (expected) %v, and (actual) %v", createdNoxuInstance, watchEvent.Object)
						}

					case <-time.After(5 * time.Second):
						t.Errorf("missing watch event")
					}
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
				eventMetadata, err := meta.Accessor(watchEvent.Object)
				if err != nil {
					t.Fatal(err)
				}

				if watchEvent.Type != watch.Deleted {
					t.Errorf("expected delete event, got %v", watchEvent.Type)
					break
				}

				// it should have a UUID
				createdMetadata, err := meta.Accessor(createdNoxuInstance)
				if err != nil {
					t.Fatal(err)
				}
				if e, a := createdMetadata.GetUID(), eventMetadata.GetUID(); e != a {
					t.Errorf("expected equal UID for (expected) %v, and (actual) %v", createdNoxuInstance, watchEvent.Object)
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

func TestInvalidCRUD(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	noxuResourceClients := map[string]dynamic.ResourceInterface{}
	noxuWatchs := map[string]watch.Interface{}
	disabledVersions := map[string]bool{}
	for _, v := range noxuDefinition.Spec.Versions {
		disabledVersions[v.Name] = !v.Served
	}
	for _, v := range noxuDefinition.Spec.Versions {
		noxuResourceClients[v.Name] = newNamespacedCustomResourceVersionedClient("", dynamicClient, noxuDefinition, v.Name)

		noxuWatch, err := noxuResourceClients[v.Name].Watch(metav1.ListOptions{})
		if disabledVersions[v.Name] {
			if !errors.IsNotFound(err) {
				t.Errorf("expected the watch operation fail with NotFound for disabled version %s, got error: %v", v.Name, err)
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
		// Case when typeless Unstructured object is passed
		typelessInstance := &unstructured.Unstructured{}
		if _, err := noxuResourceClient.Create(typelessInstance, metav1.CreateOptions{}); !errors.IsBadRequest(err) {
			t.Errorf("expected badrequest for submitting empty object, got %#v", err)
		}
		// Case when apiVersion and Kind would be set up from GVK, but no other objects are present
		typedNoBodyInstance := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "mygroup.example.com/" + version,
				"kind":       "WishIHadChosenNoxu",
			},
		}
		if _, err := noxuResourceClient.Create(typedNoBodyInstance, metav1.CreateOptions{}); !errors.IsInvalid(err) {
			t.Errorf("expected invalid request for submitting malformed object, got %#v", err)
		}
	}
}

func testFieldSelector(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, dynamicClient dynamic.Interface) {
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)
	initialList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(initialList.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	initialListTypeMeta, err := meta.TypeAccessor(initialList)
	if err != nil {
		t.Fatal(err)
	}
	if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Versions[0].Name, initialListTypeMeta.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := noxuDefinition.Spec.Names.ListKind, initialListTypeMeta.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	initialListListMeta, err := meta.ListAccessor(initialList)
	if err != nil {
		t.Fatal(err)
	}
	noxuWatch, err := noxuResourceClient.Watch(
		metav1.ListOptions{
			ResourceVersion: initialListListMeta.GetResourceVersion(),
			FieldSelector:   "metadata.name=foo",
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer noxuWatch.Stop()

	_, err = instantiateCustomResource(t, fixtures.NewNoxuInstance(ns, "bar"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	createdNoxuInstanceFoo, err := instantiateCustomResource(t, fixtures.NewNoxuInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}

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
		if e, a := "foo", createdObjectMeta.GetName(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		createdTypeMeta, err := meta.TypeAccessor(watchEvent.Object)
		if err != nil {
			t.Fatal(err)
		}
		if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Versions[0].Name, createdTypeMeta.GetAPIVersion(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		if e, a := noxuDefinition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

	case <-time.After(5 * time.Second):
		t.Errorf("missing watch event")
	}

	gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdNoxuInstanceFoo, gottenNoxuInstance; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	listWithItem, err := noxuResourceClient.List(metav1.ListOptions{FieldSelector: "metadata.name=foo"})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 1, len(listWithItem.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := *createdNoxuInstanceFoo, listWithItem.Items[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := noxuResourceClient.Delete("bar", nil); err != nil {
		t.Fatal(err)
	}
	if err := noxuResourceClient.Delete("foo", nil); err != nil {
		t.Fatal(err)
	}

	listWithoutItem, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(listWithoutItem.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

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
		createdObjectMeta, err := meta.Accessor(createdNoxuInstanceFoo)
		if err != nil {
			t.Fatal(err)
		}
		if e, a := createdObjectMeta.GetUID(), deletedObjectMeta.GetUID(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		if e, a := ns, createdObjectMeta.GetNamespace(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		if e, a := "foo", createdObjectMeta.GetName(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

	case <-time.After(5 * time.Second):
		t.Errorf("missing watch event")
	}
}

func TestDiscovery(t *testing.T) {
	group := "mygroup.example.com"
	version := "v1beta1"

	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	scope := apiextensionsv1beta1.NamespaceScoped
	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(scope)
	if _, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient); err != nil {
		t.Fatal(err)
	}

	// check whether it shows up in discovery properly
	resources, err := apiExtensionClient.Discovery().ServerResourcesForGroupVersion(group + "/" + version)
	if err != nil {
		t.Fatal(err)
	}

	if len(resources.APIResources) != 1 {
		t.Fatalf("Expected exactly the resource \"noxus\" in group version %v/%v via discovery, got: %v", group, version, resources.APIResources)
	}

	r := resources.APIResources[0]
	if r.Name != "noxus" {
		t.Fatalf("Expected exactly the resource \"noxus\" in group version %v/%v via discovery, got: %v", group, version, r.Name)
	}
	if r.Kind != "WishIHadChosenNoxu" {
		t.Fatalf("Expected exactly the kind \"WishIHadChosenNoxu\" in group version %v/%v via discovery, got: %v", group, version, r.Kind)
	}

	s := []string{"foo", "bar", "abc", "def"}
	if !reflect.DeepEqual(r.ShortNames, s) {
		t.Fatalf("Expected exactly the shortnames `foo, bar, abc, def` in group version %v/%v via discovery, got: %v", group, version, r.ShortNames)
	}

	sort.Strings(r.Verbs)
	expectedVerbs := []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"}
	if !reflect.DeepEqual([]string(r.Verbs), expectedVerbs) {
		t.Fatalf("Unexpected verbs for resource \"noxus\" in group version %v/%v via discovery: expected=%v got=%v", group, version, expectedVerbs, r.Verbs)
	}

	if !reflect.DeepEqual(r.Categories, []string{"all"}) {
		t.Fatalf("Expected exactly the category \"all\" in group version %v/%v via discovery, got: %v", group, version, r.Categories)
	}
}

func TestNoNamespaceReject(t *testing.T) {
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

	ns := ""
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)
	initialList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(initialList.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	initialListTypeMeta, err := meta.TypeAccessor(initialList)
	if err != nil {
		t.Fatal(err)
	}
	if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Version, initialListTypeMeta.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := noxuDefinition.Spec.Names.ListKind, initialListTypeMeta.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	createdNoxuInstance, err := instantiateCustomResource(t, fixtures.NewNoxuInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err == nil {
		t.Fatalf("unexpected non-error: an empty namespace may not be set during creation while creating noxu instance: %v ", createdNoxuInstance)
	}
}

func TestSameNameDiffNamespace(t *testing.T) {
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

	ns1 := "namespace-1"
	testSimpleCRUD(t, ns1, noxuDefinition, dynamicClient)
	ns2 := "namespace-2"
	testSimpleCRUD(t, ns2, noxuDefinition, dynamicClient)

}

func TestSelfLink(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	// namespace scoped
	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	noxuInstanceToCreate := fixtures.NewNoxuInstance(ns, "foo")
	createdNoxuInstance, err := noxuNamespacedResourceClient.Create(noxuInstanceToCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if e, a := "/apis/mygroup.example.com/v1beta1/namespaces/not-the-default/noxus/foo", createdNoxuInstance.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// cluster scoped
	curletDefinition := fixtures.NewCurletCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	curletDefinition, err = fixtures.CreateNewCustomResourceDefinition(curletDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	curletResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, curletDefinition)

	curletInstanceToCreate := fixtures.NewCurletInstance(ns, "foo")
	createdCurletInstance, err := curletResourceClient.Create(curletInstanceToCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if e, a := "/apis/mygroup.example.com/v1beta1/curlets/foo", createdCurletInstance.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestPreserveInt(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	noxuInstanceToCreate := fixtures.NewNoxuInstance(ns, "foo")
	createdNoxuInstance, err := noxuNamespacedResourceClient.Create(noxuInstanceToCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	originalJSON, err := runtime.Encode(unstructured.UnstructuredJSONScheme, createdNoxuInstance)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gottenNoxuInstance, err := runtime.Decode(unstructured.UnstructuredJSONScheme, originalJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check if int is preserved.
	unstructuredObj := gottenNoxuInstance.(*unstructured.Unstructured).Object
	num := unstructuredObj["num"].(map[string]interface{})
	num1 := num["num1"].(int64)
	num2 := num["num2"].(int64)
	if num1 != 9223372036854775807 || num2 != 1000000 {
		t.Errorf("Expected %v, got %v, %v", `9223372036854775807, 1000000`, num1, num2)
	}
}

func TestPatch(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)

	t.Logf("Creating foo")
	noxuInstanceToCreate := fixtures.NewNoxuInstance(ns, "foo")
	_, err = noxuNamespacedResourceClient.Create(noxuInstanceToCreate, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Patching .num.num2 to 999")
	patch := []byte(`{"num": {"num2":999}}`)
	patchedNoxuInstance, err := noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "num", "num2")
	rv, found, err := unstructured.NestedString(patchedNoxuInstance.UnstructuredContent(), "metadata", "resourceVersion")
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatalf("metadata.resourceVersion not found")
	}

	// a patch with no change
	t.Logf("Patching .num.num2 again to 999")
	patchedNoxuInstance, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, patch, metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// make sure no-op patch does not increment resourceVersion
	expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "num", "num2")
	expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

	// an empty patch
	t.Logf("Applying empty patch")
	patchedNoxuInstance, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`), metav1.PatchOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// an empty patch is a no-op patch. make sure it does not increment resourceVersion
	expectInt64(t, patchedNoxuInstance.UnstructuredContent(), 999, "num", "num2")
	expectString(t, patchedNoxuInstance.UnstructuredContent(), rv, "metadata", "resourceVersion")

	originalJSON, err := runtime.Encode(unstructured.UnstructuredJSONScheme, patchedNoxuInstance)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	gottenNoxuInstance, err := runtime.Decode(unstructured.UnstructuredJSONScheme, originalJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check if int is preserved.
	unstructuredObj := gottenNoxuInstance.(*unstructured.Unstructured).Object
	num := unstructuredObj["num"].(map[string]interface{})
	num1 := num["num1"].(int64)
	num2 := num["num2"].(int64)
	if num1 != 9223372036854775807 || num2 != 999 {
		t.Errorf("Expected %v, got %v, %v", `9223372036854775807, 999`, num1, num2)
	}
}

func TestCrossNamespaceListWatch(t *testing.T) {
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

	ns := ""
	noxuResourceClient := newNamespacedCustomResourceClient(ns, dynamicClient, noxuDefinition)
	initialList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(initialList.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	initialListListMeta, err := meta.ListAccessor(initialList)
	if err != nil {
		t.Fatal(err)
	}

	noxuWatch, err := noxuResourceClient.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatal(err)
	}
	defer noxuWatch.Stop()

	instances := make(map[string]*unstructured.Unstructured)
	ns1 := "namespace-1"
	noxuNamespacedResourceClient1 := newNamespacedCustomResourceClient(ns1, dynamicClient, noxuDefinition)
	instances[ns1] = createInstanceWithNamespaceHelper(t, ns1, "foo1", noxuNamespacedResourceClient1, noxuDefinition)
	noxuNamespacesWatch1, err := noxuNamespacedResourceClient1.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatalf("Failed to watch namespace: %s, error: %v", ns1, err)
	}
	defer noxuNamespacesWatch1.Stop()

	ns2 := "namespace-2"
	noxuNamespacedResourceClient2 := newNamespacedCustomResourceClient(ns2, dynamicClient, noxuDefinition)
	instances[ns2] = createInstanceWithNamespaceHelper(t, ns2, "foo2", noxuNamespacedResourceClient2, noxuDefinition)
	noxuNamespacesWatch2, err := noxuNamespacedResourceClient2.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatalf("Failed to watch namespace: %s, error: %v", ns2, err)
	}
	defer noxuNamespacesWatch2.Stop()

	createdList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if e, a := 2, len(createdList.Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for _, a := range createdList.Items {
		if e := instances[a.GetNamespace()]; !reflect.DeepEqual(e, &a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}

	addEvents := 0
	for addEvents < 2 {
		select {
		case watchEvent := <-noxuWatch.ResultChan():
			if e, a := watch.Added, watchEvent.Type; e != a {
				t.Fatalf("expected %v, got %v", e, a)
			}
			createdObjectMeta, err := meta.Accessor(watchEvent.Object)
			if err != nil {
				t.Fatal(err)
			}
			if len(createdObjectMeta.GetUID()) == 0 {
				t.Errorf("missing uuid: %#v", watchEvent.Object)
			}
			createdTypeMeta, err := meta.TypeAccessor(watchEvent.Object)
			if err != nil {
				t.Fatal(err)
			}
			if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Version, createdTypeMeta.GetAPIVersion(); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := noxuDefinition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			delete(instances, createdObjectMeta.GetNamespace())
			addEvents++
		case <-time.After(5 * time.Second):
			t.Fatalf("missing watch event")
		}
	}
	if e, a := 0, len(instances); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	checkNamespacesWatchHelper(t, ns1, noxuNamespacesWatch1)
	checkNamespacesWatchHelper(t, ns2, noxuNamespacesWatch2)
}

func createInstanceWithNamespaceHelper(t *testing.T, ns string, name string, noxuNamespacedResourceClient dynamic.ResourceInterface, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition) *unstructured.Unstructured {
	createdInstance, err := instantiateCustomResource(t, fixtures.NewNoxuInstance(ns, name), noxuNamespacedResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	return createdInstance
}

func checkNamespacesWatchHelper(t *testing.T, ns string, namespacedwatch watch.Interface) {
	namespacedAddEvent := 0
	for namespacedAddEvent < 2 {
		select {
		case watchEvent := <-namespacedwatch.ResultChan():
			// Check that the namespaced watch only has one result
			if namespacedAddEvent > 0 {
				t.Fatalf("extra watch event")
			}
			if e, a := watch.Added, watchEvent.Type; e != a {
				t.Fatalf("expected %v, got %v", e, a)
			}
			createdObjectMeta, err := meta.Accessor(watchEvent.Object)
			if err != nil {
				t.Fatal(err)
			}
			if e, a := ns, createdObjectMeta.GetNamespace(); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
		case <-time.After(5 * time.Second):
			if namespacedAddEvent != 1 {
				t.Fatalf("missing watch event")
			}
		}
		namespacedAddEvent++
	}
}

func TestNameConflict(t *testing.T) {
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

	noxu2Definition := fixtures.NewNoxu2CustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Create(noxu2Definition)
	if err != nil {
		t.Fatal(err)
	}

	// A NameConflict occurs
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxu2Definition.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		for _, condition := range crd.Status.Conditions {
			if condition.Type == apiextensionsv1beta1.NamesAccepted && condition.Status == apiextensionsv1beta1.ConditionFalse {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	err = fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient)
	if err != nil {
		t.Fatal(err)
	}

	// Names are now accepted
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(noxu2Definition.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		for _, condition := range crd.Status.Conditions {
			if condition.Type == apiextensionsv1beta1.NamesAccepted && condition.Status == apiextensionsv1beta1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestStatusGetAndPatch(t *testing.T) {
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

	// make sure we don't get 405 Method Not Allowed from Getting CRD/status subresource
	result := &apiextensionsv1beta1.CustomResourceDefinition{}
	err = apiExtensionClient.ApiextensionsV1beta1().RESTClient().Get().
		Resource("customresourcedefinitions").
		Name(noxuDefinition.Name).
		SubResource("status").
		Do(context.TODO()).
		Into(result)
	if err != nil {
		t.Fatal(err)
	}

	// make sure we don't get 405 Method Not Allowed from Patching CRD/status subresource
	_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().
		Patch(noxuDefinition.Name, types.StrategicMergePatchType,
			[]byte(fmt.Sprintf(`{"labels":{"test-label":"dummy"}}`)),
			"status")
	if err != nil {
		t.Fatal(err)
	}
}

func expectInt64(t *testing.T, obj map[string]interface{}, value int64, pth ...string) {
	if v, found, err := unstructured.NestedInt64(obj, pth...); err != nil {
		t.Fatalf("failed to access .%s: %v", strings.Join(pth, "."), err)
	} else if !found {
		t.Fatalf("failed to find .%s", strings.Join(pth, "."))
	} else if v != value {
		t.Fatalf("wanted %d at .%s, got %d", value, strings.Join(pth, "."), v)
	}
}
func expectString(t *testing.T, obj map[string]interface{}, value string, pth ...string) {
	if v, found, err := unstructured.NestedString(obj, pth...); err != nil {
		t.Fatalf("failed to access .%s: %v", strings.Join(pth, "."), err)
	} else if !found {
		t.Fatalf("failed to find .%s", strings.Join(pth, "."))
	} else if v != value {
		t.Fatalf("wanted %q at .%s, got %q", value, strings.Join(pth, "."), v)
	}
}
