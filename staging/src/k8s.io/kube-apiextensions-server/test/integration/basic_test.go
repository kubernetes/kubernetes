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
	"reflect"
	"sort"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	apiextensionsv1alpha1 "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1"
	"k8s.io/kube-apiextensions-server/test/integration/testserver"
)

func TestServerUp(t *testing.T) {
	stopCh, _, _, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)
}

func TestNamespaceScopedCRUD(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1alpha1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	testSimpleCRUD(t, ns, noxuDefinition, noxuVersionClient)
}

func TestClusterScopedCRUD(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1alpha1.ClusterScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	testSimpleCRUD(t, ns, noxuDefinition, noxuVersionClient)
}

func testSimpleCRUD(t *testing.T, ns string, noxuDefinition *apiextensionsv1alpha1.CustomResourceDefinition, noxuVersionClient *dynamic.Client) {
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	initialList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(initialList.(*unstructured.UnstructuredList).Items); e != a {
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

	initialListListMeta, err := meta.ListAccessor(initialList)
	if err != nil {
		t.Fatal(err)
	}
	noxuWatch, err := noxuResourceClient.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatal(err)
	}
	defer noxuWatch.Stop()

	createdNoxuInstance, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
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

	case <-time.After(5 * time.Second):
		t.Errorf("missing watch event")
	}

	gottenNoxuInstance, err := noxuResourceClient.Get("foo")
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdNoxuInstance, gottenNoxuInstance; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	listWithItem, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 1, len(listWithItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := *createdNoxuInstance, listWithItem.(*unstructured.UnstructuredList).Items[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := noxuResourceClient.Delete("foo", nil); err != nil {
		t.Fatal(err)
	}

	listWithoutItem, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(listWithoutItem.(*unstructured.UnstructuredList).Items); e != a {
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

func TestDiscovery(t *testing.T) {
	group := "mygroup.example.com"
	version := "v1alpha1"

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	scope := apiextensionsv1alpha1.NamespaceScoped
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(scope)
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
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
}

func TestNoNamespaceReject(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1alpha1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	noxuResourceClient := NewNamespacedCustomResourceClient(ns, noxuVersionClient, noxuDefinition)
	initialList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(initialList.(*unstructured.UnstructuredList).Items); e != a {
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

	createdNoxuInstance, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
	if err == nil {
		t.Fatalf("unexpected non-error: an empty namespace may not be set during creation while creating noxu instance: %v ", createdNoxuInstance)
	}
}

func TestSameNameDiffNamespace(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1alpha1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns1 := "namespace-1"
	testSimpleCRUD(t, ns1, noxuDefinition, noxuVersionClient)
	ns2 := "namespace-2"
	testSimpleCRUD(t, ns2, noxuDefinition, noxuVersionClient)

}

func TestSelfLink(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1alpha1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := noxuVersionClient.Resource(&metav1.APIResource{
		Name:       noxuDefinition.Spec.Names.Plural,
		Namespaced: true,
	}, ns)

	noxuInstanceToCreate := testserver.NewNoxuInstance(ns, "foo")
	createdNoxuInstance, err := noxuNamespacedResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}

	if e, a := "/apis/mygroup.example.com/v1alpha1/namespaces/not-the-default/noxus/foo", createdNoxuInstance.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// TODO add test for cluster scoped self-link when its available

}
