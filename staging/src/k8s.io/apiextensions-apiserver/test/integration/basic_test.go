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

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
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
	stopCh, _, _, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)
}

func TestNamespaceScopedCRUD(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	testSimpleCRUD(t, ns, noxuDefinition, noxuVersionClient)
	testFieldSelector(t, ns, noxuDefinition, noxuVersionClient)
}

func TestClusterScopedCRUD(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := ""
	testSimpleCRUD(t, ns, noxuDefinition, noxuVersionClient)
	testFieldSelector(t, ns, noxuDefinition, noxuVersionClient)
}

func testSimpleCRUD(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, noxuVersionClient dynamic.Interface) {
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

	gottenNoxuInstance, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
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

func testFieldSelector(t *testing.T, ns string, noxuDefinition *apiextensionsv1beta1.CustomResourceDefinition, noxuVersionClient dynamic.Interface) {
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

	_, err = instantiateCustomResource(t, testserver.NewNoxuInstance(ns, "bar"), noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unable to create noxu Instance:%v", err)
	}
	createdNoxuInstanceFoo, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, "foo"), noxuResourceClient, noxuDefinition)
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
		if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Version, createdTypeMeta.GetAPIVersion(); e != a {
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
	if e, a := 1, len(listWithItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := *createdNoxuInstanceFoo, listWithItem.(*unstructured.UnstructuredList).Items[0]; !reflect.DeepEqual(e, a) {
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

	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	scope := apiextensionsv1beta1.NamespaceScoped
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

	if !reflect.DeepEqual(r.Categories, []string{"all"}) {
		t.Fatalf("Expected exactly the category \"all\" in group version %v/%v via discovery, got: %v", group, version, r.Categories)
	}
}

func TestNoNamespaceReject(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
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
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
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
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	// namespace scoped
	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := noxuVersionClient.Resource(&metav1.APIResource{
		Name:       noxuDefinition.Spec.Names.Plural,
		Namespaced: noxuDefinition.Spec.Scope == apiextensionsv1beta1.NamespaceScoped,
	}, ns)

	noxuInstanceToCreate := testserver.NewNoxuInstance(ns, "foo")
	createdNoxuInstance, err := noxuNamespacedResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}

	if e, a := "/apis/mygroup.example.com/v1beta1/namespaces/not-the-default/noxus/foo", createdNoxuInstance.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// cluster scoped
	curletDefinition := testserver.NewCurletCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	curletVersionClient, err := testserver.CreateNewCustomResourceDefinition(curletDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	curletResourceClient := curletVersionClient.Resource(&metav1.APIResource{
		Name:       curletDefinition.Spec.Names.Plural,
		Namespaced: curletDefinition.Spec.Scope == apiextensionsv1beta1.NamespaceScoped,
	}, ns)

	curletInstanceToCreate := testserver.NewCurletInstance(ns, "foo")
	createdCurletInstance, err := curletResourceClient.Create(curletInstanceToCreate)
	if err != nil {
		t.Fatal(err)
	}

	if e, a := "/apis/mygroup.example.com/v1beta1/curlets/foo", createdCurletInstance.GetSelfLink(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestPreserveInt(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
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
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
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

	patch := []byte(`{"num": {"num2":999}}`)
	createdNoxuInstance, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// this call waits for the resourceVersion to be reached in the cache before returning.
	// We need to do this because the patch gets its initial object from the storage, and the cache serves that.
	// If it is out of date, then our initial patch is applied to an old resource version, which conflicts
	// and then the updated object shows a conflicting diff, which permanently fails the patch.
	// This gives expected stability in the patch without retrying on an known number of conflicts below in the test.
	// See https://issue.k8s.io/42644
	_, err = noxuNamespacedResourceClient.Get("foo", metav1.GetOptions{ResourceVersion: createdNoxuInstance.GetResourceVersion()})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// a patch with no change
	createdNoxuInstance, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, patch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// an empty patch
	createdNoxuInstance, err = noxuNamespacedResourceClient.Patch("foo", types.MergePatchType, []byte(`{}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
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
	if num1 != 9223372036854775807 || num2 != 999 {
		t.Errorf("Expected %v, got %v, %v", `9223372036854775807, 999`, num1, num2)
	}
}

func TestCrossNamespaceListWatch(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
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
	noxuNamespacedResourceClient1 := NewNamespacedCustomResourceClient(ns1, noxuVersionClient, noxuDefinition)
	instances[ns1] = createInstanceWithNamespaceHelper(t, ns1, "foo1", noxuNamespacedResourceClient1, noxuDefinition)
	noxuNamespacesWatch1, err := noxuNamespacedResourceClient1.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	defer noxuNamespacesWatch1.Stop()

	ns2 := "namespace-2"
	noxuNamespacedResourceClient2 := NewNamespacedCustomResourceClient(ns2, noxuVersionClient, noxuDefinition)
	instances[ns2] = createInstanceWithNamespaceHelper(t, ns2, "foo2", noxuNamespacedResourceClient2, noxuDefinition)
	noxuNamespacesWatch2, err := noxuNamespacedResourceClient2.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	defer noxuNamespacesWatch2.Stop()

	createdList, err := noxuResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if e, a := 2, len(createdList.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for _, a := range createdList.(*unstructured.UnstructuredList).Items {
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
	createdInstance, err := instantiateCustomResource(t, testserver.NewNoxuInstance(ns, name), noxuNamespacedResourceClient, noxuDefinition)
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
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServerWithClients()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	_, err = testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	noxu2Definition := testserver.NewNoxu2CustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	_, err = apiExtensionClient.Apiextensions().CustomResourceDefinitions().Create(noxu2Definition)
	if err != nil {
		t.Fatal(err)
	}

	// A NameConflict occurs
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := testserver.GetCustomResourceDefinition(noxu2Definition, apiExtensionClient)
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

	err = testserver.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient)
	if err != nil {
		t.Fatal(err)
	}

	// Names are now accepted
	err = wait.Poll(500*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		crd, err := testserver.GetCustomResourceDefinition(noxu2Definition, apiExtensionClient)
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
