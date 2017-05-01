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
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/watch"
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

func TestSimpleCRUD(t *testing.T) {
	stopCh, apiExtensionClient, clientPool, err := testserver.StartDefaultServer()
	if err != nil {
		t.Fatal(err)
	}
	defer close(stopCh)

	noxuDefinition := testserver.NewNoxuCustomResourceDefinition()
	noxuVersionClient, err := testserver.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, clientPool)
	if err != nil {
		t.Fatal(err)
	}

	ns := "not-the-default"
	noxuNamespacedResourceClient := noxuVersionClient.Resource(&metav1.APIResource{
		Name:       noxuDefinition.Spec.Names.Plural,
		Namespaced: noxuDefinition.Spec.Scope == apiextensionsv1alpha1.NamespaceScoped,
	}, ns)
	initialList, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
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
	noxuNamespacedWatch, err := noxuNamespacedResourceClient.Watch(metav1.ListOptions{ResourceVersion: initialListListMeta.GetResourceVersion()})
	if err != nil {
		t.Fatal(err)
	}
	defer noxuNamespacedWatch.Stop()

	noxuInstanceToCreate := testserver.NewNoxuInstance(ns, "foo")
	createdNoxuInstance, err := noxuNamespacedResourceClient.Create(noxuInstanceToCreate)
	if err != nil {
		t.Logf("%#v", createdNoxuInstance)
		t.Fatal(err)
	}
	createdObjectMeta, err := meta.Accessor(createdNoxuInstance)
	if err != nil {
		t.Fatal(err)
	}
	// it should have a UUID
	if len(createdObjectMeta.GetUID()) == 0 {
		t.Errorf("missing uuid: %#v", createdNoxuInstance)
	}
	createdTypeMeta, err := meta.TypeAccessor(createdNoxuInstance)
	if err != nil {
		t.Fatal(err)
	}
	if e, a := noxuDefinition.Spec.Group+"/"+noxuDefinition.Spec.Version, createdTypeMeta.GetAPIVersion(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := noxuDefinition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	select {
	case watchEvent := <-noxuNamespacedWatch.ResultChan():
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

	gottenNoxuInstance, err := noxuNamespacedResourceClient.Get("foo")
	if err != nil {
		t.Fatal(err)
	}
	if e, a := createdNoxuInstance, gottenNoxuInstance; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	listWithItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 1, len(listWithItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := *createdNoxuInstance, listWithItem.(*unstructured.UnstructuredList).Items[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}

	if err := noxuNamespacedResourceClient.Delete("foo", nil); err != nil {
		t.Fatal(err)
	}

	listWithoutItem, err := noxuNamespacedResourceClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if e, a := 0, len(listWithoutItem.(*unstructured.UnstructuredList).Items); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	select {
	case watchEvent := <-noxuNamespacedWatch.ResultChan():
		if e, a := watch.Deleted, watchEvent.Type; e != a {
			t.Errorf("expected %v, got %v", e, a)
			break
		}
		deletedObjectMeta, err := meta.Accessor(watchEvent.Object)
		if err != nil {
			t.Fatal(err)
		}
		// it should have a UUID
		if e, a := createdObjectMeta.GetUID(), deletedObjectMeta.GetUID(); e != a {
			t.Errorf("expected %v, got %v", e, a)
		}

	case <-time.After(5 * time.Second):
		t.Errorf("missing watch event")
	}
}
