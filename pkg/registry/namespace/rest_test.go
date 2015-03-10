/*
Copyright 2014 Google Inc. All rights reserved.

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

package namespace

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type testRegistry struct {
	*registrytest.GenericRegistry
}

func NewTestREST() (testRegistry, *REST) {
	reg := testRegistry{registrytest.NewGeneric(nil)}
	return reg, NewREST(reg)
}

func testNamespace(name string) *api.Namespace {
	return &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx       api.Context
		namespace *api.Namespace
		valid     bool
	}{
		{
			ctx:       api.NewContext(),
			namespace: testNamespace("foo"),
			valid:     true,
		}, {
			ctx:       api.NewContext(),
			namespace: testNamespace("bar"),
			valid:     true,
		},
	}

	for _, item := range table {
		_, rest := NewTestREST()
		c, err := rest.Create(item.ctx, item.namespace)
		if !item.valid {
			if err == nil {
				t.Errorf("unexpected non-error for %v", item.namespace.Name)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.namespace.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.namespace.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if e, a := item.namespace, c; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
		// Ensure we implement the interface
		_ = apiserver.ResourceWatcher(rest)
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := NewTestREST()
	namespaceA := testNamespace("foo")
	_, err := rest.Create(api.NewDefaultContext(), namespaceA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), namespaceA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := namespaceA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	namespaceB := testNamespace("foo")
	_, _, err = rest.Update(api.NewDefaultContext(), namespaceB)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got2, err := rest.Get(api.NewDefaultContext(), namespaceB.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := namespaceB, got2; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}

}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	namespaceA := testNamespace("foo")
	_, err := rest.Create(api.NewContext(), namespaceA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	c, err := rest.Delete(api.NewContext(), namespaceA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := c.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	namespaceA := testNamespace("foo")
	_, err := rest.Create(api.NewContext(), namespaceA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewContext(), namespaceA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := namespaceA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()
	namespaceA := testNamespace("foo")
	namespaceB := testNamespace("bar")
	namespaceC := testNamespace("baz")
	reg.ObjectList = &api.NamespaceList{
		Items: []api.Namespace{*namespaceA, *namespaceB, *namespaceC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.NamespaceList{
		Items: []api.Namespace{*namespaceA, *namespaceB, *namespaceC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	namespaceA := testNamespace("foo")
	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, namespaceA)
	}()
	got := <-wi.ResultChan()
	if e, a := namespaceA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
