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

package persistentvolume

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
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

func makeTestPV(name string, ns *string) *api.PersistentVolume {
	store := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}

	if ns != nil {
		store.Namespace = *ns
	}
	return store
}

func TestRESTCreate(t *testing.T) {

	ns := "ns"

	table := []struct {
		ctx   api.Context
		store *api.PersistentVolume
		valid bool
	}{
		{
			ctx:   api.WithNamespace(api.NewContext(), "namespace-not-allowed"),
			store: makeTestPV("foo", &ns),
			valid: false,
		}, {
			ctx:   api.WithNamespaceDefaultIfNone(api.NewContext()),
			store: makeTestPV("baz", nil),
			valid: true,
		},
	}

	for _, item := range table {
		_, rest := NewTestREST()
		_, err := rest.Create(item.ctx, item.store)
		if !item.valid {
			if err == nil {
				t.Errorf("unexpected non-error for %v (%v, %v)", item.store.Name, item.ctx, item.store.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.store.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.store.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		// Ensure we implement the interface
		_ = apiserver.ResourceWatcher(rest)
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	volume := makeTestPV("foo", nil)
	_, err := rest.Create(api.NewDefaultContext(), volume)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	_, err = rest.Delete(api.NewDefaultContext(), volume.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	volume := makeTestPV("foo", nil)
	_, err := rest.Create(api.NewDefaultContext(), volume)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), volume.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := volume, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()
	volumeA := makeTestPV("foo", nil)
	volumeB := makeTestPV("bar", nil)
	volumeC := makeTestPV("baz", nil)

	volumeA.Labels = map[string]string{
		"a-label-key": "some value",
	}

	reg.ObjectList = &api.PersistentVolumeList{
		Items: []api.PersistentVolume{*volumeA, *volumeB, *volumeC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.PersistentVolumeList{
		Items: []api.PersistentVolume{*volumeA, *volumeB, *volumeC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	volumeA := makeTestPV("foo", nil)

	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, volumeA)
	}()
	got := <-wi.ResultChan()
	if e, a := volumeA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
