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

package persistentvolumeclaim

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

func makeTestClaim(name string, ns string) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.AccessModeType{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx   api.Context
		claim *api.PersistentVolumeClaim
		valid bool
	}{
		{
			ctx:   api.WithNamespace(api.NewContext(), "foo"),
			claim: makeTestClaim("foo", "foo"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "bar"),
			claim: makeTestClaim("bar", "bar"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "not-baz"),
			claim: makeTestClaim("baz", "baz"),
			valid: false,
		},
	}

	for _, item := range table {
		_, rest := NewTestREST()
		_, err := rest.Create(item.ctx, item.claim)
		if !item.valid {
			if err == nil {
				t.Errorf("unexpected non-error for %v (%v, %v)", item.claim.Name, item.ctx, item.claim.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.claim.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.claim.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		// Ensure we implement the interface
		_ = apiserver.ResourceWatcher(rest)
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	claim := makeTestClaim("foo", api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), claim)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	_, err = rest.Delete(api.NewDefaultContext(), claim.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	claim := makeTestClaim("foo", api.NamespaceDefault)
	_, err := rest.Create(api.NewDefaultContext(), claim)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), claim.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := claim, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()
	claimA := makeTestClaim("foo", api.NamespaceDefault)
	claimB := makeTestClaim("bar", api.NamespaceDefault)
	claimC := makeTestClaim("baz", api.NamespaceDefault)

	claimA.Labels = map[string]string{
		"a-label-key": "some value",
	}

	reg.ObjectList = &api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{*claimA, *claimB, *claimC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{*claimA, *claimB, *claimC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	claimA := makeTestClaim("foo", api.NamespaceDefault)

	reg, rest := NewTestREST()
	_, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, claimA)
	}()
}
