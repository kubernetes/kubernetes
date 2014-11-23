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

package masterflags

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
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

func testMasterFlags(name string) *api.MasterFlags {
	return &api.MasterFlags{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: api.MasterFlagsSpec{
			CmdLineArg: map[string]string {"a": "b"},
		},
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx   api.Context
		masterFlags *api.MasterFlags
		valid bool
	}{
		{
			ctx:   api.NewDefaultContext(),
			masterFlags: testMasterFlags("foo"),
			valid: true,
		}, {
			ctx:   api.NewContext(),
			masterFlags: testMasterFlags("bar"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			masterFlags: testMasterFlags("bazzzz"),
			valid: false,
		},
	}

	for _, item := range table {
		_, rest := NewTestREST()
		c, err := rest.Create(item.ctx, item.masterFlags)
		if !item.valid {
			if err == nil {
				ctxNS := api.Namespace(item.ctx)
				t.Errorf("unexpected non-error for %v (%v, %v)", item.masterFlags.Name, ctxNS, item.masterFlags.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.masterFlags.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.masterFlags.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if e, a := item.masterFlags, (<-c).Object; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
		// Ensure we implement the interface
		_ = apiserver.ResourceWatcher(rest)
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := NewTestREST()
	masterFlagsA := testMasterFlags("foo")
	ctx := api.NewDefaultContext()
	c, err := rest.Create(ctx, masterFlagsA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	<-c
	masterFlagsB := testMasterFlags("bar")
	c2, err2 := rest.Update(ctx, masterFlagsB)
	if err2 != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, b := masterFlagsB, (<-c2).Object; !reflect.DeepEqual(e, b) {
		t.Errorf("diff: %s", util.ObjectDiff(e, b))
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	masterFlagsA := testMasterFlags("foo")
	c, err := rest.Create(api.NewDefaultContext(), masterFlagsA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	<-c
	c, err = rest.Delete(api.NewDefaultContext(), masterFlagsA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := (<-c).Object.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	masterFlagsA := testMasterFlags("foo")
	c, err := rest.Create(api.NewDefaultContext(), masterFlagsA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	<-c
	got, err := rest.Get(api.NewDefaultContext(), masterFlagsA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := masterFlagsA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()

	masterFlagsA := &api.MasterFlags{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: api.MasterFlagsSpec{
			CmdLineArg: map[string]string {"a": "b"},
		},
	}
	masterFlagsB := &api.MasterFlags{
		ObjectMeta: api.ObjectMeta{
			Name:      "bar",
			Namespace: "default",
		},
		Spec: api.MasterFlagsSpec{
			CmdLineArg: map[string]string {"c": "d"},
		},
	}
	masterFlagsC := &api.MasterFlags{
		ObjectMeta: api.ObjectMeta{
			Name:      "baz",
			Namespace: "default",
		},
		Spec: api.MasterFlagsSpec{
			CmdLineArg: map[string]string {"e": "f"},
		},
	}
	
	reg.ObjectList = &api.MasterFlagsList{
		Items: []api.MasterFlags{*masterFlagsA, *masterFlagsB, *masterFlagsC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), labels.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.MasterFlagsList{
		Items: []api.MasterFlags{*masterFlagsA, *masterFlagsB, *masterFlagsC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	masterFlagsA := &api.MasterFlags{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: api.MasterFlagsSpec{
			CmdLineArg: map[string]string {"a": "b"},
		},
	}
	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), labels.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Mux.Action(watch.Added, masterFlagsA)
	}()
	got := <-wi.ResultChan()
	if e, a := masterFlagsA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
