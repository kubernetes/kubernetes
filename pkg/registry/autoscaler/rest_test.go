/*
Copyright 2015 Google Inc. All rights reserved.

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

package autoscaler

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// testRegistry provides the REST implementation for unit tests
type testRegistry struct {
	*registrytest.GenericRegistry
}

func newTestREST() (testRegistry, *REST) {
	reg := testRegistry{registrytest.NewGeneric(nil)}
	return reg, NewREST(reg)
}

func newValidAutoScaler(name string) *api.AutoScaler {
	return &api.AutoScaler{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: api.AutoScalerSpec{
			TargetSelector:  map[string]string{"target": "selector"},
			MonitorSelector: map[string]string{"monitor": "selector"},
		},
	}
}

func TestRESTCreate(t *testing.T) {
	_, rest := newTestREST()

	testCases := []struct {
		name       string
		ctx        api.Context
		autoScaler *api.AutoScaler
		valid      bool
	}{
		{
			name:       "valid create",
			ctx:        api.NewDefaultContext(),
			autoScaler: newValidAutoScaler("valid"),
			valid:      true,
		},
		{
			name:       "empty context",
			ctx:        api.NewContext(),
			autoScaler: newValidAutoScaler("emptyContext"),
			valid:      false,
		},
		{
			name:       "non-default context",
			ctx:        api.WithNamespace(api.NewContext(), "nondefault"),
			autoScaler: newValidAutoScaler("nonDefault"),
			valid:      false,
		},
	}

	for _, tc := range testCases {
		obj, err := rest.Create(tc.ctx, tc.autoScaler)

		if tc.valid && err != nil {
			t.Errorf("test case %s failed: expected no errors but found: %v", tc.name, err)
			continue
		}

		if !tc.valid && err == nil {
			t.Errorf("test case %s failed: expected errors but found none", tc.name)
			continue
		}

		// at this point, if it is an invalid test case and it has errors we're good to go
		// the rest of the validations are for fields
		if !tc.valid {
			continue
		}

		if !api.HasObjectMetaSystemFieldValues(&tc.autoScaler.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}

		if e, a := tc.autoScaler, obj; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := newTestREST()
	autoScaler := newValidAutoScaler("foo")
	ctx := api.NewDefaultContext()

	_, err := rest.Create(ctx, autoScaler)
	if err != nil {
		t.Fatalf("unable to create auto-scaler: %v", err)
	}

	autoScaler.Spec.TargetSelector["target"] = "updated"
	updated, created, err := rest.Update(ctx, autoScaler)

	if err != nil {
		t.Fatalf("unable to create auto-scaler: %v", err)
	}
	if updated == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected created to be false")
	}

	updatedAutoScaler := updated.(*api.AutoScaler)
	if updatedAutoScaler.Spec.TargetSelector["target"] != "updated" {
		t.Errorf("expected target selector to be updated")
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := newTestREST()
	autoScaler := newValidAutoScaler("foo")
	_, err := rest.Create(api.NewDefaultContext(), autoScaler)
	if err != nil {
		t.Fatalf("unable to create auto-scaler: %v", err)
	}

	stat, err := rest.Delete(api.NewDefaultContext(), autoScaler.Name)
	if err != nil {
		t.Fatalf("unable to delete auto-scaler: %v", err)
	}
	if status := stat.(*api.Status); status.Status != api.StatusSuccess {
		t.Errorf("unexepcted status: %v", status)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := newTestREST()
	autoScaler := newValidAutoScaler("foo")
	_, err := rest.Create(api.NewDefaultContext(), autoScaler)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), autoScaler.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := autoScaler, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTgetAttrs(t *testing.T) {
	_, rest := newTestREST()
	autoScaler := newValidAutoScaler("foo")
	autoScaler.ObjectMeta.Labels = make(map[string]string, 1)
	autoScaler.ObjectMeta.Labels["foo"] = "bar"

	label, field, err := rest.getAttrs(autoScaler)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := label, labels.Set(autoScaler.ObjectMeta.Labels); !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	expect := fields.Set{}
	if e, a := expect, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := newTestREST()

	var (
		autoScalerA = newValidAutoScaler("a")
		autoScalerB = newValidAutoScaler("b")
		autoScalerC = newValidAutoScaler("c")
	)

	reg.ObjectList = &api.AutoScalerList{
		Items: []api.AutoScaler{*autoScalerA, *autoScalerB, *autoScalerC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.AutoScalerList{
		Items: []api.AutoScaler{*autoScalerA, *autoScalerB, *autoScalerC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	autoScaler := newValidAutoScaler("foo")
	reg, rest := newTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, autoScaler)
	}()
	got := <-wi.ResultChan()
	if e, a := autoScaler, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
