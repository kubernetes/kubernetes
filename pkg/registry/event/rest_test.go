/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package event

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

type testRegistry struct {
	*registrytest.GenericRegistry
}

func NewTestREST() (testRegistry, *REST) {
	reg := testRegistry{registrytest.NewGeneric(nil)}
	return reg, NewStorage(reg)
}

func testEvent(name string) *api.Event {
	return &api.Event{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		InvolvedObject: api.ObjectReference{
			Namespace: "default",
		},
		Reason: "forTesting",
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx   api.Context
		event *api.Event
		valid bool
	}{
		{
			ctx:   api.NewDefaultContext(),
			event: testEvent("foo"),
			valid: true,
		}, {
			ctx:   api.NewContext(),
			event: testEvent("bar"),
			valid: true,
		}, {
			ctx:   api.WithNamespace(api.NewContext(), "nondefault"),
			event: testEvent("bazzzz"),
			valid: false,
		},
	}

	for _, item := range table {
		_, storage := NewTestREST()
		c, err := storage.Create(item.ctx, item.event)
		if !item.valid {
			if err == nil {
				ctxNS := api.NamespaceValue(item.ctx)
				t.Errorf("unexpected non-error for %v (%v, %v)", item.event.Name, ctxNS, item.event.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error %v", item.event.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.event.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if e, a := item.event, c; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
		// Ensure we implement the interface
		_ = rest.Watcher(storage)
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := NewTestREST()
	eventA := testEvent("foo")
	_, err := rest.Create(api.NewDefaultContext(), eventA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), eventA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := eventA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	eventB := testEvent("bar")
	_, _, err = rest.Update(api.NewDefaultContext(), eventB)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got2, err := rest.Get(api.NewDefaultContext(), eventB.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := eventB, got2; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}

}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	eventA := testEvent("foo")
	_, err := rest.Create(api.NewDefaultContext(), eventA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	c, err := rest.Delete(api.NewDefaultContext(), eventA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := c.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	eventA := testEvent("foo")
	_, err := rest.Create(api.NewDefaultContext(), eventA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), eventA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := eventA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTgetAttrs(t *testing.T) {
	_, rest := NewTestREST()
	eventA := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "f0118"},
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      testapi.Version(),
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: api.EventSource{Component: "test"},
	}
	label, field, err := rest.getAttrs(eventA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := label, (labels.Set{}); !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	expect := fields.Set{
		"metadata.name":                  "f0118",
		"involvedObject.kind":            "Pod",
		"involvedObject.name":            "foo",
		"involvedObject.namespace":       "baz",
		"involvedObject.uid":             "long uid string",
		"involvedObject.apiVersion":      testapi.Version(),
		"involvedObject.resourceVersion": "0",
		"involvedObject.fieldPath":       "",
		"reason":                         "ForTesting",
		"source":                         "test",
	}
	if e, a := expect, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()
	eventA := &api.Event{
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			UID:             "long uid string",
			APIVersion:      testapi.Version(),
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: api.EventSource{Component: "GoodSource"},
	}
	eventB := &api.Event{
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "bar",
			UID:             "other long uid string",
			APIVersion:      testapi.Version(),
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: api.EventSource{Component: "GoodSource"},
	}
	eventC := &api.Event{
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "baz",
			UID:             "yet another long uid string",
			APIVersion:      testapi.Version(),
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: api.EventSource{Component: "OtherSource"},
	}
	reg.ObjectList = &api.EventList{
		Items: []api.Event{*eventA, *eventB, *eventC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Set{"source": "GoodSource"}.AsSelector())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.EventList{
		Items: []api.Event{*eventA, *eventB},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	eventA := &api.Event{
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			UID:             "long uid string",
			APIVersion:      testapi.Version(),
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
	}
	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, eventA)
	}()
	got := <-wi.ResultChan()
	if e, a := eventA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}
