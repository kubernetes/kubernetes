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

package runtime_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

	"github.com/google/gofuzz"
)

func TestIsList(t *testing.T) {
	tests := []struct {
		obj    runtime.Object
		isList bool
	}{
		{&api.PodList{}, true},
		{&api.Pod{}, false},
	}
	for _, item := range tests {
		if e, a := item.isList, runtime.IsListType(item.obj); e != a {
			t.Errorf("%v: Expected %v, got %v", reflect.TypeOf(item.obj), e, a)
		}
	}
}

func TestExtractList(t *testing.T) {
	pl := &api.PodList{
		Items: []api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "1"}},
			{ObjectMeta: api.ObjectMeta{Name: "2"}},
			{ObjectMeta: api.ObjectMeta{Name: "3"}},
		},
	}
	list, err := runtime.ExtractList(pl)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i].(*api.Pod).Name, pl.Items[i].Name; e != a {
			t.Fatalf("Expected %v, got %v", e, a)
		}
	}
}

func TestExtractListGeneric(t *testing.T) {
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
			&api.Service{ObjectMeta: api.ObjectMeta{Name: "2"}},
		},
	}
	list, err := runtime.ExtractList(pl)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	if obj, ok := list[0].(*api.Pod); !ok {
		t.Fatalf("Expected list[0] to be *api.Pod, it is %#v", obj)
	}
	if obj, ok := list[1].(*api.Service); !ok {
		t.Fatalf("Expected list[1] to be *api.Service, it is %#v", obj)
	}
}

type fakePtrInterfaceList struct {
	Items *[]runtime.Object
}

func (f fakePtrInterfaceList) IsAnAPIObject() {}

func TestExtractListOfInterfacePtrs(t *testing.T) {
	pl := &fakePtrInterfaceList{
		Items: &[]runtime.Object{},
	}
	list, err := runtime.ExtractList(pl)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if len(list) > 0 {
		t.Fatalf("Expected empty list, got %#v", list)
	}
}

type fakePtrValueList struct {
	Items []*api.Pod
}

func (f fakePtrValueList) IsAnAPIObject() {}

func TestExtractListOfValuePtrs(t *testing.T) {
	pl := &fakePtrValueList{
		Items: []*api.Pod{
			{ObjectMeta: api.ObjectMeta{Name: "1"}},
			{ObjectMeta: api.ObjectMeta{Name: "2"}},
		},
	}
	list, err := runtime.ExtractList(pl)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if obj, ok := list[i].(*api.Pod); !ok {
			t.Fatalf("Expected list[%d] to be *api.Pod, it is %#v", i, obj)
		}
	}
}

func TestDecodeList(t *testing.T) {
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
			&runtime.Unknown{TypeMeta: runtime.TypeMeta{Kind: "Pod", APIVersion: testapi.Default.Version()}, RawJSON: []byte(`{"kind":"Pod","apiVersion":"` + testapi.Default.Version() + `","metadata":{"name":"test"}}`)},
			&runtime.Unstructured{TypeMeta: runtime.TypeMeta{Kind: "Foo", APIVersion: "Bar"}, Object: map[string]interface{}{"test": "value"}},
		},
	}
	if errs := runtime.DecodeList(pl.Items, api.Scheme); len(errs) != 0 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*api.Pod); !ok || pod.Name != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
}

func TestSetList(t *testing.T) {
	pl := &api.PodList{}
	list := []runtime.Object{
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "3"}},
	}
	err := runtime.SetList(pl, list)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i].(*api.Pod).Name, pl.Items[i].Name; e != a {
			t.Fatalf("Expected %v, got %v", e, a)
		}
	}
}

func TestSetListToRuntimeObjectArray(t *testing.T) {
	pl := &api.List{}
	list := []runtime.Object{
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "2"}},
		&api.Pod{ObjectMeta: api.ObjectMeta{Name: "3"}},
	}
	err := runtime.SetList(pl, list)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i], pl.Items[i]; e != a {
			t.Fatalf("%d: unmatched: %s", i, util.ObjectDiff(e, a))
		}
	}
}

func TestSetExtractListRoundTrip(t *testing.T) {
	fuzzer := fuzz.New().NilChance(0).NumElements(1, 5)
	for i := 0; i < 5; i++ {
		start := &api.PodList{}
		fuzzer.Fuzz(&start.Items)

		list, err := runtime.ExtractList(start)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		got := &api.PodList{}
		err = runtime.SetList(got, list)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		if e, a := start, got; !reflect.DeepEqual(e, a) {
			t.Fatalf("Expected %#v, got %#v", e, a)
		}
	}
}
