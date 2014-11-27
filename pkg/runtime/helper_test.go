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

package runtime_test

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

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
