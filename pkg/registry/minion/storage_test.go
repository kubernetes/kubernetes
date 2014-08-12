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

package minion

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestMinionRegistryStorage(t *testing.T) {
	m := NewRegistry([]string{"foo", "bar"})
	ms := NewRegistryStorage(m)

	if obj, err := ms.Get("foo"); err != nil || obj.(api.Minion).ID != "foo" {
		t.Errorf("missing expected object")
	}
	if obj, err := ms.Get("bar"); err != nil || obj.(api.Minion).ID != "bar" {
		t.Errorf("missing expected object")
	}
	if _, err := ms.Get("baz"); err != ErrDoesNotExist {
		t.Errorf("has unexpected object")
	}

	c, err := ms.Create(&api.Minion{JSONBase: api.JSONBase{ID: "baz"}})
	if err != nil {
		t.Errorf("insert failed")
	}
	obj := <-c
	if m, ok := obj.(api.Minion); !ok || m.ID != "baz" {
		t.Errorf("insert return value was weird: %#v", obj)
	}
	if obj, err := ms.Get("baz"); err != nil || obj.(api.Minion).ID != "baz" {
		t.Errorf("insert didn't actually insert")
	}

	c, err = ms.Delete("bar")
	if err != nil {
		t.Errorf("delete failed")
	}
	obj = <-c
	if s, ok := obj.(*api.Status); !ok || s.Status != api.StatusSuccess {
		t.Errorf("delete return value was weird: %#v", obj)
	}
	if _, err := ms.Get("bar"); err != ErrDoesNotExist {
		t.Errorf("delete didn't actually delete")
	}

	_, err = ms.Delete("bar")
	if err != ErrDoesNotExist {
		t.Errorf("delete returned wrong error")
	}

	list, err := ms.List(labels.Everything())
	if err != nil {
		t.Errorf("got error calling List")
	}
	expect := []api.Minion{
		{
			JSONBase: api.JSONBase{ID: "baz"},
		}, {
			JSONBase: api.JSONBase{ID: "foo"},
		},
	}
	if !reflect.DeepEqual(list.(api.MinionList).Items, expect) {
		t.Errorf("Unexpected list value: %#v", list)
	}
}
