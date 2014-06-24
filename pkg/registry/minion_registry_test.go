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

package registry

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestMinionRegistry(t *testing.T) {
	m := MakeMinionRegistry([]string{"foo", "bar"})
	if has, err := m.Contains("foo"); !has || err != nil {
		t.Errorf("missing expected object")
	}
	if has, err := m.Contains("bar"); !has || err != nil {
		t.Errorf("missing expected object")
	}
	if has, err := m.Contains("baz"); has || err != nil {
		t.Errorf("has unexpected object")
	}

	if err := m.Insert("baz"); err != nil {
		t.Errorf("insert failed")
	}
	if has, err := m.Contains("baz"); !has || err != nil {
		t.Errorf("insert didn't actually insert")
	}

	if err := m.Delete("bar"); err != nil {
		t.Errorf("delete failed")
	}
	if has, err := m.Contains("bar"); has || err != nil {
		t.Errorf("delete didn't actually delete")
	}

	list, err := m.List()
	if err != nil {
		t.Errorf("got error calling List")
	}
	if !reflect.DeepEqual(list, []string{"baz", "foo"}) {
		t.Errorf("Unexpected list value: %#v", list)
	}
}

func TestMinionRegistryStorage(t *testing.T) {
	m := MakeMinionRegistry([]string{"foo", "bar"})
	ms := MakeMinionRegistryStorage(m)

	if obj, err := ms.Get("foo"); err != nil || obj.(api.Minion).ID != "foo" {
		t.Errorf("missing expected object")
	}
	if obj, err := ms.Get("bar"); err != nil || obj.(api.Minion).ID != "bar" {
		t.Errorf("missing expected object")
	}
	if _, err := ms.Get("baz"); err != ErrDoesNotExist {
		t.Errorf("has unexpected object")
	}

	if _, err := ms.Create(api.Minion{JSONBase: api.JSONBase{ID: "baz"}}); err != nil {
		t.Errorf("insert failed")
	}
	if obj, err := ms.Get("baz"); err != nil || obj.(api.Minion).ID != "baz" {
		t.Errorf("insert didn't actually insert")
	}

	if _, err := ms.Delete("bar"); err != nil {
		t.Errorf("delete failed")
	}
	if _, err := ms.Get("bar"); err != ErrDoesNotExist {
		t.Errorf("delete didn't actually delete")
	}
	if _, err := ms.Delete("bar"); err != ErrDoesNotExist {
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
