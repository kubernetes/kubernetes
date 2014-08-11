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
)

func TestRegistry(t *testing.T) {
	m := NewRegistry([]string{"foo", "bar"})
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
