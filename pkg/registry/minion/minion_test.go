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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestRegistry(t *testing.T) {
	m := NewRegistry([]string{"foo", "bar"}, api.NodeResources{})
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
	if len(list.Items) != 2 || !contains(list, "foo") || !contains(list, "baz") {
		t.Errorf("unexpected %v", list)
	}
}

func contains(nodes *api.MinionList, nodeID string) bool {
	for _, node := range nodes.Items {
		if node.ID == nodeID {
			return true
		}
	}
	return false
}
