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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestMinionRegistryREST(t *testing.T) {
	ms := NewREST(registrytest.NewMinionRegistry([]string{"foo", "bar"}, api.NodeResources{}))
	ctx := api.NewContext()
	if obj, err := ms.Get(ctx, "foo"); err != nil || obj.(*api.Node).Name != "foo" {
		t.Errorf("missing expected object")
	}
	if obj, err := ms.Get(ctx, "bar"); err != nil || obj.(*api.Node).Name != "bar" {
		t.Errorf("missing expected object")
	}
	if _, err := ms.Get(ctx, "baz"); !errors.IsNotFound(err) {
		t.Errorf("has unexpected error: %v", err)
	}

	c, err := ms.Create(ctx, &api.Node{ObjectMeta: api.ObjectMeta{Name: "baz"}})
	if err != nil {
		t.Errorf("insert failed")
	}
	obj := <-c
	if !api.HasObjectMetaSystemFieldValues(&obj.Object.(*api.Node).ObjectMeta) {
		t.Errorf("storage did not populate object meta field values")
	}
	if m, ok := obj.Object.(*api.Node); !ok || m.Name != "baz" {
		t.Errorf("insert return value was weird: %#v", obj)
	}
	if obj, err := ms.Get(ctx, "baz"); err != nil || obj.(*api.Node).Name != "baz" {
		t.Errorf("insert didn't actually insert")
	}

	c, err = ms.Delete(ctx, "bar")
	if err != nil {
		t.Errorf("delete failed")
	}
	obj = <-c
	if s, ok := obj.Object.(*api.Status); !ok || s.Status != api.StatusSuccess {
		t.Errorf("delete return value was weird: %#v", obj)
	}
	if _, err := ms.Get(ctx, "bar"); !errors.IsNotFound(err) {
		t.Errorf("delete didn't actually delete: %v", err)
	}

	_, err = ms.Delete(ctx, "bar")
	if err != ErrDoesNotExist {
		t.Errorf("delete returned wrong error")
	}

	list, err := ms.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		t.Errorf("got error calling List")
	}
	expect := []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: "foo"},
		}, {
			ObjectMeta: api.ObjectMeta{Name: "baz"},
		},
	}
	nodeList := list.(*api.NodeList)
	if len(expect) != len(nodeList.Items) || !contains(nodeList, "foo") || !contains(nodeList, "baz") {
		t.Errorf("Unexpected list value: %#v", list)
	}
}

func TestMinionRegistryHealthCheck(t *testing.T) {
	minionRegistry := registrytest.NewMinionRegistry([]string{}, api.NodeResources{})
	minionHealthRegistry := HealthyRegistry{
		delegate: minionRegistry,
		client:   &notMinion{minion: "m1"},
	}

	ms := NewREST(&minionHealthRegistry)
	ctx := api.NewContext()

	c, err := ms.Create(ctx, &api.Node{ObjectMeta: api.ObjectMeta{Name: "m1"}})
	if err != nil {
		t.Errorf("insert failed")
	}
	result := <-c
	if m, ok := result.Object.(*api.Node); !ok || m.Name != "m1" {
		t.Errorf("insert return value was weird: %#v", result)
	}
	if _, err := ms.Get(ctx, "m1"); err != nil {
		t.Errorf("node is unhealthy, expect no error: %v", err)
	}
}

func contains(nodes *api.NodeList, nodeID string) bool {
	for _, node := range nodes.Items {
		if node.Name == nodeID {
			return true
		}
	}
	return false
}

func TestMinionRegistryInvalidUpdate(t *testing.T) {
	storage := NewREST(registrytest.NewMinionRegistry([]string{"foo", "bar"}, api.NodeResources{}))
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	minion, ok := obj.(*api.Node)
	if !ok {
		t.Fatalf("Object is not a minion: %#v", obj)
	}
	minion.Status.HostIP = "1.2.3.4"
	if _, err = storage.Update(ctx, minion); err == nil {
		t.Error("Unexpected non-error.")
	}
}

func TestMinionRegistryValidUpdate(t *testing.T) {
	storage := NewREST(registrytest.NewMinionRegistry([]string{"foo", "bar"}, api.NodeResources{}))
	ctx := api.NewContext()
	obj, err := storage.Get(ctx, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	minion, ok := obj.(*api.Node)
	if !ok {
		t.Fatalf("Object is not a minion: %#v", obj)
	}
	minion.Labels = map[string]string{
		"foo": "bar",
		"baz": "home",
	}
	if _, err = storage.Update(ctx, minion); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestMinionRegistryValidatesCreate(t *testing.T) {
	storage := NewREST(registrytest.NewMinionRegistry([]string{"foo", "bar"}, api.NodeResources{}))
	ctx := api.NewContext()
	validSelector := map[string]string{"a": "b"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	failureCases := map[string]api.Node{
		"zero-length Name": {
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				HostIP: "something",
			},
		},
		"invalid-labels": {
			ObjectMeta: api.ObjectMeta{
				Name:   "abc-123",
				Labels: invalidSelector,
			},
		},
	}
	for _, failureCase := range failureCases {
		c, err := storage.Create(ctx, &failureCase)
		if c != nil {
			t.Errorf("Expected nil channel")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
	}
}
