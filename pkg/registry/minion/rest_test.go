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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest/resttest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestMinionRegistryREST(t *testing.T) {
	registryNodeList := registrytest.MakeMinionList([]string{"foo", "bar"}, api.NodeResources{})
	ms := NewREST(registrytest.NewMinionRegistry(registryNodeList))

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

	obj, err := ms.Create(ctx, &api.Node{ObjectMeta: api.ObjectMeta{Name: "baz"}})
	if err != nil {
		t.Fatalf("insert failed: %v", err)
	}
	if !api.HasObjectMetaSystemFieldValues(&obj.(*api.Node).ObjectMeta) {
		t.Errorf("storage did not populate object meta field values")
	}
	if m, ok := obj.(*api.Node); !ok || m.Name != "baz" {
		t.Errorf("insert return value was weird: %#v", obj)
	}
	if obj, err := ms.Get(ctx, "baz"); err != nil || obj.(*api.Node).Name != "baz" {
		t.Errorf("insert didn't actually insert")
	}

	obj, err = ms.Delete(ctx, "bar")
	if err != nil {
		t.Fatalf("delete failed")
	}
	if s, ok := obj.(*api.Status); !ok || s.Status != api.StatusSuccess {
		t.Errorf("delete return value was weird: %#v", obj)
	}
	if _, err := ms.Get(ctx, "bar"); !errors.IsNotFound(err) {
		t.Errorf("delete didn't actually delete: %v", err)
	}

	_, err = ms.Delete(ctx, "bar")
	if err != ErrDoesNotExist {
		t.Fatalf("delete returned wrong error")
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

func TestMinionRegistryValidUpdate(t *testing.T) {
	registryNodeList := registrytest.MakeMinionList([]string{"foo", "bar"}, api.NodeResources{})
	storage := NewREST(registrytest.NewMinionRegistry(registryNodeList))
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
	if _, _, err = storage.Update(ctx, minion); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

var (
	validSelector   = map[string]string{"a": "b"}
	invalidSelector = map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
)

func TestMinionRegistryValidatesCreate(t *testing.T) {
	registryNodeList := registrytest.MakeMinionList([]string{"foo", "bar"}, api.NodeResources{})
	storage := NewREST(registrytest.NewMinionRegistry(registryNodeList))
	ctx := api.NewContext()
	failureCases := map[string]api.Node{
		"zero-length Name": {
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeLegacyHostIP, Address: "something"},
				},
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
			t.Errorf("Expected nil object")
		}
		if !errors.IsInvalid(err) {
			t.Errorf("Expected to get an invalid resource error, got %v", err)
		}
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

func TestCreate(t *testing.T) {
	registryNodeList := registrytest.MakeMinionList([]string{"foo", "bar"}, api.NodeResources{})
	registry := registrytest.NewMinionRegistry(registryNodeList)
	test := resttest.New(t, NewREST(registry), registry.SetError).ClusterScope()
	test.TestCreate(
		// valid
		&api.Node{
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeLegacyHostIP, Address: "something"},
				},
			},
		},
		// invalid
		&api.Node{
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		})
}

func TestListNodeListSelection(t *testing.T) {
	registryNodeList := &api.NodeList{
		Items: []api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"name": "nginx",
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name: "bar",
					Labels: map[string]string{
						"name": "nginx",
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name: "baz",
					Labels: map[string]string{
						"name": "redis",
					},
				},
			},
		},
	}
	storage := NewREST(registrytest.NewMinionRegistry(registryNodeList))
	ctx := api.NewContext()
	label, err := labels.Parse("name=nginx")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	nodeListObj, err := storage.List(ctx, label, labels.Everything())
	nodeList := nodeListObj.(*api.NodeList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(nodeList.Items) != 2 {
		t.Errorf("Unexpected node list: %#v", nodeList)
	}
	if nodeList.Items[0].Name != "foo" {
		t.Errorf("Unexpected node list: %#v", nodeList)
	}
	if nodeList.Items[1].Name != "bar" {
		t.Errorf("Unexpected node list: %#v", nodeList)
	}
}
