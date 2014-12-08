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

package controller

import (
	"fmt"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
)

func newNode(name string) *api.Node {
	return &api.Node{ObjectMeta: api.ObjectMeta{Name: name}}
}

type FakeMinionHandler struct {
	client.Fake
	client.FakeNodes

	// Input: Hooks determine if request is valid or not
	CreateHook func(*FakeMinionHandler, *api.Node) bool
	Existing   []*api.Node

	// Output
	CreatedMinions []*api.Node
	DeletedMinions []*api.Node
	RequestCount   int
}

func (c *FakeMinionHandler) Nodes() client.NodeInterface {
	return c
}

func (m *FakeMinionHandler) Create(minion *api.Node) (*api.Node, error) {
	defer func() { m.RequestCount++ }()
	if m.CreateHook == nil || m.CreateHook(m, minion) {
		m.CreatedMinions = append(m.CreatedMinions, minion)
		return minion, nil
	} else {
		return nil, fmt.Errorf("Create error.")
	}
}

func (m *FakeMinionHandler) List() (*api.NodeList, error) {
	defer func() { m.RequestCount++ }()
	minions := []api.Node{}
	for i := 0; i < len(m.Existing); i++ {
		if !contains(m.Existing[i], m.DeletedMinions) {
			minions = append(minions, *m.Existing[i])
		}
	}
	for i := 0; i < len(m.CreatedMinions); i++ {
		if !contains(m.Existing[i], m.DeletedMinions) {
			minions = append(minions, *m.CreatedMinions[i])
		}
	}
	return &api.NodeList{Items: minions}, nil
}

func (m *FakeMinionHandler) Delete(id string) error {
	m.DeletedMinions = append(m.DeletedMinions, newNode(id))
	m.RequestCount++
	return nil
}

func TestSyncStaticCreateMinion(t *testing.T) {
	fakeMinionHandler := &FakeMinionHandler{
		CreateHook: func(fake *FakeMinionHandler, minion *api.Node) bool {
			return true
		},
	}
	minionController := NewMinionController(nil, ".*", []string{"minion0"}, &api.NodeResources{}, fakeMinionHandler)
	if err := minionController.SyncStatic(time.Millisecond); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeMinionHandler.RequestCount != 1 {
		t.Errorf("Expected 1 call, but got %v.", fakeMinionHandler.RequestCount)
	}
	if len(fakeMinionHandler.CreatedMinions) != 1 {
		t.Errorf("expect only 1 minion created, got %v", len(fakeMinionHandler.CreatedMinions))
	}
	if fakeMinionHandler.CreatedMinions[0].Name != "minion0" {
		t.Errorf("unexpect minion %v created", fakeMinionHandler.CreatedMinions[0].Name)
	}
}

func TestSyncStaticCreateMinionWithError(t *testing.T) {
	fakeMinionHandler := &FakeMinionHandler{
		CreateHook: func(fake *FakeMinionHandler, minion *api.Node) bool {
			if fake.RequestCount == 0 {
				return false
			}
			return true
		},
	}
	minionController := NewMinionController(nil, ".*", []string{"minion0"}, &api.NodeResources{}, fakeMinionHandler)
	if err := minionController.SyncStatic(time.Millisecond); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeMinionHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeMinionHandler.RequestCount)
	}
	if len(fakeMinionHandler.CreatedMinions) != 1 {
		t.Errorf("expect only 1 minion created, got %v", len(fakeMinionHandler.CreatedMinions))
	}
	if fakeMinionHandler.CreatedMinions[0].Name != "minion0" {
		t.Errorf("unexpect minion %v created", fakeMinionHandler.CreatedMinions[0].Name)
	}
}

func TestSyncCloudCreateMinion(t *testing.T) {
	fakeMinionHandler := &FakeMinionHandler{
		Existing: []*api.Node{newNode("minion0")},
	}
	instances := []string{"minion0", "minion1"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, nil, fakeMinionHandler)
	if err := minionController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeMinionHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeMinionHandler.RequestCount)
	}
	if len(fakeMinionHandler.CreatedMinions) != 1 {
		t.Errorf("expect only 1 minion created, got %v", len(fakeMinionHandler.CreatedMinions))
	}
	if fakeMinionHandler.CreatedMinions[0].Name != "minion1" {
		t.Errorf("unexpect minion %v created", fakeMinionHandler.CreatedMinions[0].Name)
	}
}

func TestSyncCloudDeleteMinion(t *testing.T) {
	fakeMinionHandler := &FakeMinionHandler{
		Existing: []*api.Node{newNode("minion0"), newNode("minion1")},
	}
	instances := []string{"minion0"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, nil, fakeMinionHandler)
	if err := minionController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeMinionHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeMinionHandler.RequestCount)
	}
	if len(fakeMinionHandler.DeletedMinions) != 1 {
		t.Errorf("expect only 1 minion deleted, got %v", len(fakeMinionHandler.DeletedMinions))
	}
	if fakeMinionHandler.DeletedMinions[0].Name != "minion1" {
		t.Errorf("unexpect minion %v created", fakeMinionHandler.DeletedMinions[0].Name)
	}
}

func TestSyncCloudRegexp(t *testing.T) {
	fakeMinionHandler := &FakeMinionHandler{
		Existing: []*api.Node{newNode("minion0")},
	}
	instances := []string{"minion0", "minion1", "node0"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, "minion[0-9]+", nil, nil, fakeMinionHandler)
	if err := minionController.SyncCloud(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if fakeMinionHandler.RequestCount != 2 {
		t.Errorf("Expected 2 call, but got %v.", fakeMinionHandler.RequestCount)
	}
	if len(fakeMinionHandler.CreatedMinions) != 1 {
		t.Errorf("expect only 1 minion created, got %v", len(fakeMinionHandler.CreatedMinions))
	}
	if fakeMinionHandler.CreatedMinions[0].Name != "minion1" {
		t.Errorf("unexpect minion %v created", fakeMinionHandler.CreatedMinions[0].Name)
	}
}

func contains(minion *api.Node, minions []*api.Node) bool {
	for i := 0; i < len(minions); i++ {
		if minion.Name == minions[i].Name {
			return true
		}
	}
	return false
}
