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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type alwaysYes struct{}

func (alwaysYes) HealthCheck(host string) (probe.Status, error) {
	return probe.Healthy, nil
}

func TestBasicDelegation(t *testing.T) {
	ctx := api.NewContext()
	mockMinionRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2", "m3"}, api.NodeResources{})
	healthy := NewHealthyRegistry(
		mockMinionRegistry,
		alwaysYes{},
		&util.FakeClock{},
		60*time.Second,
	)
	list, err := healthy.ListMinions(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(list, &mockMinionRegistry.Minions) {
		t.Errorf("Expected %v, Got %v", mockMinionRegistry.Minions, list)
	}
	err = healthy.CreateMinion(ctx, &api.Node{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	minion, err := healthy.GetMinion(ctx, "m1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if minion == nil {
		t.Errorf("Unexpected absence of 'm1'")
	}
	minion, err = healthy.GetMinion(ctx, "m5")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	if minion != nil {
		t.Errorf("Unexpected presence of 'm5'")
	}
}

type notMinion struct {
	minion string
}

func (n *notMinion) HealthCheck(host string) (probe.Status, error) {
	if host != n.minion {
		return probe.Healthy, nil
	} else {
		return probe.Unhealthy, nil
	}
}

func TestFiltering(t *testing.T) {
	ctx := api.NewContext()
	mockMinionRegistry := registrytest.NewMinionRegistry([]string{"m1", "m2", "m3"}, api.NodeResources{})
	healthy := NewHealthyRegistry(
		mockMinionRegistry,
		&notMinion{minion: "m1"},
		&util.FakeClock{},
		60*time.Second,
	)
	expected := []string{"m1", "m2", "m3"}
	list, err := healthy.ListMinions(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	expectedMinions := registrytest.MakeMinionList(expected, api.NodeResources{})
	expectedMinions.Items[0].Status.Conditions = []api.NodeCondition{{Kind: api.NodeReady, Status: api.ConditionNone}}
	expectedMinions.Items[1].Status.Conditions = []api.NodeCondition{{Kind: api.NodeReady, Status: api.ConditionFull}}
	expectedMinions.Items[2].Status.Conditions = []api.NodeCondition{{Kind: api.NodeReady, Status: api.ConditionFull}}
	if !reflect.DeepEqual(list, expectedMinions) {
		t.Errorf("Expected %v, Got %v", expected, list)
	}
	minion, err := healthy.GetMinion(ctx, "m1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if minion == nil {
		t.Errorf("Unexpected empty 'm1'")
	}
}
