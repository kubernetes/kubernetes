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
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

func TestCloudList(t *testing.T) {
	ctx := api.NewContext()
	instances := []string{"m1", "m2"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	registry, err := NewCloudRegistry(&fakeCloud, ".*", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	list, err := registry.ListMinions(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(list, registrytest.MakeMinionList(instances, api.NodeResources{})) {
		t.Errorf("Unexpected inequality: %#v, %#v", list, instances)
	}
}

func TestCloudGet(t *testing.T) {
	ctx := api.NewContext()
	instances := []string{"m1", "m2"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	registry, err := NewCloudRegistry(&fakeCloud, ".*", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	minion, err := registry.GetMinion(ctx, "m1")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if minion == nil {
		t.Errorf("Unexpected !contains")
	}

	minion, err = registry.GetMinion(ctx, "m100")
	if err == nil {
		t.Errorf("unexpected non error")
	}

	if minion != nil {
		t.Errorf("Unexpected contains")
	}
}

func TestCloudListRegexp(t *testing.T) {
	ctx := api.NewContext()
	instances := []string{"m1", "m2", "n1", "n2"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	registry, err := NewCloudRegistry(&fakeCloud, "m[0-9]+", nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	list, err := registry.ListMinions(ctx)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	expectedList := registrytest.MakeMinionList([]string{"m1", "m2"}, api.NodeResources{})
	if !reflect.DeepEqual(list, expectedList) {
		t.Errorf("Unexpected inequality: %#v, %#v", list, expectedList)
	}
}
