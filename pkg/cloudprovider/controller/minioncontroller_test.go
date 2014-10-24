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
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	fake_cloud "github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	etcdregistry "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pod"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"

	etcd "github.com/coreos/go-etcd/etcd"
)

func NewTestEtcdRegistry(client tools.EtcdClient) *etcdregistry.Registry {
	registry := etcdregistry.NewRegistry(
		tools.EtcdHelper{client, latest.Codec, tools.RuntimeVersionAdapter{latest.ResourceVersioner}},
		&pod.BasicBoundPodFactory{
			ServiceRegistry: &registrytest.ServiceRegistry{},
		},
	)
	return registry
}

func TestSyncCreateMinion(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	m1 := runtime.EncodeOrDie(latest.Codec, &api.Minion{ObjectMeta: api.ObjectMeta{Name: "m1"}})
	m2 := runtime.EncodeOrDie(latest.Codec, &api.Minion{ObjectMeta: api.ObjectMeta{Name: "m2"}})
	fakeClient.Set("/registry/minions/m1", m1, 0)
	fakeClient.Set("/registry/minions/m2", m2, 0)
	fakeClient.ExpectNotFoundGet("/registry/minions/m3")
	fakeClient.Data["/registry/minions"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{Value: m1},
					{Value: m2},
				},
			},
		},
		E: nil,
	}

	registry := NewTestEtcdRegistry(fakeClient)
	instances := []string{"m1", "m2", "m3"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, registry, time.Second)

	minion, err := registry.GetMinion(ctx, "m3")
	if minion != nil {
		t.Errorf("Unexpected contains")
	}

	err = minionController.Sync()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	minion, err = registry.GetMinion(ctx, "m3")
	if minion == nil {
		t.Errorf("Unexpected !contains")
	}
}

func TestSyncDeleteMinion(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	m1 := runtime.EncodeOrDie(latest.Codec, &api.Minion{ObjectMeta: api.ObjectMeta{Name: "m1"}})
	m2 := runtime.EncodeOrDie(latest.Codec, &api.Minion{ObjectMeta: api.ObjectMeta{Name: "m2"}})
	m3 := runtime.EncodeOrDie(latest.Codec, &api.Minion{ObjectMeta: api.ObjectMeta{Name: "m3"}})
	fakeClient.Set("/registry/minions/m1", m1, 0)
	fakeClient.Set("/registry/minions/m2", m2, 0)
	fakeClient.Set("/registry/minions/m3", m3, 0)
	fakeClient.Data["/registry/minions"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{Value: m1},
					{Value: m2},
					{Value: m3},
				},
			},
		},
		E: nil,
	}

	registry := NewTestEtcdRegistry(fakeClient)
	instances := []string{"m1", "m2"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, registry, time.Second)

	minion, err := registry.GetMinion(ctx, "m3")
	if minion == nil {
		t.Errorf("Unexpected !contains")
	}

	err = minionController.Sync()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	minion, err = registry.GetMinion(ctx, "m3")
	if minion != nil {
		t.Errorf("Unexpected contains")
	}
}

func TestSyncMinionRegexp(t *testing.T) {
	ctx := api.NewContext()
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.Data["/registry/minions"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{},
			},
		},
		E: nil,
	}

	registry := NewTestEtcdRegistry(fakeClient)
	instances := []string{"m1", "m2", "n1", "n2"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, "m[0-9]+", nil, registry, time.Second)

	err := minionController.Sync()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	var minion *api.Minion
	fakeClient.ExpectNotFoundGet("/registry/minions/n1")
	fakeClient.ExpectNotFoundGet("/registry/minions/n2")

	minion, err = registry.GetMinion(ctx, "m1")
	if minion == nil {
		t.Errorf("Unexpected !contains")
	}
	minion, err = registry.GetMinion(ctx, "m2")
	if minion == nil {
		t.Errorf("Unexpected !contains")
	}
	minion, err = registry.GetMinion(ctx, "n1")
	if minion != nil {
		t.Errorf("Unexpected !contains")
	}
	minion, err = registry.GetMinion(ctx, "n2")
	if minion != nil {
		t.Errorf("Unexpected !contains")
	}
}
