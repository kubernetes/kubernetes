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

package master

import (
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
)

type FakeContainerInfo struct {
	host string
	id   string
	data interface{}
	err  error
}

func (f *FakeContainerInfo) GetContainerInfo(host, id string) (interface{}, error) {
	f.host = host
	f.id = id
	return f.data, f.err
}

func TestPodCacheGet(t *testing.T) {
	cache := NewPodCache(nil, nil, time.Second*1)

	pod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	cache.podInfo["foo"] = pod

	info, err := cache.GetContainerInfo("host", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, pod) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", pod, info)
	}
}

func TestPodCacheGetMissing(t *testing.T) {
	cache := NewPodCache(nil, nil, time.Second*1)

	info, err := cache.GetContainerInfo("host", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if info != nil {
		t.Errorf("Unexpected info: %#v", info)
	}
}

func TestPodGetContainerInfo(t *testing.T) {
	pod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
	}
	fake := FakeContainerInfo{
		data: pod,
	}
	cache := NewPodCache(&fake, nil, time.Second*1)

	cache.updateContainerInfo("host", "foo")

	if fake.host != "host" || fake.id != "foo" {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetContainerInfo("host", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, pod) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", pod, info)
	}
}

func TestPodUpdateAllContainers(t *testing.T) {
	pod := api.Pod{
		JSONBase: api.JSONBase{ID: "foo"},
		CurrentState: api.PodState{
			Host: "machine",
		},
	}
	pods := []api.Pod{pod}
	mockRegistry := registry.MakeMockPodRegistry(pods)
	fake := FakeContainerInfo{
		data: pod,
	}
	cache := NewPodCache(&fake, mockRegistry, time.Second*1)

	cache.UpdateAllContainers()

	if fake.host != "machine" || fake.id != "foo" {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetContainerInfo("machine", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, pod) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", pod, info)
	}
}
