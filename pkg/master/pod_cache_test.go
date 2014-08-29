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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/fsouza/go-dockerclient"
)

type FakePodInfoGetter struct {
	host string
	id   string
	data api.PodInfo
	err  error
}

func (f *FakePodInfoGetter) GetPodInfo(host, id string) (api.PodInfo, error) {
	f.host = host
	f.id = id
	return f.data, f.err
}

func TestPodCacheGet(t *testing.T) {
	cache := NewPodCache(nil, nil)

	expected := api.PodInfo{"foo": docker.Container{ID: "foo"}}
	cache.podInfo["foo"] = expected

	info, err := cache.GetPodInfo("host", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
	}
}

func TestPodCacheGetMissing(t *testing.T) {
	cache := NewPodCache(nil, nil)

	info, err := cache.GetPodInfo("host", "foo")
	if err == nil {
		t.Errorf("Unexpected non-error: %#v", err)
	}
	if info != nil {
		t.Errorf("Unexpected info: %#v", info)
	}
}

func TestPodGetPodInfoGetter(t *testing.T) {
	expected := api.PodInfo{"foo": docker.Container{ID: "foo"}}
	fake := FakePodInfoGetter{
		data: expected,
	}
	cache := NewPodCache(&fake, nil)

	cache.updatePodInfo("host", "foo")

	if fake.host != "host" || fake.id != "foo" {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetPodInfo("host", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
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
	mockRegistry := registrytest.NewPodRegistry(&api.PodList{Items: pods})

	expected := api.PodInfo{"foo": docker.Container{ID: "foo"}}
	fake := FakePodInfoGetter{
		data: expected,
	}
	cache := NewPodCache(&fake, mockRegistry)

	cache.UpdateAllContainers()

	if fake.host != "machine" || fake.id != "foo" {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetPodInfo("machine", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
	}
}
