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
)

type FakePodInfoGetter struct {
	host      string
	id        string
	namespace string
	data      api.PodInfo
	err       error
}

func (f *FakePodInfoGetter) GetPodInfo(host, namespace, id string) (api.PodInfo, error) {
	f.host = host
	f.id = id
	f.namespace = namespace
	return f.data, f.err
}

func TestPodCacheGetDifferentNamespace(t *testing.T) {
	cache := NewPodCache(nil, nil)

	expectedDefault := api.PodInfo{
		"foo": api.ContainerStatus{},
	}
	expectedOther := api.PodInfo{
		"bar": api.ContainerStatus{},
	}

	cache.podInfo[makePodCacheKey(api.NamespaceDefault, "foo")] = expectedDefault
	cache.podInfo[makePodCacheKey("other", "foo")] = expectedOther

	info, err := cache.GetPodInfo("host", api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expectedDefault) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expectedOther, info)
	}

	info, err = cache.GetPodInfo("host", "other", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expectedOther) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expectedOther, info)
	}
}

func TestPodCacheGet(t *testing.T) {
	cache := NewPodCache(nil, nil)

	expected := api.PodInfo{
		"foo": api.ContainerStatus{},
	}
	cache.podInfo[makePodCacheKey(api.NamespaceDefault, "foo")] = expected

	info, err := cache.GetPodInfo("host", api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
	}
}

func TestPodCacheGetMissing(t *testing.T) {
	cache := NewPodCache(nil, nil)

	info, err := cache.GetPodInfo("host", api.NamespaceDefault, "foo")
	if err == nil {
		t.Errorf("Unexpected non-error: %#v", err)
	}
	if info != nil {
		t.Errorf("Unexpected info: %#v", info)
	}
}

func TestPodGetPodInfoGetter(t *testing.T) {
	expected := api.PodInfo{
		"foo": api.ContainerStatus{},
	}
	fake := FakePodInfoGetter{
		data: expected,
	}
	cache := NewPodCache(&fake, nil)

	cache.updatePodInfo("host", api.NamespaceDefault, "foo")

	if fake.host != "host" || fake.id != "foo" || fake.namespace != api.NamespaceDefault {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetPodInfo("host", api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
	}
}

func TestPodUpdateAllContainers(t *testing.T) {
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		CurrentState: api.PodState{
			Host: "machine",
		},
	}

	pods := []api.Pod{pod}
	mockRegistry := registrytest.NewPodRegistry(&api.PodList{Items: pods})

	expected := api.PodInfo{
		"foo": api.ContainerStatus{},
	}
	fake := FakePodInfoGetter{
		data: expected,
	}
	cache := NewPodCache(&fake, mockRegistry)

	cache.UpdateAllContainers()

	if fake.host != "machine" || fake.id != "foo" || fake.namespace != api.NamespaceDefault {
		t.Errorf("Unexpected access: %#v", fake)
	}

	info, err := cache.GetPodInfo("machine", api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	if !reflect.DeepEqual(info, expected) {
		t.Errorf("Unexpected mismatch. Expected: %#v, Got: #%v", &expected, info)
	}
}
