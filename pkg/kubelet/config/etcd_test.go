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

package config

import (
	"reflect"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestEtcdSourceExistingBoundPods(t *testing.T) {
	// Arrange
	key := "/registry/nodes/machine/boundpods"
	fakeEtcdClient := tools.NewFakeEtcdClient(t)
	updates := make(chan interface{})

	fakeEtcdClient.Set(
		key,
		runtime.EncodeOrDie(latest.Codec, &api.BoundPods{
			Items: []api.BoundPod{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: "default"},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image: "foo:v1",
							}}}},
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "bar",
						Namespace: "default"},
					Spec: api.PodSpec{
						Containers: []api.Container{
							{
								Image: "foo:v1",
							}}}}}}),
		0)

	// Act
	NewSourceEtcd(key, fakeEtcdClient, updates)

	// Assert
	select {
	case got := <-updates:
		update := got.(kubelet.PodUpdate)
		if len(update.Pods) != 2 ||
			update.Pods[0].ObjectMeta.Name != "foo" ||
			update.Pods[1].ObjectMeta.Name != "bar" {
			t.Errorf("Unexpected update response: %#v", update)
		}
	case <-time.After(200 * time.Millisecond):
		t.Errorf("Expected update, timeout instead")
	}
}

func TestEventToPods(t *testing.T) {
	tests := []struct {
		input watch.Event
		pods  []api.BoundPod
		fail  bool
	}{
		{
			input: watch.Event{Object: nil},
			pods:  []api.BoundPod{},
			fail:  false,
		},
		{
			input: watch.Event{Object: &api.BoundPods{}},
			pods:  []api.BoundPod{},
			fail:  false,
		},
		{
			input: watch.Event{
				Object: &api.BoundPods{
					Items: []api.BoundPod{
						{ObjectMeta: api.ObjectMeta{UID: "111", Name: "foo", Namespace: "foo"}},
						{ObjectMeta: api.ObjectMeta{UID: "222", Name: "bar", Namespace: "bar"}},
					},
				},
			},
			pods: []api.BoundPod{
				{ObjectMeta: api.ObjectMeta{UID: "111", Name: "foo", Namespace: "foo"}, Spec: api.PodSpec{}},
				{ObjectMeta: api.ObjectMeta{UID: "222", Name: "bar", Namespace: "bar"}, Spec: api.PodSpec{}},
			},
			fail: false,
		},
		{
			input: watch.Event{
				Object: &api.BoundPods{
					Items: []api.BoundPod{
						{ObjectMeta: api.ObjectMeta{UID: "111", Name: "foo"}},
					},
				},
			},
			pods: []api.BoundPod{
				{ObjectMeta: api.ObjectMeta{UID: "111", Name: "foo", Namespace: "default"}, Spec: api.PodSpec{}},
			},
			fail: false,
		},
	}

	for i, tt := range tests {
		pods, err := eventToPods(tt.input)
		if !reflect.DeepEqual(tt.pods, pods) {
			t.Errorf("case %d: expected output %#v, got %#v", i, tt.pods, pods)
		}
		if tt.fail != (err != nil) {
			t.Errorf("case %d: got fail=%t but err=%v", i, tt.fail, err)
		}
	}
}
