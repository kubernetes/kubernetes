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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestEventToPods(t *testing.T) {
	tests := []struct {
		input watch.Event
		pods  []kubelet.Pod
		fail  bool
	}{
		{
			input: watch.Event{Object: nil},
			pods:  []kubelet.Pod{},
			fail:  true,
		},
		{
			input: watch.Event{Object: &api.ContainerManifestList{}},
			pods:  []kubelet.Pod{},
			fail:  false,
		},
		{
			input: watch.Event{
				Object: &api.ContainerManifestList{
					Items: []api.ContainerManifest{
						{ID: "foo"},
						{ID: "bar"},
					},
				},
			},
			pods: []kubelet.Pod{
				{Name: "foo", Manifest: api.ContainerManifest{ID: "foo"}},
				{Name: "bar", Manifest: api.ContainerManifest{ID: "bar"}},
			},
			fail: false,
		},
		{
			input: watch.Event{
				Object: &api.ContainerManifestList{
					Items: []api.ContainerManifest{
						{ID: ""},
						{ID: ""},
					},
				},
			},
			pods: []kubelet.Pod{
				{Name: "1", Manifest: api.ContainerManifest{ID: ""}},
				{Name: "2", Manifest: api.ContainerManifest{ID: ""}},
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
