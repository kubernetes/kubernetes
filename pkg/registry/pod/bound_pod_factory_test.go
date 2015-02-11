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

package pod

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestMakeBoundPodNoServices(t *testing.T) {
	factory := &BasicBoundPodFactory{}

	pod, err := factory.MakeBoundPod("machine", &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foobar"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: "foo",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	container := pod.Spec.Containers[0]
	if len(container.Env) != 0 {
		t.Errorf("Expected zero env vars, got: %#v", pod)
	}
	if pod.Name != "foobar" {
		t.Errorf("Failed to assign ID to pod: %#v", pod.Name)
	}

	if _, err := api.GetReference(pod); err != nil {
		t.Errorf("Unable to get a reference to bound pod: %v", err)
	}
}
