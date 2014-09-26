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

package scheduler

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestPodFitsPorts(t *testing.T) {
	tests := []struct {
		pod          api.Pod
		existingPods []api.Pod
		fits         bool
		test         string
	}{
		{
			pod:          api.Pod{},
			existingPods: []api.Pod{},
			fits:         true,
			test:         "nothing running",
		},
		{
			pod: newPod("m1", 8080),
			existingPods: []api.Pod{
				newPod("m1", 9090),
			},
			fits: true,
			test: "other port",
		},
		{
			pod: newPod("m1", 8080),
			existingPods: []api.Pod{
				newPod("m1", 8080),
			},
			fits: false,
			test: "same port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			existingPods: []api.Pod{
				newPod("m1", 8080),
			},
			fits: false,
			test: "second port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			existingPods: []api.Pod{
				newPod("m1", 8001, 8080),
			},
			fits: false,
			test: "second port",
		},
	}
	for _, test := range tests {
		fits, err := PodFitsPorts(test.pod, test.existingPods, "machine")
		if err != nil {
			t.Errorf("unexpected error: %v")
		}
		if test.fits != fits {
			t.Errorf("%s: expected %v, saw %v", test.test, test.fits, fits)
		}
	}
}
