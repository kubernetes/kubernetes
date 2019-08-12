/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"testing"

	"k8s.io/api/core/v1"
	apimachinery "k8s.io/apimachinery/pkg/types"
)

func TestContainerMap(t *testing.T) {
	testCases := []struct {
		podUID         string
		containerNames []string
		containerIDs   []string
	}{
		{
			"fakePodUID",
			[]string{"fakeContainerName-1", "fakeContainerName-2"},
			[]string{"fakeContainerID-1", "fakeContainerName-2"},
		},
	}

	for _, tc := range testCases {
		pod := v1.Pod{}
		pod.UID = apimachinery.UID(tc.podUID)

		// Build a new containerMap from the testCases, checking proper
		// addition, retrieval along the way.
		cm := newContainerMap()
		for i := range tc.containerNames {
			container := v1.Container{Name: tc.containerNames[i]}

			cm.Add(&pod, &container, tc.containerIDs[i])
			containerID, err := cm.Get(&pod, &container)
			if err != nil {
				t.Errorf("error adding and retrieving container: %v", err)
			}
			if containerID != tc.containerIDs[i] {
				t.Errorf("mismatched containerIDs %v, %v", containerID, tc.containerIDs[i])
			}
		}

		// Remove all entries from the containerMap, checking proper removal of
		// each along the way.
		for i := range tc.containerNames {
			container := v1.Container{Name: tc.containerNames[i]}
			cm.Remove(tc.containerIDs[i])
			containerID, err := cm.Get(&pod, &container)
			if err == nil {
				t.Errorf("unexpected retrieval of containerID after removal: %v", containerID)
			}
		}

		// Verify containerMap now empty.
		if len(cm) != 0 {
			t.Errorf("unexpected entries still in containerMap: %v", cm)
		}

	}
}
