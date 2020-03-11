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

package containermap

import (
	"testing"
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
		// Build a new containerMap from the testCases, checking proper
		// addition, retrieval along the way.
		cm := NewContainerMap()
		for i := range tc.containerNames {
			cm.Add(tc.podUID, tc.containerNames[i], tc.containerIDs[i])

			containerID, err := cm.GetContainerID(tc.podUID, tc.containerNames[i])
			if err != nil {
				t.Errorf("error adding and retrieving containerID: %v", err)
			}
			if containerID != tc.containerIDs[i] {
				t.Errorf("mismatched containerIDs %v, %v", containerID, tc.containerIDs[i])
			}

			podUID, containerName, err := cm.GetContainerRef(containerID)
			if err != nil {
				t.Errorf("error retrieving container reference: %v", err)
			}
			if podUID != tc.podUID {
				t.Errorf("mismatched pod UID %v, %v", tc.podUID, podUID)
			}
			if containerName != tc.containerNames[i] {
				t.Errorf("mismatched container Name %v, %v", tc.containerNames[i], containerName)
			}
		}

		// Remove all entries from the containerMap, checking proper removal of
		// each along the way.
		for i := range tc.containerNames {
			cm.RemoveByContainerID(tc.containerIDs[i])
			containerID, err := cm.GetContainerID(tc.podUID, tc.containerNames[i])
			if err == nil {
				t.Errorf("unexpected retrieval of containerID after removal: %v", containerID)
			}

			cm.Add(tc.podUID, tc.containerNames[i], tc.containerIDs[i])

			cm.RemoveByContainerRef(tc.podUID, tc.containerNames[i])
			podUID, containerName, err := cm.GetContainerRef(tc.containerIDs[i])
			if err == nil {
				t.Errorf("unexpected retrieval of container reference after removal: (%v, %v)", podUID, containerName)
			}
		}

		// Verify containerMap now empty.
		if len(cm) != 0 {
			t.Errorf("unexpected entries still in containerMap: %v", cm)
		}

	}
}
