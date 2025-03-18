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

func TestContainerMapCloneUnshared(t *testing.T) {
	cm := NewContainerMap()
	// add random fake data
	cm.Add("fakePodUID-1", "fakeContainerName-a1", "fakeContainerID-A")
	cm.Add("fakePodUID-2", "fakeContainerName-b2", "fakeContainerID-B")
	cm.Add("fakePodUID-2", "fakeContainerName-c2", "fakeContainerID-C")
	cm.Add("fakePodUID-3", "fakeContainerName-d3", "fakeContainerID-D")
	cm.Add("fakePodUID-3", "fakeContainerName-e3", "fakeContainerID-E")
	cm.Add("fakePodUID-3", "fakeContainerName-f3", "fakeContainerID-F")

	// early sanity check, random ID, no special meaning
	podUID, containerName, err := cm.GetContainerRef("fakeContainerID-C")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if podUID != "fakePodUID-2" || containerName != "fakeContainerName-c2" {
		t.Fatalf("unexpected data: uid=%q name=%q", podUID, containerName)
	}
	if cID, err := cm.GetContainerID(podUID, containerName); err != nil || cID != "fakeContainerID-C" {
		t.Fatalf("unexpected data: cid=%q err=%v", cID, err)
	}

	cloned := cm.Clone()
	cloned.RemoveByContainerRef("fakePodUID-2", "fakeContainerName-c2")
	// check is actually gone
	if cID, err := cloned.GetContainerID("fakePodUID-2", "fakeContainerName-c2"); err == nil || cID != "" {
		t.Fatalf("unexpected data found: cid=%q", cID)
	}

	// check the original copy didn't change
	// early sanity check, random ID, no special meaning
	podUIDRedo, containerNameRedo, err2 := cm.GetContainerRef("fakeContainerID-C")
	if err != nil {
		t.Fatalf("unexpected error: %v", err2)
	}
	if podUIDRedo != "fakePodUID-2" || containerNameRedo != "fakeContainerName-c2" {
		t.Fatalf("unexpected data: uid=%q name=%q", podUIDRedo, containerNameRedo)
	}
	if cID, err := cm.GetContainerID(podUIDRedo, containerNameRedo); err != nil || cID != "fakeContainerID-C" {
		t.Fatalf("unexpected data: cid=%q", cID)
	}
}

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
		cm.Visit(func(podUID string, containerName string, containerID string) {
			cm.RemoveByContainerID(containerID)
			containerID, err := cm.GetContainerID(podUID, containerName)
			if err == nil {
				t.Errorf("unexpected retrieval of containerID after removal: %v", containerID)
			}

			cm.Add(podUID, containerName, containerID)

			cm.RemoveByContainerRef(podUID, containerName)
			id, cn, err := cm.GetContainerRef(containerID)
			if err == nil {
				t.Errorf("unexpected retrieval of container reference after removal: (%v, %v)", id, cn)
			}
		})

		// Verify containerMap now empty.
		if len(cm) != 0 {
			t.Errorf("unexpected entries still in containerMap: %v", cm)
		}
	}
}
