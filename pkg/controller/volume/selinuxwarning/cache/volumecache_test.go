/*
Copyright 2024 The Kubernetes Authors.

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

package cache

import (
	"reflect"
	"sort"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/internal/parse"
	"k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/translator"
)

func getTestLoggers(t *testing.T) (klog.Logger, klog.Logger) {
	// Logger with the default V(5), does not log dumps
	logger, _ := ktesting.NewTestContext(t)

	// Logger with V(dumpLogLevel), logs dumps
	logConfig := ktesting.NewConfig(ktesting.Verbosity(dumpLogLevel))
	dumpLogger := ktesting.NewLogger(t, logConfig)

	return logger, dumpLogger
}

func sortConflicts(conflicts []Conflict) {
	sort.Slice(conflicts, func(i, j int) bool {
		return conflicts[i].Pod.String() < conflicts[j].Pod.String()
	})
}

// verifyReverseIndexConsistency checks that forward and reverse indexes are symmetric
func verifyReverseIndexConsistency(t *testing.T, c *volumeCache) {
	t.Helper()

	// For every (pod, volume) in reverse index, verify it exists in forward index.
	for podKey, volumes := range c.podToVolumes {
		for volumeName := range volumes {
			volume, found := c.volumes[volumeName]
			if !found {
				t.Errorf("Reverse index has pod %s -> volume %s, but volume not in forward index", podKey, volumeName)
				continue
			}
			if _, found := volume.pods[podKey]; !found {
				t.Errorf("Reverse index has pod %s -> volume %s, but pod not in volume's pod list", podKey, volumeName)
			}
		}
	}

	// For every (volume, pod) in forward index, verify it exists in reverse index.
	for volumeName, volume := range c.volumes {
		for podKey := range volume.pods {
			podVolumes, found := c.podToVolumes[podKey]
			if !found {
				t.Errorf("Forward index has volume %s -> pod %s, but pod not in reverse index", volumeName, podKey)
				continue
			}
			if _, found := podVolumes[volumeName]; !found {
				t.Errorf("Forward index has volume %s -> pod %s, but volume not in pod's volume list", volumeName, podKey)
			}
		}
	}
}

// Delete all items in a bigger cache and check it's empty
func TestVolumeCache_DeleteAll(t *testing.T) {
	var podsToDelete []cache.ObjectName
	seLinuxTranslator := &translator.ControllerSELinuxTranslator{}
	c := NewVolumeLabelCache(seLinuxTranslator).(*volumeCache)
	logger, dumpLogger := getTestLoggers(t)

	// Arrange: add a lot of volumes to the cache
	for _, namespace := range []string{"ns1", "ns2", "ns3", "ns4"} {
		for _, name := range []string{"pod1", "pod2", "pod3", "pod4"} {
			for _, volumeName := range []v1.UniqueVolumeName{"vol1", "vol2", "vol3", "vol4"} {
				podKey := cache.ObjectName{Namespace: namespace, Name: name}
				podsToDelete = append(podsToDelete, podKey)
				conflicts := c.AddVolume(logger, volumeName, podKey, "label1", v1.SELinuxChangePolicyMountOption, "csiDriver1")
				if len(conflicts) != 0 {
					// All volumes have the same labels and policy, there should be no conflicts
					t.Errorf("AddVolume %s/%s %s returned unexpected conflicts: %+v", namespace, name, volumeName, conflicts)
				}
			}
		}
	}
	t.Log("Before deleting all pods:")
	c.dump(dumpLogger)

	verifyReverseIndexConsistency(t, c)

	// Act: delete all pods
	for _, podKey := range podsToDelete {
		c.DeletePod(logger, podKey)
	}

	// Assert: the cache is empty
	if len(c.volumes) != 0 {
		t.Errorf("Expected cache to be empty, got %d volumes", len(c.volumes))
		c.dump(dumpLogger)
	}

	// Assert: the reverse index is also empty
	if len(c.podToVolumes) != 0 {
		t.Errorf("Expected reverse index to be empty, got %d pods", len(c.podToVolumes))
	}
	verifyReverseIndexConsistency(t, c)
}

type podWithVolume struct {
	podNamespace string
	podName      string
	volumeName   v1.UniqueVolumeName
	label        string
	changePolicy v1.PodSELinuxChangePolicy
}

func addReverseConflict(conflicts []Conflict) []Conflict {
	newConflicts := make([]Conflict, 0, len(conflicts)*2)
	for _, c := range conflicts {
		reversedConflict := Conflict{
			PropertyName:       c.PropertyName,
			EventReason:        c.EventReason,
			Pod:                c.OtherPod,
			PropertyValue:      c.OtherPropertyValue,
			OtherPod:           c.Pod,
			OtherPropertyValue: c.PropertyValue,
		}
		newConflicts = append(newConflicts, c, reversedConflict)
	}
	return newConflicts
}

// Test that AddVolume and GetConflicts return the same []conflict data
func TestVolumeCache_AddVolumeGetConflicts(t *testing.T) {
	existingPods := []podWithVolume{
		{
			podNamespace: "ns1",
			podName:      "pod1-mountOption",
			volumeName:   "vol1",
			label:        "system_u:system_r:label1",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns2",
			podName:      "pod2-recursive",
			volumeName:   "vol2",
			label:        "", // labels on volumes with Recursive policy are cleared, we don't want the controller to report label conflicts on them
			changePolicy: v1.SELinuxChangePolicyRecursive,
		},
		{
			podNamespace: "ns3",
			podName:      "pod3-1",
			volumeName:   "vol3", // vol3 is used by 2 pods with the same label + recursive policy
			label:        "",     // labels on volumes with Recursive policy are cleared, we don't want the controller to report label conflicts on them
			changePolicy: v1.SELinuxChangePolicyRecursive,
		},
		{
			podNamespace: "ns3",
			podName:      "pod3-2",
			volumeName:   "vol3", // vol3 is used by 2 pods with the same label + recursive policy
			label:        "",     // labels on volumes with Recursive policy are cleared, we don't want the controller to report label conflicts on them
			changePolicy: v1.SELinuxChangePolicyRecursive,
		},
		{
			podNamespace: "ns4",
			podName:      "pod4-1",
			volumeName:   "vol4", // vol4 is used by 2 pods with the same label + mount policy
			label:        "system_u:system_r:label4",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns4",
			podName:      "pod4-2",
			volumeName:   "vol4", // vol4 is used by 2 pods with the same label + mount policy
			label:        "system_u:system_r:label4",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns5",
			podName:      "pod5",
			volumeName:   "vol5", // vol5 has no user and role
			label:        "::label5",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns6",
			podName:      "pod6",
			volumeName:   "vol6", // vol6 has no user
			label:        ":system_r:label6",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns7",
			podName:      "pod7",
			volumeName:   "vol7", // vol7 has no user and role, but has categories
			label:        "::label7:c0,c1",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
		{
			podNamespace: "ns8",
			podName:      "pod8",
			volumeName:   "vol8", // vol has no label
			label:        "",
			changePolicy: v1.SELinuxChangePolicyMountOption,
		},
	}

	tests := []struct {
		name              string
		initialPods       []podWithVolume
		podToAdd          podWithVolume
		expectedConflicts []Conflict // conflicts for the other direction (pod, otherPod = otherPod, pod) will be added automatically
	}{
		{
			name:        "new volume in empty cache",
			initialPods: nil,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol-new",
				label:        "system_u:system_r:label-new",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: nil,
		},
		{
			name:        "new volume",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol-new",
				label:        "system_u:system_r:label-new",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: nil,
		},
		{
			name:        "existing volume in a new pod with existing policy and label",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol1",
				label:        "system_u:system_r:label1",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: nil,
		},
		{
			name:        "existing volume in a new pod with existing policy and new conflicting label",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol1",
				label:        "system_u:system_r:label-new",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: "testns", Name: "testpod"},
					PropertyValue:      "system_u:system_r:label-new",
					OtherPod:           cache.ObjectName{Namespace: "ns1", Name: "pod1-mountOption"},
					OtherPropertyValue: "system_u:system_r:label1",
				},
			},
		},
		{
			name:        "existing volume in a new pod with new conflicting policy",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol1",
				label:        "",
				changePolicy: v1.SELinuxChangePolicyRecursive,
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxChangePolicy",
					EventReason:        "SELinuxChangePolicyConflict",
					Pod:                cache.ObjectName{Namespace: "testns", Name: "testpod"},
					PropertyValue:      "Recursive",
					OtherPod:           cache.ObjectName{Namespace: "ns1", Name: "pod1-mountOption"},
					OtherPropertyValue: "MountOption",
				},
			},
		},
		{
			name:        "existing pod is replaced with different non-conflicting policy and label",
			initialPods: existingPods,
			podToAdd: podWithVolume{

				podNamespace: "ns2",
				podName:      "pod2-recursive",
				volumeName:   "vol2",                            // there is no other pod that uses vol2 -> change of policy and label is possible
				label:        "system_u:system_r:label-new",     // was label2 in the original pod2
				changePolicy: v1.SELinuxChangePolicyMountOption, // was Recursive in the original pod2
			},
			expectedConflicts: nil,
		},
		{
			name:        "existing pod is replaced with conflicting policy",
			initialPods: existingPods,
			podToAdd: podWithVolume{

				podNamespace: "ns3",
				podName:      "pod3-1",
				volumeName:   "vol3",                            // vol3 is used by pod3-2 with label3 and Recursive policy
				label:        "system_u:system_r:label-new",     // Technically, it's not possible to change a label of an existing pod, but we still check for conflicts
				changePolicy: v1.SELinuxChangePolicyMountOption, // ChangePolicy change can happen when CSIDriver is updated from SELinuxMount: false to SELinuxMount: true
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxChangePolicy",
					EventReason:        "SELinuxChangePolicyConflict",
					Pod:                cache.ObjectName{Namespace: "ns3", Name: "pod3-1"},
					PropertyValue:      "MountOption",
					OtherPod:           cache.ObjectName{Namespace: "ns3", Name: "pod3-2"},
					OtherPropertyValue: "Recursive",
				},
			},
		},
		{
			name:        "existing volume in a new pod with existing policy and new incomparable label (missing user and role)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol5",
				label:        "system_u:system_r:label5",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{},
		},
		{
			name:        "existing volume in a new pod with conflicting policy with incomparable parts",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol5",
				label:        "::label6",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: "testns", Name: "testpod"},
					PropertyValue:      "::label6",
					OtherPod:           cache.ObjectName{Namespace: "ns5", Name: "pod5"},
					OtherPropertyValue: "::label5",
				},
			},
		},
		{
			name:        "existing volume in a new pod with existing policy and new incomparable label (missing user)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol6",
				label:        "system_u::label6",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{},
		},
		{
			name:        "existing volume in a new pod with existing policy and new comparable label (missing categories)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol7",
				label:        "system_u:system_r:label7",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: "testns", Name: "testpod"},
					PropertyValue:      "system_u:system_r:label7",
					OtherPod:           cache.ObjectName{Namespace: "ns7", Name: "pod7"},
					OtherPropertyValue: "::label7:c0,c1",
				},
			},
		},
		{
			name:        "existing volume in a new pod with existing policy and new incomparable label (missing everything)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol8",
				label:        "system_u:system_r:label8",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{},
		},
		{
			name:        "existing volume in a new pod with existing policy and new comparable label (missing everything but categories)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol8",
				label:        "system_u:system_r:label8:c0,c1",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: "testns", Name: "testpod"},
					PropertyValue:      "system_u:system_r:label8:c0,c1",
					OtherPod:           cache.ObjectName{Namespace: "ns8", Name: "pod8"},
					OtherPropertyValue: "",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, dumpLogger := getTestLoggers(t)
			// Arrange: add initial pods to the cache
			seLinuxTranslator := &translator.ControllerSELinuxTranslator{}
			c := NewVolumeLabelCache(seLinuxTranslator).(*volumeCache)
			for _, podToAdd := range tt.initialPods {
				conflicts := c.AddVolume(logger, podToAdd.volumeName, cache.ObjectName{Namespace: podToAdd.podNamespace, Name: podToAdd.podName}, podToAdd.label, podToAdd.changePolicy, "csiDriver1")
				if len(conflicts) != 0 {
					t.Errorf("Initial AddVolume %s/%s %s returned unexpected conflicts: %+v", podToAdd.podNamespace, podToAdd.podName, podToAdd.volumeName, conflicts)
				}
			}

			// Act
			conflicts := c.AddVolume(logger, tt.podToAdd.volumeName, cache.ObjectName{Namespace: tt.podToAdd.podNamespace, Name: tt.podToAdd.podName}, tt.podToAdd.label, tt.podToAdd.changePolicy, "csiDriver1")

			// Assert
			expectedConflicts := addReverseConflict(tt.expectedConflicts)
			sortConflicts(conflicts)
			sortConflicts(expectedConflicts)
			if !reflect.DeepEqual(conflicts, expectedConflicts) {
				t.Errorf("AddVolume returned unexpected conflicts: %+v", conflicts)
				t.Logf("Expected conflicts: %+v", expectedConflicts)
				c.dump(dumpLogger)
			}
			// Expect the pod + volume to be present in the cache
			volume, ok := c.volumes[tt.podToAdd.volumeName]
			if !ok {
				t.Errorf("volume %s is not present in the cache", tt.podToAdd.volumeName)
			}
			podKey := cache.ObjectName{Namespace: tt.podToAdd.podNamespace, Name: tt.podToAdd.podName}
			existingInfo, ok := volume.pods[podKey]
			if !ok {
				t.Errorf("pod %s is not present in the cache", podKey)
			}
			expectedPodInfo := podInfo{
				seLinuxLabel: tt.podToAdd.label,
				seLinuxParts: parse.ParseSELinuxLabel(tt.podToAdd.label),
				changePolicy: tt.podToAdd.changePolicy,
			}
			if !reflect.DeepEqual(existingInfo, expectedPodInfo) {
				t.Errorf("pod %s has unexpected info: %+v", podKey, existingInfo)
			}

			// Verify reverse index consistency
			verifyReverseIndexConsistency(t, c)

			// Verify that GetConflicts returns the same conflicts
			receivedConflicts := c.GetConflicts(logger)
			sortConflicts(receivedConflicts)
			if !reflect.DeepEqual(receivedConflicts, expectedConflicts) {
				t.Errorf("GetConflicts returned unexpected conflicts: %+v", receivedConflicts)
				c.dump(dumpLogger)
			}
		})
	}
}

// Test that conflicts are tracked per-volume: a pod with conflicts on
// multiple volumes retains all of them after successive AddVolume calls.
func TestVolumeCache_MultiVolumeConflicts(t *testing.T) {
	logger, _ := getTestLoggers(t)
	seLinuxTranslator := &translator.ControllerSELinuxTranslator{}
	c := NewVolumeLabelCache(seLinuxTranslator).(*volumeCache)

	podA := cache.ObjectName{Namespace: "ns", Name: "podA"}
	podB := cache.ObjectName{Namespace: "ns", Name: "podB"}
	podC := cache.ObjectName{Namespace: "ns", Name: "podC"}

	// podB uses vol1 with label1
	c.AddVolume(logger, "vol1", podB, "system_u:system_r:labelB", v1.SELinuxChangePolicyMountOption, "driver1")
	// podC uses vol2 with label2
	c.AddVolume(logger, "vol2", podC, "system_u:system_r:labelC", v1.SELinuxChangePolicyMountOption, "driver1")

	// podA uses vol1 with a different label (conflict with podB)
	conflicts1 := c.AddVolume(logger, "vol1", podA, "system_u:system_r:labelA", v1.SELinuxChangePolicyMountOption, "driver1")
	if len(conflicts1) == 0 {
		t.Fatal("Expected conflicts on vol1 between podA and podB")
	}

	// podA also uses vol2 with a different label (conflict with podC)
	conflicts2 := c.AddVolume(logger, "vol2", podA, "system_u:system_r:labelA", v1.SELinuxChangePolicyMountOption, "driver1")
	if len(conflicts2) == 0 {
		t.Fatal("Expected conflicts on vol2 between podA and podC")
	}

	// GetConflicts must return conflicts from BOTH volumes
	allConflicts := c.GetConflicts(logger)
	expectedCount := len(conflicts1) + len(conflicts2)
	if len(allConflicts) != expectedCount {
		t.Errorf("GetConflicts returned %d conflicts, expected %d (vol1: %d + vol2: %d)",
			len(allConflicts), expectedCount, len(conflicts1), len(conflicts2))
	}

	// After deleting podA, all conflicts should be gone
	c.DeletePod(logger, podA)
	remaining := c.GetConflicts(logger)
	if len(remaining) != 0 {
		t.Errorf("Expected no conflicts after deleting podA, got %d: %+v", len(remaining), remaining)
	}

	// Verify deduplication: podD and podE conflict on two volumes with the same labels.
	// Identical Conflict entries from different volumes must be deduplicated by GetConflicts.
	podD := cache.ObjectName{Namespace: "ns", Name: "podD"}
	podE := cache.ObjectName{Namespace: "ns", Name: "podE"}

	c.AddVolume(logger, "vol3", podD, "system_u:system_r:labelD", v1.SELinuxChangePolicyMountOption, "driver1")
	c.AddVolume(logger, "vol4", podD, "system_u:system_r:labelD", v1.SELinuxChangePolicyMountOption, "driver1")

	conflictsVol3 := c.AddVolume(logger, "vol3", podE, "system_u:system_r:labelE", v1.SELinuxChangePolicyMountOption, "driver1")
	conflictsVol4 := c.AddVolume(logger, "vol4", podE, "system_u:system_r:labelE", v1.SELinuxChangePolicyMountOption, "driver1")

	if len(conflictsVol3) != len(conflictsVol4) {
		t.Fatalf("Expected same number of conflicts from vol3 and vol4 (%d vs %d)", len(conflictsVol3), len(conflictsVol4))
	}
	if len(conflictsVol3) == 0 {
		t.Fatal("Expected conflicts between podD and podE")
	}

	allConflicts = c.GetConflicts(logger)
	deCount := 0
	for _, conflict := range allConflicts {
		if conflict.Pod == podD || conflict.Pod == podE || conflict.OtherPod == podD || conflict.OtherPod == podE {
			deCount++
		}
	}
	if deCount != len(conflictsVol3) {
		t.Errorf("Expected %d deduplicated conflicts for podD/podE (from 2 volumes), got %d", len(conflictsVol3), deCount)
	}
}

func TestVolumeCache_DeletePodConflicts(t *testing.T) {
	podA := cache.ObjectName{Namespace: "ns", Name: "podA"}
	podB := cache.ObjectName{Namespace: "ns", Name: "podB"}
	podC := cache.ObjectName{Namespace: "ns", Name: "podC"}
	podD := cache.ObjectName{Namespace: "ns", Name: "podD"}

	tests := []struct {
		name string
		// Pods to add before deletion.
		initialPods []podWithVolume
		// Pod to delete.
		podToDelete cache.ObjectName
		// If true, delete the pod a second time to verify idempotency.
		deleteTwice bool
		// Pod pairs that must still have symmetric conflicts after deletion.
		// Each pair [2]cache.ObjectName expects both (A→B) and (B→A) to be present.
		expectedSurvivingPairs [][2]cache.ObjectName
	}{
		{
			name: "delete one of two conflicting pods clears all conflicts",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podA,
			expectedSurvivingPairs: nil,
		},
		{
			name: "delete non-conflicting pod preserves existing conflicts",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podC", volumeName: "vol2", label: "system_u:system_r:labelC", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podC,
			expectedSurvivingPairs: [][2]cache.ObjectName{{podA, podB}},
		},
		{
			name: "three pods on same volume delete one leaves remaining pair conflict",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podC", volumeName: "vol1", label: "system_u:system_r:labelC", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podA,
			expectedSurvivingPairs: [][2]cache.ObjectName{{podB, podC}},
		},
		{
			name: "delete pod with conflicts on multiple volumes",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podC", volumeName: "vol2", label: "system_u:system_r:labelC", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podA", volumeName: "vol2", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podA,
			expectedSurvivingPairs: nil,
		},
		{
			name: "delete pod preserves conflicts on unrelated volumes",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podC", volumeName: "vol2", label: "system_u:system_r:labelC", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podD", volumeName: "vol2", label: "system_u:system_r:labelD", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podA,
			expectedSurvivingPairs: [][2]cache.ObjectName{{podC, podD}},
		},
		{
			name: "delete pod that was already deleted is a no-op",
			initialPods: []podWithVolume{
				{podNamespace: "ns", podName: "podA", volumeName: "vol1", label: "system_u:system_r:labelA", changePolicy: v1.SELinuxChangePolicyMountOption},
				{podNamespace: "ns", podName: "podB", volumeName: "vol1", label: "system_u:system_r:labelB", changePolicy: v1.SELinuxChangePolicyMountOption},
			},
			podToDelete:            podA,
			deleteTwice:            true,
			expectedSurvivingPairs: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := getTestLoggers(t)
			seLinuxTranslator := &translator.ControllerSELinuxTranslator{}
			c := NewVolumeLabelCache(seLinuxTranslator).(*volumeCache)

			for _, pod := range tt.initialPods {
				c.AddVolume(logger, pod.volumeName, cache.ObjectName{Namespace: pod.podNamespace, Name: pod.podName}, pod.label, pod.changePolicy, "driver1")
			}

			c.DeletePod(logger, tt.podToDelete)
			if tt.deleteTwice {
				c.DeletePod(logger, tt.podToDelete)
			}

			remaining := c.GetConflicts(logger)

			// Deleted pod must not appear in any conflict
			for _, conflict := range remaining {
				if conflict.Pod == tt.podToDelete || conflict.OtherPod == tt.podToDelete {
					t.Errorf("found conflict involving deleted pod %s: %+v", tt.podToDelete, conflict)
				}
			}

			// Verify each expected surviving pair exists in both directions
			for _, pair := range tt.expectedSurvivingPairs {
				hasForward := false
				hasReverse := false
				for _, conflict := range remaining {
					if conflict.Pod == pair[0] && conflict.OtherPod == pair[1] {
						hasForward = true
					}
					if conflict.Pod == pair[1] && conflict.OtherPod == pair[0] {
						hasReverse = true
					}
				}
				if !hasForward || !hasReverse {
					t.Errorf("expected symmetric conflict between %s and %s, got %+v", pair[0], pair[1], remaining)
				}
			}

			// If no pairs are expected, there should be no conflicts at all
			if len(tt.expectedSurvivingPairs) == 0 && len(remaining) != 0 {
				t.Errorf("expected no conflicts, got %+v", remaining)
			}

			verifyReverseIndexConsistency(t, c)
		})
	}
}

func TestVolumeCache_GetPodsForCSIDriver(t *testing.T) {
	seLinuxTranslator := &translator.ControllerSELinuxTranslator{}
	c := NewVolumeLabelCache(seLinuxTranslator).(*volumeCache)
	logger, dumpLogger := getTestLoggers(t)

	existingPods := map[string][]podWithVolume{
		"csiDriver1": {
			{
				podNamespace: "ns1",
				podName:      "pod1-mountOption",
				volumeName:   "vol1",
				label:        "label1",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
		},
		"csiDriver2": {
			{
				podNamespace: "ns2",
				podName:      "pod2-recursive",
				volumeName:   "vol2",
				label:        "label2",
				changePolicy: v1.SELinuxChangePolicyRecursive,
			},
			{
				podNamespace: "ns3",
				podName:      "pod3-1",
				volumeName:   "vol3", // vol3 is used by 2 pods with the same label + recursive policy
				label:        "label3",
				changePolicy: v1.SELinuxChangePolicyRecursive,
			},
			{
				podNamespace: "ns3",
				podName:      "pod3-2",
				volumeName:   "vol3", // vol3 is used by 2 pods with the same label + recursive policy
				label:        "label3",
				changePolicy: v1.SELinuxChangePolicyRecursive,
			},
		},
		"csiDriver3": {
			{
				podNamespace: "ns4",
				podName:      "pod4-1",
				volumeName:   "vol4", // vol4 is used by 2 pods with the same label + mount policy
				label:        "label4",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			{
				podNamespace: "ns4",
				podName:      "pod4-2",
				volumeName:   "vol4", // vol4 is used by 2 pods with the same label + mount policy
				label:        "label4",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
		},
	}

	for csiDriverName, pods := range existingPods {
		for _, podToAdd := range pods {
			conflicts := c.AddVolume(logger, podToAdd.volumeName, cache.ObjectName{Namespace: podToAdd.podNamespace, Name: podToAdd.podName}, podToAdd.label, podToAdd.changePolicy, csiDriverName)
			if len(conflicts) != 0 {
				t.Errorf("Initial AddVolume %s/%s %s returned unexpected conflicts: %+v", podToAdd.podNamespace, podToAdd.podName, podToAdd.volumeName, conflicts)
			}
		}
	}

	// Act
	expectedPods := map[string][]cache.ObjectName{
		"csiDriver1": {
			{Namespace: "ns1", Name: "pod1-mountOption"},
		},
		"csiDriver2": {
			{Namespace: "ns2", Name: "pod2-recursive"},
			{Namespace: "ns3", Name: "pod3-1"},
			{Namespace: "ns3", Name: "pod3-2"},
		},
		"csiDriver3": {
			{Namespace: "ns4", Name: "pod4-1"},
			{Namespace: "ns4", Name: "pod4-2"},
		},
		"csiDriver4": nil, // totally unknown CSI driver
	}
	for csiDriverName, expectedPodsForDriver := range expectedPods {
		podsForDriver := c.GetPodsForCSIDriver(csiDriverName)
		sort.Slice(podsForDriver, func(i, j int) bool {
			return podsForDriver[i].String() < podsForDriver[j].String()
		})
		if !reflect.DeepEqual(podsForDriver, expectedPodsForDriver) {
			t.Errorf("GetPodsForCSIDriver(%s) returned unexpected pods: %+v", csiDriverName, podsForDriver)
			c.dump(dumpLogger)
		}
	}
}
