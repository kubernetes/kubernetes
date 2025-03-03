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

	// Act: delete all pods
	for _, podKey := range podsToDelete {
		c.DeletePod(logger, podKey)
	}

	// Assert: the cache is empty
	if len(c.volumes) != 0 {
		t.Errorf("Expected cache to be empty, got %d volumes", len(c.volumes))
		c.dump(dumpLogger)
	}
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

// Test AddVolume and SendConflicts together, they both provide []conflict with the same data
func TestVolumeCache_AddVolumeSendConflicts(t *testing.T) {
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
			name:        "existing volume in a new pod with existing policy and new incomparable label (missing categories)",
			initialPods: existingPods,
			podToAdd: podWithVolume{
				podNamespace: "testns",
				podName:      "testpod",
				volumeName:   "vol7",
				label:        "system_u:system_r:label7",
				changePolicy: v1.SELinuxChangePolicyMountOption,
			},
			expectedConflicts: []Conflict{},
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
				changePolicy: tt.podToAdd.changePolicy,
			}
			if !reflect.DeepEqual(existingInfo, expectedPodInfo) {
				t.Errorf("pod %s has unexpected info: %+v", podKey, existingInfo)
			}

			// Act again: get the conflicts via SendConflicts
			ch := make(chan Conflict)
			go func() {
				c.SendConflicts(logger, ch)
				close(ch)
			}()

			// Assert
			receivedConflicts := []Conflict{}
			for c := range ch {
				receivedConflicts = append(receivedConflicts, c)
			}
			sortConflicts(receivedConflicts)
			if !reflect.DeepEqual(receivedConflicts, expectedConflicts) {
				t.Errorf("SendConflicts returned unexpected conflicts: %+v", receivedConflicts)
				c.dump(dumpLogger)
			}
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
