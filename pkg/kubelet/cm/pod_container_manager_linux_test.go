//go:build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestIsCgroupPod(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}
	podUID := types.UID("123")
	testCases := []struct {
		input          CgroupName
		expectedResult bool
		expectedUID    types.UID
	}{
		{
			input:          RootCgroupName,
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.Guaranteed),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.Guaranteed, GetPodCgroupNameSuffix(podUID)),
			expectedResult: true,
			expectedUID:    podUID,
		},
		{
			input:          NewCgroupName(qosContainersInfo.Guaranteed, GetPodCgroupNameSuffix(podUID), "container.scope"),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.Burstable),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.Burstable, GetPodCgroupNameSuffix(podUID)),
			expectedResult: true,
			expectedUID:    podUID,
		},
		{
			input:          NewCgroupName(qosContainersInfo.Burstable, GetPodCgroupNameSuffix(podUID), "container.scope"),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.BestEffort),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(qosContainersInfo.BestEffort, GetPodCgroupNameSuffix(podUID)),
			expectedResult: true,
			expectedUID:    podUID,
		},
		{
			input:          NewCgroupName(qosContainersInfo.BestEffort, GetPodCgroupNameSuffix(podUID), "container.scope"),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(RootCgroupName, "system"),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			input:          NewCgroupName(RootCgroupName, "system", "kubelet"),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
		{
			// contains reserved word "pod" in cgroup name
			input:          NewCgroupName(RootCgroupName, GetPodCgroupNameSuffix("this-uid-contains-reserved-word-pod")),
			expectedResult: false,
			expectedUID:    types.UID(""),
		},
	}
	for _, cgroupDriver := range []string{"cgroupfs", "systemd"} {
		pcm := &podContainerManagerImpl{
			cgroupManager:     NewCgroupManager(logger, nil, cgroupDriver),
			enforceCPULimits:  true,
			qosContainersInfo: qosContainersInfo,
		}
		for _, testCase := range testCases {
			// Give the right cgroup structure based on whether systemd is enabled.
			var name string
			if cgroupDriver == "systemd" {
				name = testCase.input.ToSystemd()
			} else {
				name = testCase.input.ToCgroupfs()
			}
			// check if this is a pod or not with the literal cgroupfs input
			result, resultUID := pcm.IsPodCgroup(name)
			if result != testCase.expectedResult {
				t.Errorf("Unexpected result for driver: %v, input: %v, expected: %v, actual: %v", cgroupDriver, testCase.input, testCase.expectedResult, result)
			}
			if resultUID != testCase.expectedUID {
				t.Errorf("Unexpected result for driver: %v, input: %v, expected: %v, actual: %v", cgroupDriver, testCase.input, testCase.expectedUID, resultUID)
			}

		}
	}
}

func TestGetPodContainerName(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	newGuaranteedPodWithUID := func(uid types.UID) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: uid,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000m"),
								v1.ResourceMemory: resource.MustParse("1G"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000m"),
								v1.ResourceMemory: resource.MustParse("1G"),
							},
						},
					},
				},
			},
		}
	}
	newBurstablePodWithUID := func(uid types.UID) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: uid,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000m"),
								v1.ResourceMemory: resource.MustParse("1G"),
							},
						},
					},
				},
			},
		}
	}
	newBestEffortPodWithUID := func(uid types.UID) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID: uid,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "container",
					},
				},
			},
		}
	}

	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}

	type fields struct {
		cgroupManager CgroupManager
	}
	type args struct {
		pod *v1.Pod
	}

	tests := []struct {
		name                string
		fields              fields
		args                args
		wantCgroupName      CgroupName
		wantLiteralCgroupfs string
	}{
		{
			name: "pod with qos guaranteed and cgroupfs",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "cgroupfs"),
			},
			args: args{
				pod: newGuaranteedPodWithUID("fake-uid-1"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.Guaranteed, "podfake-uid-1"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.Guaranteed, "podfake-uid-1").ToCgroupfs(),
		}, {
			name: "pod with qos guaranteed and systemd",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "systemd"),
			},
			args: args{
				pod: newGuaranteedPodWithUID("fake-uid-2"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.Guaranteed, "podfake-uid-2"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.Guaranteed, "podfake-uid-2").ToSystemd(),
		}, {
			name: "pod with qos burstable and cgroupfs",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "cgroupfs"),
			},
			args: args{
				pod: newBurstablePodWithUID("fake-uid-3"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.Burstable, "podfake-uid-3"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.Burstable, "podfake-uid-3").ToCgroupfs(),
		}, {
			name: "pod with qos burstable and systemd",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "systemd"),
			},
			args: args{
				pod: newBurstablePodWithUID("fake-uid-4"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.Burstable, "podfake-uid-4"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.Burstable, "podfake-uid-4").ToSystemd(),
		}, {
			name: "pod with qos best-effort and cgroupfs",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "cgroupfs"),
			},
			args: args{
				pod: newBestEffortPodWithUID("fake-uid-5"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.BestEffort, "podfake-uid-5"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.BestEffort, "podfake-uid-5").ToCgroupfs(),
		}, {
			name: "pod with qos best-effort and systemd",
			fields: fields{
				cgroupManager: NewCgroupManager(logger, nil, "systemd"),
			},
			args: args{
				pod: newBestEffortPodWithUID("fake-uid-6"),
			},
			wantCgroupName:      NewCgroupName(qosContainersInfo.BestEffort, "podfake-uid-6"),
			wantLiteralCgroupfs: NewCgroupName(qosContainersInfo.BestEffort, "podfake-uid-6").ToSystemd(),
		},
	}

	for _, tt := range tests {
		pcm := &podContainerManagerImpl{
			cgroupManager:     tt.fields.cgroupManager,
			qosContainersInfo: qosContainersInfo,
		}

		t.Run(tt.name, func(t *testing.T) {
			actualCgroupName, actualLiteralCgroupfs := pcm.GetPodContainerName(tt.args.pod)
			require.Equalf(t, tt.wantCgroupName, actualCgroupName, "Unexpected cgroup name for pod with UID %s, container resources: %v", tt.args.pod.UID, tt.args.pod.Spec.Containers[0].Resources)
			require.Equalf(t, tt.wantLiteralCgroupfs, actualLiteralCgroupfs, "Unexpected literal cgroupfs for pod with UID %s, container resources: %v", tt.args.pod.UID, tt.args.pod.Spec.Containers[0].Resources)
		})
	}
}

// testQOSContainersInfo returns the default partition QoS roots under cgroupRoot.
func testQOSContainersInfo(cgroupRoot CgroupName) QOSContainersInfo {
	return QOSContainersInfo{
		Guaranteed: cgroupRoot,
		Burstable:  NewCgroupName(cgroupRoot, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(cgroupRoot, strings.ToLower(string(v1.PodQOSBestEffort))),
	}
}

func TestGetPodContainerNameWithSystemPartition(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cgroupRoot := NewCgroupName(RootCgroupName, "kubepods")
	defaultQOS := testQOSContainersInfo(cgroupRoot)
	systemQOS := systemPartitionQOSContainersInfo(cgroupRoot)
	partition := &SystemPartitionConfig{Namespaces: sets.New("kube-system")}
	burstable := newPartitionTestPod("kube-system", "100m/64Mi", "")
	guaranteed := newPartitionTestPod("kube-system", "100m/64Mi", "100m/64Mi")
	userPod := newPartitionTestPod(metav1.NamespaceDefault, "100m/64Mi", "")

	tests := []struct {
		name            string
		systemPartition *SystemPartitionConfig
		pod             *v1.Pod
		wantParent      CgroupName
	}{
		{
			name:       "no partition",
			pod:        burstable,
			wantParent: NewCgroupName(RootCgroupName, "kubepods", "burstable"),
		},
		{
			name:            "pod of the partition",
			systemPartition: partition,
			pod:             burstable,
			wantParent:      NewCgroupName(RootCgroupName, "kubepods", "system", "burstable"),
		},
		{
			name:            "guaranteed pod of the partition sits under the partition root",
			systemPartition: partition,
			pod:             guaranteed,
			wantParent:      NewCgroupName(RootCgroupName, "kubepods", "system"),
		},
		{
			name:            "pod outside the partition",
			systemPartition: partition,
			pod:             userPod,
			wantParent:      NewCgroupName(RootCgroupName, "kubepods", "burstable"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &podContainerManagerImpl{
				cgroupManager:           NewCgroupManager(logger, nil, "cgroupfs"),
				qosContainersInfo:       defaultQOS,
				systemQOSContainersInfo: systemQOS,
				systemPartition:         tt.systemPartition,
			}
			gotName, _ := m.GetPodContainerName(tt.pod)
			require.Equal(t, NewCgroupName(tt.wantParent, GetPodCgroupNameSuffix(tt.pod.UID)), gotName)
		})
	}
}

func TestQOSContainerRoots(t *testing.T) {
	cgroupRoot := NewCgroupName(RootCgroupName, "kubepods")
	defaultQOS := testQOSContainersInfo(cgroupRoot)
	systemQOS := systemPartitionQOSContainersInfo(cgroupRoot)

	tests := []struct {
		name            string
		systemPartition *SystemPartitionConfig
	}{
		{
			name:            "partition configured",
			systemPartition: &SystemPartitionConfig{Namespaces: sets.New("kube-system")},
		},
		{
			// The system partition roots stay in the list so that pod cgroups left
			// behind after the feature is turned off are still reclaimed.
			name:            "partition not configured",
			systemPartition: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &podContainerManagerImpl{
				qosContainersInfo:       defaultQOS,
				systemQOSContainersInfo: systemQOS,
				systemPartition:         tt.systemPartition,
			}
			require.ElementsMatch(t, []CgroupName{
				defaultQOS.BestEffort, defaultQOS.Burstable, defaultQOS.Guaranteed,
				systemQOS.BestEffort, systemQOS.Burstable, systemQOS.Guaranteed,
			}, m.qosContainerRoots())
		})
	}
}

func TestIsPodCgroupInSystemPartition(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cgroupRoot := NewCgroupName(RootCgroupName, "kubepods")
	systemQOS := systemPartitionQOSContainersInfo(cgroupRoot)
	podUID := types.UID("123")

	testCases := []struct {
		name    string
		input   CgroupName
		wantUID types.UID
	}{
		{name: "partition root", input: systemQOS.Guaranteed},
		{name: "partition burstable root", input: systemQOS.Burstable},
		{name: "partition besteffort root", input: systemQOS.BestEffort},
		{
			name:    "guaranteed pod",
			input:   NewCgroupName(systemQOS.Guaranteed, GetPodCgroupNameSuffix(podUID)),
			wantUID: podUID,
		},
		{
			name:    "besteffort pod",
			input:   NewCgroupName(systemQOS.BestEffort, GetPodCgroupNameSuffix(podUID)),
			wantUID: podUID,
		},
		{
			// The partition's roots are as deep as the default partition's, and
			// must not keep its pods from being found.
			name:    "burstable pod outside the partition",
			input:   NewCgroupName(cgroupRoot, "burstable", GetPodCgroupNameSuffix(podUID)),
			wantUID: podUID,
		},
		{
			name:  "container of a pod in the partition",
			input: NewCgroupName(systemQOS.Burstable, GetPodCgroupNameSuffix(podUID), "container.scope"),
		},
	}
	for _, cgroupDriver := range []string{"cgroupfs", "systemd"} {
		// Without a configured partition, so that the orphan cleanup still
		// finds the pod cgroups an earlier one left behind.
		m := &podContainerManagerImpl{
			cgroupManager:           NewCgroupManager(logger, nil, cgroupDriver),
			qosContainersInfo:       testQOSContainersInfo(cgroupRoot),
			systemQOSContainersInfo: systemQOS,
		}
		for _, tc := range testCases {
			t.Run(cgroupDriver+"/"+tc.name, func(t *testing.T) {
				name := tc.input.ToCgroupfs()
				if cgroupDriver == "systemd" {
					name = tc.input.ToSystemd()
				}
				isPod, uid := m.IsPodCgroup(name)
				require.Equal(t, tc.wantUID != "", isPod)
				require.Equal(t, tc.wantUID, uid)
			})
		}
	}
}
