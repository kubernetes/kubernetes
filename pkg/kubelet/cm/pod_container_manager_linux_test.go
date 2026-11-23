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

	libcontainercgroups "github.com/opencontainers/cgroups"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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

// recordingCgroupManager captures the CgroupConfig passed to Create so the
// test can inspect the ResourceParameters without touching the real cgroup fs.
type recordingCgroupManager struct {
	unsupportedCgroupManager // satisfies all unneeded interface methods
	created []*CgroupConfig
}

func (r *recordingCgroupManager) Create(_ klog.Logger, cfg *CgroupConfig) error {
	r.created = append(r.created, cfg)
	return nil
}
func (r *recordingCgroupManager) Exists(CgroupName) bool    { return false }
func (r *recordingCgroupManager) Name(n CgroupName) string  { return n.ToCgroupfs() }
func (r *recordingCgroupManager) CgroupName(s string) CgroupName {
	return CgroupName([]string{s})
}

// TestEnsureExists_NoSwap_SetsPodCgroupSwapMaxZero verifies that EnsureExists
// writes memory.swap.max=0 in the pod cgroup Unified map when memorySwapBehavior
// is NoSwap on a cgroup v2 node.
//
// Before this fix the pod cgroup was left at the kernel default (max), which
// allowed the sandbox (pause process, Kata VM, etc.) to swap even though the
// per-container memory.swap.max was 0.
func TestEnsureExists_NoSwap_SetsPodCgroupSwapMaxZero(t *testing.T) {
	// This test relies on cgroup v2 mode; skip if the kernel is cgroup v1 only.
	if !libcontainercgroups.IsCgroup2UnifiedMode() {
		t.Skip("cgroup v2 not available on this host")
	}

	logger, _ := ktesting.NewTestContext(t)
	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "test-uid-noswap"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "app",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
						Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
					},
				},
			},
		},
	}

	tests := []struct {
		name             string
		swapBehavior     string
		wantSwapMaxZero  bool
	}{
		{
			name:            "NoSwap sets memory.swap.max=0 on pod cgroup",
			swapBehavior:    string(kubetypes.NoSwap),
			wantSwapMaxZero: true,
		},
		{
			name:            "LimitedSwap leaves memory.swap.max unset on pod cgroup",
			swapBehavior:    string(kubetypes.LimitedSwap),
			wantSwapMaxZero: false,
		},
		{
			name:            "empty swap behavior leaves memory.swap.max unset on pod cgroup",
			swapBehavior:    "",
			wantSwapMaxZero: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &recordingCgroupManager{}
			pcm := &podContainerManagerImpl{
				cgroupManager:      NewCgroupManager(logger, nil, "cgroupfs"),
				qosContainersInfo:  qosContainersInfo,
				enforceCPULimits:   true,
				memorySwapBehavior: tt.swapBehavior,
			}
			// Override the cgroup manager with our recorder after construction so
			// IsPodCgroup / Name still work via the real manager, but Create is captured.
			pcm.cgroupManager = rec

			err := pcm.EnsureExists(logger, pod)
			require.NoError(t, err)
			require.Len(t, rec.created, 1, "expected exactly one cgroup Create call")

			cfg := rec.created[0]
			swapVal, hasSwap := cfg.ResourceParameters.Unified[Cgroup2MaxSwapFilename]
			if tt.wantSwapMaxZero {
				require.True(t, hasSwap, "expected memory.swap.max in Unified map for %s", tt.swapBehavior)
				require.Equal(t, "0", swapVal, "expected memory.swap.max=0 for %s", tt.swapBehavior)
			} else {
				require.False(t, hasSwap, "expected memory.swap.max absent from Unified map for %s", tt.swapBehavior)
			}
		})
	}
}
