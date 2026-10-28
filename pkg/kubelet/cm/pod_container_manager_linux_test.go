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
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
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

// cpuWeightCgroupManager models the cpu.weight file of cgroup v2 pod cgroups:
// shares are converted to a weight on write and back to shares on read, using
// whatever scale is in effect at the time, like the real cgroup v2 manager.
type cpuWeightCgroupManager struct {
	fakeCgroupManager
	weights map[string]uint64
	writes  int
}

func (f *cpuWeightCgroupManager) Exists(name CgroupName) bool {
	_, ok := f.weights[name.ToCgroupfs()]
	return ok
}

func (f *cpuWeightCgroupManager) GetCgroupConfig(name CgroupName, _ v1.ResourceName) (*ResourceConfig, error) {
	shares := cpuWeightToCPUShares(f.weights[name.ToCgroupfs()])
	return &ResourceConfig{CPUShares: &shares}, nil
}

func (f *cpuWeightCgroupManager) SetCgroupConfig(_ klog.Logger, name CgroupName, rc *ResourceConfig) error {
	f.writes++
	f.weights[name.ToCgroupfs()] = getCPUWeight(rc.CPUShares)
	return nil
}

func (f *cpuWeightCgroupManager) Version() int { return 2 }

func TestRescaleCPUWeightAfterKubeletRestart(t *testing.T) {
	burstablePod := func(cpu string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: types.UID("pod-" + cpu)},
			Spec: v1.PodSpec{Containers: []v1.Container{{
				Name: "c",
				Resources: v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(cpu),
					v1.ResourceMemory: resource.MustParse("64Mi"),
				}},
			}}},
		}
	}

	tests := []struct {
		name string
		// capacityMilliCPU is the node CPU capacity seen by the restarted kubelet.
		capacityMilliCPU int64
		cpu              string
		// weight overrides the cpu.weight found in the pod cgroup after the
		// restart. Zero means the weight the previous kubelet wrote at the
		// fixed MaxShares scale.
		weight      uint64
		wantRewrite bool
	}{
		{name: "1 CPU pod on a 384 CPU node", capacityMilliCPU: 384000, cpu: "1", wantRewrite: true},
		{name: "300 CPU pod on a 384 CPU node", capacityMilliCPU: 384000, cpu: "300", wantRewrite: true},
		{name: "pod at node capacity keeps the maximum weight", capacityMilliCPU: 384000, cpu: "384"},
		{name: "pod at the minimum weight", capacityMilliCPU: 384000, cpu: "1m"},
		{name: "small node keeps the legacy scale", capacityMilliCPU: 64000, cpu: "1"},
		{name: "256 CPU node keeps the legacy scale", capacityMilliCPU: 256000, cpu: "300"},
		// 26 is the new-scale weight of 1 CPU on a 384 CPU node, i.e. a pod
		// created by the restarted kubelet itself.
		{name: "weight already at the new scale", capacityMilliCPU: 384000, cpu: "1", weight: 26},
		// A weight matching neither scale, e.g. an in-place resize that has
		// not been actuated yet, is left for the resize path.
		{name: "unrelated weight is left alone", capacityMilliCPU: 384000, cpu: "1", weight: 157},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			old := cgroupV2MaxShares.Load()
			t.Cleanup(func() { cgroupV2MaxShares.Store(old) })

			pod := burstablePod(tc.cpu)
			cgroups := &cpuWeightCgroupManager{weights: map[string]uint64{}}
			m := &podContainerManagerImpl{
				qosContainersInfo: QOSContainersInfo{Burstable: RootCgroupName},
				cgroupManager:     cgroups,
				cpuCFSQuotaPeriod: 100000,
			}
			podCgroup, _ := m.GetPodContainerName(pod)

			// The previous kubelet created the pod cgroup at the fixed scale.
			cgroupV2MaxShares.Store(MaxShares)
			allocated := ResourceConfigForPod(pod, false, 100000, false, "")
			legacyWeight := getCPUWeight(allocated.CPUShares)
			cgroups.weights[podCgroup.ToCgroupfs()] = legacyWeight
			if tc.weight != 0 {
				cgroups.weights[podCgroup.ToCgroupfs()] = tc.weight
			}
			before := cgroups.weights[podCgroup.ToCgroupfs()]

			// The kubelet restarts and scales to the node's capacity.
			SetCgroupV2CPUCapacity(tc.capacityMilliCPU)
			request := resource.MustParse(tc.cpu)
			shares := MilliCPUToCgroupShares(request.MilliValue())
			wantWeight := getCPUWeight(&shares)

			require.NoError(t, m.RescaleCPUWeight(logger, pod))
			got := cgroups.weights[podCgroup.ToCgroupfs()]
			if !tc.wantRewrite {
				require.Equal(t, before, got, "cpu.weight must not change")
				require.Zero(t, cgroups.writes)
				return
			}
			require.NotEqual(t, wantWeight, legacyWeight, "test case must change the weight")
			require.Equal(t, wantWeight, got)

			// The readback now matches the pod's request, so status.resources
			// reports the request instead of an inflated value.
			readback, err := m.GetPodCgroupConfig(pod, v1.ResourceCPU)
			require.NoError(t, err)
			require.True(t, CPUSharesEqualAfterV2RoundTrip(shares, *readback.CPUShares))

			// Later syncs find the new weight and do not write again.
			require.NoError(t, m.RescaleCPUWeight(logger, pod))
			require.Equal(t, 1, cgroups.writes)
		})
	}
}

func TestRescaleCPUWeightMissingCgroup(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	old := cgroupV2MaxShares.Load()
	t.Cleanup(func() { cgroupV2MaxShares.Store(old) })
	SetCgroupV2CPUCapacity(384000)

	cgroups := &cpuWeightCgroupManager{weights: map[string]uint64{}}
	m := &podContainerManagerImpl{qosContainersInfo: QOSContainersInfo{Burstable: RootCgroupName}, cgroupManager: cgroups}
	require.NoError(t, m.RescaleCPUWeight(logger, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "missing"}}))
	require.Zero(t, cgroups.writes)
}
