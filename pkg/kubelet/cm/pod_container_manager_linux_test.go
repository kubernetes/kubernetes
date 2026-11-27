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
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"

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

func TestPodRequestsWritableCgroups(t *testing.T) {
	readWrite := v1.CgroupMountModeReadWrite
	readOnly := v1.CgroupMountModeReadOnly
	scWith := func(mode *v1.CgroupMountMode) *v1.SecurityContext {
		return &v1.SecurityContext{CgroupOptions: &v1.CgroupOptions{MountMode: mode}}
	}

	tests := []struct {
		name string
		pod  *v1.Pod
		want bool
	}{
		{
			name: "no security context",
			pod:  &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c"}}}},
			want: false,
		},
		{
			name: "read-only mount mode",
			pod:  &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c", SecurityContext: scWith(&readOnly)}}}},
			want: false,
		},
		{
			name: "read-write on a container",
			pod:  &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c", SecurityContext: scWith(&readWrite)}}}},
			want: true,
		},
		{
			name: "read-write on an init container",
			pod:  &v1.Pod{Spec: v1.PodSpec{InitContainers: []v1.Container{{Name: "i", SecurityContext: scWith(&readWrite)}}}},
			want: true,
		},
		{
			name: "read-write on an ephemeral container",
			pod: &v1.Pod{Spec: v1.PodSpec{EphemeralContainers: []v1.EphemeralContainer{{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", SecurityContext: scWith(&readWrite)},
			}}}},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, podRequestsWritableCgroups(tt.pod))
		})
	}
}

func TestGetPodPIDLimit(t *testing.T) {
	tests := []struct {
		name     string
		pod      *v1.Pod
		expected int64
	}{
		{
			name: "no pid limit set",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1"},
					},
				},
			},
			expected: 0,
		},
		{
			name: "nil pod resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources:  nil,
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			expected: 0,
		},
		{
			name: "pod-level pid limit set",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("2048"),
						},
					},
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			expected: 2048,
		},
		{
			name: "pod-level resources set but no pid",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
						},
					},
					Containers: []v1.Container{{Name: "c1"}},
				},
			},
			expected: 0,
		},
		{
			name: "multiple containers share single pod-level pid limit",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("4096"),
						},
					},
					Containers: []v1.Container{
						{Name: "app"},
						{Name: "sidecar"},
						{Name: "logger"},
					},
				},
			},
			expected: 4096,
		},
		{
			name: "multiple containers with init containers share pod-level pid limit",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("2048"),
						},
					},
					InitContainers: []v1.Container{
						{Name: "init1"},
					},
					Containers: []v1.Container{
						{Name: "app"},
						{Name: "sidecar"},
					},
				},
			},
			expected: 2048,
		},
		{
			name: "static pod (mirror pod) with pid limit",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-master-0",
					Namespace: "kube-system",
					Annotations: map[string]string{
						"kubernetes.io/config.mirror": "abc123",
					},
				},
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourcePID: resource.MustParse("4096"),
						},
					},
					Containers: []v1.Container{{Name: "kube-apiserver"}},
				},
			},
			expected: 4096,
		},
		{
			name: "static pod without pid limit",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "etcd-master-0",
					Namespace: "kube-system",
					Annotations: map[string]string{
						"kubernetes.io/config.mirror": "def456",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "etcd"}},
				},
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getPodPIDLimit(tt.pod)
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestEnsureExistsWritableCgroupLimits(t *testing.T) {
	tests := []struct {
		name                 string
		cgroupOptionsEnabled bool
		mountMode            v1.CgroupMountMode
		wantLimits           bool
	}{
		{
			name:                 "limits the pod cgroup when a container requests writable cgroups",
			cgroupOptionsEnabled: true,
			mountMode:            v1.CgroupMountModeReadWrite,
			wantLimits:           true,
		},
		{
			name:                 "no limits with CgroupOptions disabled",
			cgroupOptionsEnabled: false,
			mountMode:            v1.CgroupMountModeReadWrite,
		},
		{
			name:                 "no limits for a read-only mount",
			cgroupOptionsEnabled: true,
			mountMode:            v1.CgroupMountModeReadOnly,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.CgroupOptions, tt.cgroupOptionsEnabled)
			logger, _ := ktesting.NewTestContext(t)
			pod := &v1.Pod{
				Spec: v1.PodSpec{Containers: []v1.Container{{
					Name: "c",
					SecurityContext: &v1.SecurityContext{
						CgroupOptions: &v1.CgroupOptions{MountMode: &tt.mountMode},
					},
				}}},
			}
			fake := &fakeCgroupManager{}
			m := &podContainerManagerImpl{
				cgroupManager:       fake,
				podContainerManager: NewStubContainerManager(),
			}

			require.NoError(t, m.EnsureExists(logger, pod))
			require.Len(t, fake.created, 1)
			unified := fake.created[0].ResourceParameters.Unified
			if tt.wantLimits {
				require.Equal(t, "250", unified["cgroup.max.descendants"])
				require.Equal(t, "50", unified["cgroup.max.depth"])
				return
			}
			require.NotContains(t, unified, "cgroup.max.descendants")
			require.NotContains(t, unified, "cgroup.max.depth")
		})
	}
}

func TestEnsureExistsWithPodLevelSwap(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.PodLevelResources, true)

	origIsCgroup2 := isCgroup2UnifiedMode
	t.Cleanup(func() { isCgroup2UnifiedMode = origIsCgroup2 })

	const (
		gi             = 1024 * 1024 * 1024
		nodeMem  int64 = 40 * gi
		nodeSwap int64 = 5 * gi
	)

	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}

	containerStub := NewStubContainerManager()

	burstablePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod-swap-1"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
				Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("4Gi")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}
	ineligibleBurstablePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod-swap-2"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("2Gi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("2"), v1.ResourceMemory: resource.MustParse("2Gi")},
			},
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")},
						Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
					},
				},
			},
		},
	}

	containerOnlyBurstablePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod-swap-ctr-only"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
						Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("4Gi")},
					},
				},
			},
		},
	}
	plrCPUOnlyPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "pod-swap-plr-cpu"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}

	expectedSwapBytes, err := CalcSwapForBurstablePods(2*gi, nodeMem, nodeSwap)
	require.NoError(t, err)
	expectedSwap2Gi := strconv.FormatInt(expectedSwapBytes, 10)

	tests := []struct {
		name                      string
		pod                       *v1.Pod
		swapBehavior              kubetypes.SwapBehavior
		isCgroup2                 bool
		alreadyExists             bool
		nilSwapControllerCallback bool
		swapControllerAvailable   bool
		expectedSwapMax           *string
	}{
		{
			name:                    "LimitedSwap eligible pod sets proportional memory.swap.max",
			pod:                     burstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         &expectedSwap2Gi,
		},
		{
			name:                    "LimitedSwap ineligible pod (PodMemoryRequest == PodMemoryLimit) sets memory.swap.max to 0",
			pod:                     ineligibleBurstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         new("0"),
		},
		{
			name:                    "LimitedSwap pod with pod-level CPU only and no memory sets memory.swap.max to 0",
			pod:                     plrCPUOnlyPod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         new("0"),
		},
		{
			name:                    "NoSwap sets memory.swap.max to 0 on pod with pod-level resources",
			pod:                     burstablePod,
			swapBehavior:            kubetypes.NoSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         new("0"),
		},
		{
			name:                    "container-only pod under LimitedSwap omits pod cgroup memory.swap.max",
			pod:                     containerOnlyBurstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         nil,
		},
		{
			name:                    "container-only pod under NoSwap omits pod cgroup memory.swap.max",
			pod:                     containerOnlyBurstablePod,
			swapBehavior:            kubetypes.NoSwap,
			isCgroup2:               true,
			swapControllerAvailable: true,
			expectedSwapMax:         nil,
		},
		{
			name:                    "swap controller unavailable omits memory.swap.max",
			pod:                     burstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			swapControllerAvailable: false,
			expectedSwapMax:         nil,
		},
		{
			name:                      "nil getSwapControllerAvailable callback omits memory.swap.max",
			pod:                       burstablePod,
			swapBehavior:              kubetypes.LimitedSwap,
			isCgroup2:                 true,
			nilSwapControllerCallback: true,
			expectedSwapMax:           nil,
		},
		{
			name:                    "cgroup v1 omits memory.swap.max",
			pod:                     burstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               false,
			swapControllerAvailable: true,
			expectedSwapMax:         nil,
		},
		{
			name:                    "existing pod cgroup skips cgroup creation",
			pod:                     burstablePod,
			swapBehavior:            kubetypes.LimitedSwap,
			isCgroup2:               true,
			alreadyExists:           true,
			swapControllerAvailable: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isCgroup2UnifiedMode = func() bool { return tt.isCgroup2 }
			fakeCM := &fakeCgroupManager{exists: tt.alreadyExists}
			var swapAvailableFn func() bool
			if !tt.nilSwapControllerCallback {
				swapAvailableFn = func() bool { return tt.swapControllerAvailable }
			}
			pcm := &podContainerManagerImpl{
				qosContainersInfo:          qosContainersInfo,
				cgroupManager:              fakeCM,
				enforceCPULimits:           true,
				cpuCFSQuotaPeriod:          QuotaPeriod,
				podContainerManager:        containerStub,
				memorySwapBehavior:         tt.swapBehavior,
				memoryCapacity:             nodeMem,
				swapCapacity:               nodeSwap,
				getSwapControllerAvailable: swapAvailableFn,
			}
			require.NoError(t, pcm.EnsureExists(logger, tt.pod))
			if tt.alreadyExists {
				require.Empty(t, fakeCM.created)
				return
			}
			require.Len(t, fakeCM.created, 1)
			gotUnified := fakeCM.created[0].ResourceParameters.Unified
			if tt.expectedSwapMax == nil {
				require.NotContains(t, gotUnified, Cgroup2MaxSwapFilename)
			} else {
				require.Equal(t, *tt.expectedSwapMax, gotUnified[Cgroup2MaxSwapFilename])
			}
		})
	}
}

// TestEnsureExistsPodLevelSwapLimit covers how an explicit pod-level
// limits.swap (KEP-5359) and the LimitedSwap value share the pod cgroup's
// memory.swap.max.
func TestEnsureExistsPodLevelSwapLimit(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.PodLevelResources, true)

	origIsCgroup2 := isCgroup2UnifiedMode
	t.Cleanup(func() { isCgroup2UnifiedMode = origIsCgroup2 })
	isCgroup2UnifiedMode = func() bool { return true }

	const (
		gi             = 1024 * 1024 * 1024
		nodeMem  int64 = 40 * gi
		nodeSwap int64 = 5 * gi
	)
	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}
	newPod := func(uid string, swapLimit string) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: types.UID(uid)},
			Spec: v1.PodSpec{
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("2Gi")},
					Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("4Gi")},
				},
				Containers: []v1.Container{{Name: "c1"}},
			},
		}
		if swapLimit != "" {
			pod.Spec.Resources.Limits[v1.ResourceSwap] = resource.MustParse(swapLimit)
		}
		return pod
	}
	explicit := strconv.FormatInt(512*1024*1024, 10)
	// 2Gi of 40Gi node memory, of 5Gi node swap.
	proportional := strconv.FormatInt(gi/4, 10)

	for _, tc := range []struct {
		name                 string
		gate                 bool
		swapBehavior         kubetypes.SwapBehavior
		noSwapController     bool
		podSwapLimit         string
		expectedSwapMax      string
		expectedSwapMaxUnset bool
	}{
		{
			name: "LimitedSwap node: pod-level limits.swap wins",
			gate: true, swapBehavior: kubetypes.LimitedSwap, podSwapLimit: "512Mi",
			expectedSwapMax: explicit,
		},
		{
			name: "LimitedSwap node: no limits.swap keeps the LimitedSwap value",
			gate: true, swapBehavior: kubetypes.LimitedSwap,
			expectedSwapMax: proportional,
		},
		{
			name: "LimitedSwap node, gate off: limits.swap is ignored",
			gate: false, swapBehavior: kubetypes.LimitedSwap, podSwapLimit: "512Mi",
			expectedSwapMax: proportional,
		},
		{
			name: "WorkloadControlledSwap node: pod-level limits.swap is applied",
			gate: true, swapBehavior: kubetypes.WorkloadControlledSwap, podSwapLimit: "512Mi",
			expectedSwapMax: explicit,
		},
		{
			name: "WorkloadControlledSwap node: no limits.swap leaves the pod cgroup unlimited",
			gate: true, swapBehavior: kubetypes.WorkloadControlledSwap,
			expectedSwapMaxUnset: true,
		},
		{
			name: "NoSwap node: limits.swap is not applied",
			gate: true, swapBehavior: kubetypes.NoSwap, podSwapLimit: "512Mi",
			expectedSwapMax: "0",
		},
		{
			name: "no swap controller: memory.swap.max is not written",
			gate: true, swapBehavior: kubetypes.LimitedSwap, noSwapController: true, podSwapLimit: "512Mi",
			expectedSwapMaxUnset: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.WorkloadControlledSwap, tc.gate)
			fakeCM := &fakeCgroupManager{}
			pcm := &podContainerManagerImpl{
				qosContainersInfo:          qosContainersInfo,
				cgroupManager:              fakeCM,
				enforceCPULimits:           true,
				cpuCFSQuotaPeriod:          QuotaPeriod,
				podContainerManager:        NewStubContainerManager(),
				memorySwapBehavior:         tc.swapBehavior,
				memoryCapacity:             nodeMem,
				swapCapacity:               nodeSwap,
				getSwapControllerAvailable: func() bool { return !tc.noSwapController },
			}
			require.NoError(t, pcm.EnsureExists(logger, newPod("pod-level-swap", tc.podSwapLimit)))
			require.Len(t, fakeCM.created, 1)
			gotUnified := fakeCM.created[0].ResourceParameters.Unified
			if tc.expectedSwapMaxUnset {
				require.NotContains(t, gotUnified, Cgroup2MaxSwapFilename)
			} else {
				require.Equal(t, tc.expectedSwapMax, gotUnified[Cgroup2MaxSwapFilename])
			}
		})
	}
}

// failingCgroupManager is a fakeCgroupManager whose Create always fails.
type failingCgroupManager struct {
	fakeCgroupManager
}

func (f *failingCgroupManager) Create(klog.Logger, *CgroupConfig) error {
	return fmt.Errorf("cgroup create failed")
}

func TestEnsureExistsPodPIDLimit(t *testing.T) {
	tests := []struct {
		name            string
		featureEnabled  bool
		nodePidsLimit   int64
		podPidsLimit    string // empty means unset
		expectPidsLimit *int64
		expectCapped    bool
	}{
		{
			name:            "pod limit below node limit is applied",
			featureEnabled:  true,
			nodePidsLimit:   4096,
			podPidsLimit:    "2048",
			expectPidsLimit: new(int64(2048)),
		},
		{
			name:            "pod limit above node limit is capped and reported",
			featureEnabled:  true,
			nodePidsLimit:   4096,
			podPidsLimit:    "8192",
			expectPidsLimit: new(int64(4096)),
			expectCapped:    true,
		},
		{
			name:            "pod limit equal to node limit is applied without event",
			featureEnabled:  true,
			nodePidsLimit:   4096,
			podPidsLimit:    "4096",
			expectPidsLimit: new(int64(4096)),
		},
		{
			name:            "pod limit applied when node limit is unlimited",
			featureEnabled:  true,
			nodePidsLimit:   -1,
			podPidsLimit:    "2048",
			expectPidsLimit: new(int64(2048)),
		},
		{
			name:            "no pod limit falls back to node limit",
			featureEnabled:  true,
			nodePidsLimit:   4096,
			expectPidsLimit: new(int64(4096)),
		},
		{
			name:            "feature disabled ignores pod limit",
			featureEnabled:  false,
			nodePidsLimit:   4096,
			podPidsLimit:    "2048",
			expectPidsLimit: new(int64(4096)),
		},
		{
			name:           "feature disabled with unlimited node limit sets nothing",
			featureEnabled: false,
			nodePidsLimit:  -1,
			podPidsLimit:   "2048",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.PerPodPIDLimit, tc.featureEnabled)
			logger, _ := ktesting.NewTestContext(t)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default", UID: "fake-uid"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}},
				},
			}
			if tc.podPidsLimit != "" {
				pod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourcePID: resource.MustParse(tc.podPidsLimit),
					},
				}
			}

			fakeRecorder := record.NewFakeRecorder(10)
			fakeCM := &fakeCgroupManager{}
			pcm := &podContainerManagerImpl{
				podPidsLimit:        tc.nodePidsLimit,
				cgroupManager:       fakeCM,
				qosContainersInfo:   QOSContainersInfo{Guaranteed: RootCgroupName, Burstable: NewCgroupName(RootCgroupName, "burstable"), BestEffort: NewCgroupName(RootCgroupName, "besteffort")},
				podContainerManager: NewFakeContainerManager(logger),
				recorder:            fakeRecorder,
			}

			require.NoError(t, pcm.EnsureExists(logger, pod))
			require.Len(t, fakeCM.created, 1)
			gotPidsLimit := fakeCM.created[0].ResourceParameters.PidsLimit
			if tc.expectPidsLimit == nil {
				require.Nil(t, gotPidsLimit)
			} else {
				require.NotNil(t, gotPidsLimit)
				require.Equal(t, *tc.expectPidsLimit, *gotPidsLimit)
			}

			select {
			case event := <-fakeRecorder.Events:
				require.True(t, tc.expectCapped, "unexpected event: %s", event)
				require.Contains(t, event, "PIDLimitCapped")
				require.Contains(t, event, tc.podPidsLimit)
			default:
				require.False(t, tc.expectCapped, "expected PIDLimitCapped event but none was emitted")
			}
		})
	}
}

func TestEnsureExistsNoCappedEventOnCreateFailure(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.PerPodPIDLimit, true)
	logger, _ := ktesting.NewTestContext(t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default", UID: "fake-uid"},
		Spec: v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("8192")},
			},
			Containers: []v1.Container{{Name: "c1"}},
		},
	}

	fakeRecorder := record.NewFakeRecorder(10)
	pcm := &podContainerManagerImpl{
		podPidsLimit:        4096,
		cgroupManager:       &failingCgroupManager{},
		qosContainersInfo:   QOSContainersInfo{Guaranteed: RootCgroupName, Burstable: NewCgroupName(RootCgroupName, "burstable"), BestEffort: NewCgroupName(RootCgroupName, "besteffort")},
		podContainerManager: NewFakeContainerManager(logger),
		recorder:            fakeRecorder,
	}

	require.Error(t, pcm.EnsureExists(logger, pod))
	select {
	case event := <-fakeRecorder.Events:
		t.Fatalf("no event should be emitted when cgroup creation fails, got: %s", event)
	default:
	}
}

// validatingCgroupManager returns a fixed Validate error and records the cgroup it was asked about.
type validatingCgroupManager struct {
	fakeCgroupManager
	err       error
	validated CgroupName
}

func (m *validatingCgroupManager) Validate(name CgroupName) error {
	m.validated = name
	return m.err
}

func (m *validatingCgroupManager) Exists(name CgroupName) bool {
	return m.Validate(name) == nil
}

func TestPodContainerManagerValidate(t *testing.T) {
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "fake-uid"}}
	missing := errors.New("cgroup is missing controllers: cpuset")
	for _, err := range []error{nil, missing} {
		cgroupManager := &validatingCgroupManager{err: err}
		pcm := &podContainerManagerImpl{
			cgroupManager:     cgroupManager,
			qosContainersInfo: QOSContainersInfo{BestEffort: RootCgroupName},
		}
		wantName, _ := pcm.GetPodContainerName(pod)

		require.ErrorIs(t, pcm.Validate(pod), err)
		require.Equal(t, wantName, cgroupManager.validated)
		require.Equal(t, err == nil, pcm.Exists(pod))
	}
}
