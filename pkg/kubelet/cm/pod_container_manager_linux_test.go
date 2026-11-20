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
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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
			unconfiguredPods:  newPodCgroupTracker(),
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
			unconfiguredPods:  newPodCgroupTracker(),
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
				unconfiguredPods:    newPodCgroupTracker(),
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
				unconfiguredPods:           newPodCgroupTracker(),
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
				unconfiguredPods:           newPodCgroupTracker(),
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

type testPodCgroupManager struct {
	*fakeCgroupManager
	mu               sync.Mutex
	exists           map[string]bool
	createCalls      int
	updateCalls      int
	createErr        error
	updateErr        error
	createPartialDir bool
}

func newTestPodCgroupManager() *testPodCgroupManager {
	return &testPodCgroupManager{
		fakeCgroupManager: &fakeCgroupManager{},
		exists:            make(map[string]bool),
	}
}

func (m *testPodCgroupManager) Exists(name CgroupName) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.exists[m.Name(name)]
}

func (m *testPodCgroupManager) Create(_ klog.Logger, config *CgroupConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.createCalls++
	key := m.Name(config.Name)
	if m.createErr != nil {
		if m.createPartialDir {
			m.exists[key] = true
		}
		return m.createErr
	}
	m.exists[key] = true
	return nil
}

func (m *testPodCgroupManager) Update(_ klog.Logger, _ *CgroupConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.updateCalls++
	return m.updateErr
}

func (m *testPodCgroupManager) Destroy(_ klog.Logger, config *CgroupConfig) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.exists, m.Name(config.Name))
	return nil
}

func TestEnsureExists(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	qosContainersInfo := QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}

	newPod := func(uid types.UID) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: uid},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container"}},
			},
		}
	}

	newPCM := func(fakeCM *testPodCgroupManager, tracker *podCgroupTracker) *podContainerManagerImpl {
		return &podContainerManagerImpl{
			unconfiguredPods:    tracker,
			cgroupManager:       fakeCM,
			podContainerManager: NewStubContainerManager(),
			qosContainersInfo:   qosContainersInfo,
			enforceCPULimits:    true,
		}
	}

	t.Run("creates cgroup once and skips subsequent calls", func(t *testing.T) {
		fakeCM := newTestPodCgroupManager()
		pcm := newPCM(fakeCM, newPodCgroupTracker())
		pod := newPod("uid-create-once")

		require.NoError(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.createCalls)
		require.Equal(t, 0, fakeCM.updateCalls)

		require.NoError(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.createCalls)
		require.Equal(t, 0, fakeCM.updateCalls)
	})

	t.Run("reconciles partial Create failure via Update on retry", func(t *testing.T) {
		fakeCM := newTestPodCgroupManager()
		fakeCM.createErr = fmt.Errorf("simulated Set failure")
		fakeCM.createPartialDir = true
		pcm := newPCM(fakeCM, newPodCgroupTracker())
		pod := newPod("uid-partial-fail")

		// Create leaves the cgroup on disk but fails to apply its parameters.
		require.Error(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.createCalls)
		require.True(t, pcm.Exists(pod))
		require.True(t, pcm.unconfiguredPods.has(pod.UID))

		// The retry reapplies the parameters via Update.
		fakeCM.createErr = nil
		require.NoError(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.createCalls)
		require.Equal(t, 1, fakeCM.updateCalls)
		require.False(t, pcm.unconfiguredPods.has(pod.UID))

		// Once configured, further calls are no-ops.
		require.NoError(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.updateCalls)
	})

	t.Run("reconciliation state is shared across pod container manager instances", func(t *testing.T) {
		// The kubelet creates a new pod container manager on every pod sync,
		// so the retry must see the failure recorded by the earlier instance.
		fakeCM := newTestPodCgroupManager()
		fakeCM.createErr = fmt.Errorf("simulated Set failure")
		fakeCM.createPartialDir = true
		tracker := newPodCgroupTracker()
		pod := newPod("uid-shared")

		require.Error(t, newPCM(fakeCM, tracker).EnsureExists(logger, pod))

		fakeCM.createErr = nil
		require.NoError(t, newPCM(fakeCM, tracker).EnsureExists(logger, pod))
		require.Equal(t, 1, fakeCM.updateCalls)
	})

	t.Run("does not update an existing configured pod cgroup", func(t *testing.T) {
		// Covers later syncs, in-place resizes and kubelet restarts: only the
		// resize path may change the parameters of a configured pod cgroup.
		fakeCM := newTestPodCgroupManager()
		pod := newPod("uid-existing")
		pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceMemory: resource.MustParse("50Mi")}
		pod.Spec.Containers[0].Resources.Limits = v1.ResourceList{v1.ResourceMemory: resource.MustParse("200Mi")}
		require.NoError(t, newPCM(fakeCM, newPodCgroupTracker()).EnsureExists(logger, pod))

		// Shrink the memory limit while keeping the pod Burstable, so that the
		// pod cgroup path is unchanged.
		resized := pod.DeepCopy()
		resized.Spec.Containers[0].Resources.Limits[v1.ResourceMemory] = resource.MustParse("100Mi")
		for range 3 {
			require.NoError(t, newPCM(fakeCM, newPodCgroupTracker()).EnsureExists(logger, resized))
		}
		require.Equal(t, 1, fakeCM.createCalls)
		require.Equal(t, 0, fakeCM.updateCalls)
	})

	t.Run("recreates pod cgroup removed out-of-band", func(t *testing.T) {
		fakeCM := newTestPodCgroupManager()
		pcm := newPCM(fakeCM, newPodCgroupTracker())
		pod := newPod("uid-removed")

		require.NoError(t, pcm.EnsureExists(logger, pod))
		podCgroup, _ := pcm.GetPodContainerName(pod)
		fakeCM.mu.Lock()
		delete(fakeCM.exists, fakeCM.Name(podCgroup))
		fakeCM.mu.Unlock()

		require.NoError(t, pcm.EnsureExists(logger, pod))
		require.Equal(t, 2, fakeCM.createCalls)
		require.True(t, pcm.Exists(pod))
	})

	t.Run("Destroy clears reconciliation state", func(t *testing.T) {
		fakeCM := newTestPodCgroupManager()
		fakeCM.createErr = fmt.Errorf("simulated Set failure")
		fakeCM.createPartialDir = true
		pcm := newPCM(fakeCM, newPodCgroupTracker())
		pod := newPod("uid-destroy")

		require.Error(t, pcm.EnsureExists(logger, pod))
		require.True(t, pcm.unconfiguredPods.has(pod.UID))

		podCgroup, _ := pcm.GetPodContainerName(pod)
		require.NoError(t, pcm.Destroy(logger, podCgroup))
		require.False(t, pcm.unconfiguredPods.has(pod.UID))
	})

	t.Run("concurrent EnsureExists and Destroy are race-free", func(t *testing.T) {
		fakeCM := newTestPodCgroupManager()
		fakeCM.createErr = fmt.Errorf("simulated Set failure")
		fakeCM.createPartialDir = true
		tracker := newPodCgroupTracker()

		var wg sync.WaitGroup
		for i := range 16 {
			wg.Go(func() {
				pcm := newPCM(fakeCM, tracker)
				pod := newPod(types.UID(fmt.Sprintf("uid-concurrent-%d", i)))
				// Partial Create failure, then reconciliation via Update.
				// require must not be used outside the test goroutine.
				if err := pcm.EnsureExists(logger, pod); err == nil {
					t.Errorf("expected partial Create failure for pod %s", pod.UID)
				}
				if err := pcm.EnsureExists(logger, pod); err != nil {
					t.Errorf("reconciling pod %s: %v", pod.UID, err)
				}
				podCgroup, _ := pcm.GetPodContainerName(pod)
				if err := pcm.Destroy(logger, podCgroup); err != nil {
					t.Errorf("destroying pod %s: %v", pod.UID, err)
				}
			})
		}
		wg.Wait()
	})
}
