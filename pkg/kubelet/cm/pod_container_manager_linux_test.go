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
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
