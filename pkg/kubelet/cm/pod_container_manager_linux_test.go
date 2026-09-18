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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

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
	writable := v1.CgroupMountModeWritable
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
			name: "writable on a container",
			pod:  &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "c", SecurityContext: scWith(&writable)}}}},
			want: true,
		},
		{
			name: "writable on an init container",
			pod:  &v1.Pod{Spec: v1.PodSpec{InitContainers: []v1.Container{{Name: "i", SecurityContext: scWith(&writable)}}}},
			want: true,
		},
		{
			name: "ephemeral containers are ignored",
			pod: &v1.Pod{Spec: v1.PodSpec{EphemeralContainers: []v1.EphemeralContainer{{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", SecurityContext: scWith(&writable)},
			}}}},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, podRequestsWritableCgroups(tt.pod))
		})
	}
}

func TestEnsureExists(t *testing.T) {
	if !libcontainercgroups.IsCgroup2UnifiedMode() {
		t.Skip("skipping cgroup v2 test on a cgroup v1 system")
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: types.UID("123")},
		Spec: v1.PodSpec{Containers: []v1.Container{{
			Name: "c",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				Limits:   v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			},
		}}},
	}
	tests := []struct {
		name                 string
		cgroupOptionsEnabled bool
		memoryQoSEnabled     bool
		mountMode            v1.CgroupMountMode
		wantUnified          map[string]string
	}{
		{
			name:                 "preserves existing resource settings with writable cgroups",
			cgroupOptionsEnabled: true,
			memoryQoSEnabled:     true,
			mountMode:            v1.CgroupMountModeWritable,
			wantUnified: map[string]string{
				Cgroup2MemoryLow:      "134217728",
				Cgroup2MaxDescendants: defaultWritableCgroupMaxDescendants,
				Cgroup2MaxDepth:       defaultWritableCgroupMaxDepth,
			},
		},
		{
			name:                 "sets writable cgroup limits with MemoryQoS disabled",
			cgroupOptionsEnabled: true,
			memoryQoSEnabled:     false,
			mountMode:            v1.CgroupMountModeWritable,
			wantUnified: map[string]string{
				Cgroup2MaxDescendants: defaultWritableCgroupMaxDescendants,
				Cgroup2MaxDepth:       defaultWritableCgroupMaxDepth,
			},
		},
		{
			name:                 "omits writable cgroup limits with CgroupOptions disabled",
			cgroupOptionsEnabled: false,
			memoryQoSEnabled:     true,
			mountMode:            v1.CgroupMountModeWritable,
			wantUnified:          map[string]string{Cgroup2MemoryLow: "134217728"},
		},
		{
			name:                 "omits writable cgroup limits for a read-only mount",
			cgroupOptionsEnabled: true,
			memoryQoSEnabled:     true,
			mountMode:            v1.CgroupMountModeReadOnly,
			wantUnified:          map[string]string{Cgroup2MemoryLow: "134217728"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.CgroupOptions, tt.cgroupOptionsEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, kubefeatures.MemoryQoS, tt.memoryQoSEnabled)
			testPod := pod.DeepCopy()
			testPod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
				CgroupOptions: &v1.CgroupOptions{MountMode: &tt.mountMode},
			}
			logger, _ := ktesting.NewTestContext(t)
			fake := &fakeCgroupManager{}
			m := &podContainerManagerImpl{
				cgroupManager:           fake,
				podContainerManager:     NewFakeContainerManager(logger),
				memoryReservationPolicy: kubeletconfig.TieredReservationMemoryReservationPolicy,
			}

			require.NoError(t, m.EnsureExists(logger, testPod))
			require.Len(t, fake.created, 1)
			require.Equal(t, tt.wantUnified, fake.created[0].ResourceParameters.Unified)
		})
	}
}
