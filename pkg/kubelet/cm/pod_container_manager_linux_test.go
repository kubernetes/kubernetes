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
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"

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

func TestPodContainerManagerValidateDelegatesToCgroupManager(t *testing.T) {
	pod := newBurstablePodForPodContainerManagerTest("pod-uid")
	qosContainersInfo := podContainerManagerTestQOSContainersInfo()
	expectedName := NewCgroupName(qosContainersInfo.Burstable, GetPodCgroupNameSuffix(pod.UID))
	expectedErr := errors.New("pod cgroup validation failed")
	cgroupManager := &recordingCgroupManager{validateErr: expectedErr}
	pcm := &podContainerManagerImpl{
		cgroupManager:     cgroupManager,
		qosContainersInfo: qosContainersInfo,
	}

	err := pcm.Validate(pod)

	validatedName, validateCalls, existsCalls := cgroupManager.snapshot()
	require.ErrorIs(t, err, expectedErr)
	require.Equal(t, 1, validateCalls)
	require.Equal(t, expectedName, validatedName)
	require.Zero(t, existsCalls)
}

func TestPodContainerManagerExistsUsesValidateResult(t *testing.T) {
	pod := newBurstablePodForPodContainerManagerTest("pod-uid")
	qosContainersInfo := podContainerManagerTestQOSContainersInfo()

	testCases := []struct {
		name          string
		validateErr   error
		expectedExist bool
	}{
		{
			name:          "returns true when validation succeeds",
			validateErr:   nil,
			expectedExist: true,
		},
		{
			name:          "returns false when validation fails",
			validateErr:   errors.New("missing pod cgroup"),
			expectedExist: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			cgroupManager := &recordingCgroupManager{validateErr: tc.validateErr}
			pcm := &podContainerManagerImpl{
				cgroupManager:     cgroupManager,
				qosContainersInfo: qosContainersInfo,
			}

			exists := pcm.Exists(pod)

			_, validateCalls, existsCalls := cgroupManager.snapshot()
			require.Equal(t, tc.expectedExist, exists)
			require.Equal(t, 1, validateCalls)
			require.Zero(t, existsCalls)
		})
	}
}

func TestPodContainerManagerNoopValidateAlwaysSucceeds(t *testing.T) {
	pcm := &podContainerManagerNoop{}
	pod := newBurstablePodForPodContainerManagerTest("pod-uid")

	require.NoError(t, pcm.Validate(pod))
	require.True(t, pcm.Exists(pod))
}

func TestFakePodContainerManagerUsesConfiguredResults(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		validateErr := errors.New("validation failed")
		manager := NewFakePodContainerManager()
		manager.ValidateError = validateErr

		err := manager.Validate(&v1.Pod{})

		require.ErrorIs(t, err, validateErr)
		require.Equal(t, []string{"Validate"}, manager.CalledFunctions)
	})

	t.Run("Exists", func(t *testing.T) {
		existsResult := false
		manager := NewFakePodContainerManager()
		manager.ExistsResult = ptr.To(existsResult)

		exists := manager.Exists(&v1.Pod{})

		require.Equal(t, existsResult, exists)
		require.Equal(t, []string{"Exists"}, manager.CalledFunctions)
	})

	t.Run("EnsureExists", func(t *testing.T) {
		ensureExistsErr := errors.New("ensure exists failed")
		manager := NewFakePodContainerManager()
		manager.EnsureExistsErr = ensureExistsErr

		err := manager.EnsureExists(klog.TODO(), &v1.Pod{})

		require.ErrorIs(t, err, ensureExistsErr)
		require.Equal(t, []string{"EnsureExists"}, manager.CalledFunctions)
	})
}

func TestPodContainerManagerStubValidateAndExistsAreNoops(t *testing.T) {
	manager := &podContainerManagerStub{}

	require.NoError(t, manager.Validate(&v1.Pod{}))
	require.True(t, manager.Exists(&v1.Pod{}))
	require.NoError(t, manager.EnsureExists(klog.TODO(), &v1.Pod{}))
}

func podContainerManagerTestQOSContainersInfo() QOSContainersInfo {
	return QOSContainersInfo{
		Guaranteed: RootCgroupName,
		Burstable:  NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBurstable))),
		BestEffort: NewCgroupName(RootCgroupName, strings.ToLower(string(v1.PodQOSBestEffort))),
	}
}

func newBurstablePodForPodContainerManagerTest(uid types.UID) *v1.Pod {
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

type recordingCgroupManager struct {
	mu            sync.Mutex
	validatedName CgroupName
	validateErr   error
	validateCalls int
	existsCalls   int
}

var _ CgroupManager = &recordingCgroupManager{}

func (m *recordingCgroupManager) Create(_ klog.Logger, _ *CgroupConfig) error {
	return nil
}

func (m *recordingCgroupManager) Destroy(_ klog.Logger, _ *CgroupConfig) error {
	return nil
}

func (m *recordingCgroupManager) Update(_ klog.Logger, _ *CgroupConfig) error {
	return nil
}

func (m *recordingCgroupManager) Validate(name CgroupName) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.validatedName = append(CgroupName(nil), name...)
	m.validateCalls++
	return m.validateErr
}

func (m *recordingCgroupManager) Exists(_ CgroupName) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.existsCalls++
	return false
}

func (m *recordingCgroupManager) snapshot() (CgroupName, int, int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append(CgroupName(nil), m.validatedName...), m.validateCalls, m.existsCalls
}

func (m *recordingCgroupManager) Name(name CgroupName) string {
	return strings.Join(name, "/")
}

func (m *recordingCgroupManager) CgroupName(name string) CgroupName {
	if name == "" {
		return nil
	}
	return strings.Split(name, "/")
}

func (m *recordingCgroupManager) Pids(_ klog.Logger, _ CgroupName) []int {
	return nil
}

func (m *recordingCgroupManager) ReduceCPULimits(_ klog.Logger, _ CgroupName) error {
	return nil
}

func (m *recordingCgroupManager) MemoryUsage(_ CgroupName) (int64, error) {
	return 0, nil
}

func (m *recordingCgroupManager) GetCgroupConfig(_ CgroupName, _ v1.ResourceName) (*ResourceConfig, error) {
	return nil, nil
}

func (m *recordingCgroupManager) SetCgroupConfig(_ klog.Logger, _ CgroupName, _ *ResourceConfig) error {
	return nil
}

func (m *recordingCgroupManager) Version() int {
	return 0
}
