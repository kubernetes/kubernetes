//go:build windows

/*
Copyright 2022 The Kubernetes Authors.

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

package kuberuntime

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

func TestApplyPlatformSpecificContainerConfig(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeRuntimeSvc, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	containerConfig := &runtimeapi.ContainerConfig{}

	resources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("128Mi"),
			v1.ResourceCPU:    resource.MustParse("1"),
		},
		Limits: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("256Mi"),
			v1.ResourceCPU:    resource.MustParse("3"),
		},
	}

	gmsaCredSpecName := "gmsa spec name"
	gmsaCredSpec := "credential spec"
	username := "ContainerAdministrator"
	asHostProcess := true
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					Resources:       resources,
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							GMSACredentialSpecName: &gmsaCredSpecName,
							GMSACredentialSpec:     &gmsaCredSpec,
							RunAsUserName:          &username,
							HostProcess:            &asHostProcess,
						},
					},
				},
			},
		},
	}

	err = fakeRuntimeSvc.applyPlatformSpecificContainerConfig(tCtx, containerConfig, &pod.Spec.Containers[0], pod, new(int64), "foo", nil)
	require.NoError(t, err)

	limit := int64(3000)
	expectedCpuMax := 10 * limit / int64(winstats.ProcessorCount())
	// Above, we're setting the limit to 3 CPUs. But we can't expect more than 100% of the CPUs
	// we have. (e.g.: if we only have 2 CPUs, we can't have 150% CPU max).
	if expectedCpuMax > 10000 {
		expectedCpuMax = 10000
	}
	expectedWindowsConfig := &runtimeapi.WindowsContainerConfig{
		Resources: &runtimeapi.WindowsContainerResources{
			CpuMaximum:         expectedCpuMax,
			MemoryLimitInBytes: 256 * 1024 * 1024,
		},
		SecurityContext: &runtimeapi.WindowsContainerSecurityContext{
			CredentialSpec: gmsaCredSpec,
			RunAsUsername:  "ContainerAdministrator",
			HostProcess:    true,
		},
	}
	assert.Equal(t, expectedWindowsConfig, containerConfig.Windows)
}

func TestCalculateCPUMaximum(t *testing.T) {
	tests := []struct {
		name     string
		cpuLimit resource.Quantity
		cpuCount int64
		want     int64
	}{
		{
			name:     "max range when same amount",
			cpuLimit: resource.MustParse("1"),
			cpuCount: 1,
			want:     10000,
		},
		{
			name:     "percentage calculation is working as intended",
			cpuLimit: resource.MustParse("94"),
			cpuCount: 96,
			want:     9791,
		},
		{
			name:     "half range when half amount",
			cpuLimit: resource.MustParse("1"),
			cpuCount: 2,
			want:     5000,
		},
		{
			name:     "max range when more requested than available",
			cpuLimit: resource.MustParse("2"),
			cpuCount: 1,
			want:     10000,
		},
		{
			name:     "min range when less than minimum",
			cpuLimit: resource.MustParse("1m"),
			cpuCount: 100,
			want:     1,
		},
		{
			// 10 * MilliValue() fits int64 up to and including this value.
			name:     "largest limit whose product fits int64",
			cpuLimit: resource.MustParse("922337203685477580m"),
			cpuCount: 4,
			want:     10000,
		},
		{
			// TODO(#141166): MilliValue() fits int64 but 10 * MilliValue() does not; must clamp to 10000.
			name:     "smallest limit whose product overflows int64",
			cpuLimit: resource.MustParse("922337203685477581m"),
			cpuCount: 4,
			want:     1,
		},
		{
			// TODO(#141166): A limit past int64 must clamp to 10000.
			name:     "limit past int64",
			cpuLimit: resource.MustParse("100E"),
			cpuCount: 4,
			want:     1,
		},
		{
			// TODO(#141166): A limit past int64 must clamp to 10000.
			name:     "limit of MaxInt64 plus one milli",
			cpuLimit: resource.MustParse("9223372036854775808m"),
			cpuCount: 4,
			want:     1,
		},
		{
			// TODO(#141166): A limit past int64 must clamp to 10000.
			name:     "limit of 2^64 plus 4000 milli",
			cpuLimit: resource.MustParse("18446744073709555616m"),
			cpuCount: 4,
			want:     1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, calculateCPUMaximum(&tt.cpuLimit, tt.cpuCount))
		})
	}
}

func TestCalculateWindowsResources(t *testing.T) {
	// TODO: remove skip once the failing test has been fixed.
	t.Skip("Skip failing test on Windows.")

	tCtx := ktesting.Init(t)
	_, _, fakeRuntimeSvc, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	tests := []struct {
		name     string
		cpuLim   resource.Quantity
		memLim   resource.Quantity
		expected *runtimeapi.WindowsContainerResources
	}{
		{
			name:   "Request128MBLimit256MB",
			cpuLim: resource.MustParse("2"),
			memLim: resource.MustParse("128Mi"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         2500,
				MemoryLimitInBytes: 134217728,
			},
		},
		{
			name:   "RequestNoMemory",
			cpuLim: resource.MustParse("8"),
			memLim: resource.MustParse("0"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         10000,
				MemoryLimitInBytes: 0,
			},
		},
		{
			name:   "RequestZeroCPU",
			cpuLim: resource.MustParse("0"),
			memLim: resource.MustParse("128Mi"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         1,
				MemoryLimitInBytes: 134217728,
			},
		},
	}
	for _, test := range tests {
		windowsContainerResources := fakeRuntimeSvc.calculateWindowsResources(tCtx, &test.cpuLim, &test.memLim)
		assert.Equal(t, test.expected, windowsContainerResources)
	}
}

func TestResizePodInPlacePlatformWindows(t *testing.T) {
	newPod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{UID: "resize-pod", Name: "resize-pod", Namespace: "default"},
			Spec: v1.PodSpec{Containers: []v1.Container{{
				Name: "app",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("192Mi")},
					Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("256Mi")},
				},
			}}},
		}
	}
	usage := uint64(220 * 1024 * 1024)
	tests := []struct {
		name       string
		currentMi  int64
		desiredMi  int64
		wantErr    string
		wantUpdate bool
	}{
		{name: "memory increase is applied", currentMi: 128, desiredMi: 256, wantUpdate: true},
		{name: "memory decrease below usage is rejected", currentMi: 256, desiredMi: 200, wantErr: "current usage is 230686720", wantUpdate: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			fakeRuntime, _, m, err := createTestRuntimeManager(tCtx)
			require.NoError(t, err)
			helper := &containertest.FakeRuntimeHelper{PodStats: map[types.UID]*statsapi.PodStats{
				"resize-pod": {Containers: []statsapi.ContainerStats{{Name: "app", Memory: &statsapi.MemoryStats{UsageBytes: &usage}}}},
			}}
			m.runtimeHelper = helper
			pod := newPod()
			pod.Spec.Containers[0].Resources.Limits[v1.ResourceMemory] = *resource.NewQuantity(tt.desiredMi*1024*1024, resource.BinarySI)
			pod.Spec.Containers[0].Resources.Requests[v1.ResourceMemory] = *resource.NewQuantity(tt.desiredMi*1024*1024, resource.BinarySI)
			_, containers := makeAndSetFakePod(tCtx, m, fakeRuntime, pod)
			require.Len(t, containers, 1)
			var containerID kubecontainer.ContainerID
			for id := range fakeRuntime.Containers {
				containerID = kubecontainer.ContainerID{Type: "testRuntime", ID: id}
				break
			}
			require.NotEmpty(t, containerID.ID)
			fakeRuntime.Called = []string{}

			handled, err := m.resizePodInPlacePlatform(tCtx, pod, &kubecontainer.PodStatus{}, podActions{ContainersToUpdate: map[v1.ResourceName][]containerToUpdateInfo{
				v1.ResourceMemory: {{
					container:                 &pod.Spec.Containers[0],
					kubeContainerID:           containerID,
					containerType:             podutil.Containers,
					currentContainerResources: &resourceRequirements{memoryLimit: tt.currentMi * 1024 * 1024, memoryRequest: tt.currentMi * 1024 * 1024},
					desiredContainerResources: resourceRequirements{memoryLimit: tt.desiredMi * 1024 * 1024, memoryRequest: tt.desiredMi * 1024 * 1024},
				}},
			}})
			require.True(t, handled)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tt.wantUpdate, slices.Contains(fakeRuntime.Called, "UpdateContainerResources"))
			if tt.wantUpdate {
				c, ok := fakeRuntime.Containers[containerID.ID]
				require.True(t, ok)
				require.NotNil(t, c.Resources)
				require.NotNil(t, c.Resources.Windows)
				assert.Equal(t, tt.desiredMi*1024*1024, c.Resources.Windows.MemoryLimitInBytes)
			}
		})
	}
}
