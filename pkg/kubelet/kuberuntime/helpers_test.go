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

package kuberuntime

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	runtimetesting "k8s.io/cri-api/pkg/apis/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

type podStatusProviderFunc func(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error)

func (f podStatusProviderFunc) GetPodStatus(_ context.Context, uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	return f(uid, name, namespace)
}

func TestIsInitContainerFailed(t *testing.T) {
	tests := []struct {
		status      *kubecontainer.Status
		isFailed    bool
		description string
	}{
		{
			status: &kubecontainer.Status{
				State:    kubecontainer.ContainerStateExited,
				ExitCode: 1,
			},
			isFailed:    true,
			description: "Init container in exited state and non-zero exit code should return true",
		},
		{
			status: &kubecontainer.Status{
				State: kubecontainer.ContainerStateUnknown,
			},
			isFailed:    true,
			description: "Init container in unknown state should return true",
		},
		{
			status: &kubecontainer.Status{
				Reason:   "OOMKilled",
				ExitCode: 0,
			},
			isFailed:    true,
			description: "Init container which reason is OOMKilled should return true",
		},
		{
			status: &kubecontainer.Status{
				State:    kubecontainer.ContainerStateExited,
				ExitCode: 0,
			},
			isFailed:    false,
			description: "Init container in exited state and zero exit code should return false",
		},
		{
			status: &kubecontainer.Status{
				State: kubecontainer.ContainerStateRunning,
			},
			isFailed:    false,
			description: "Init container in running state should return false",
		},
		{
			status: &kubecontainer.Status{
				State: kubecontainer.ContainerStateCreated,
			},
			isFailed:    false,
			description: "Init container in created state should return false",
		},
	}
	for i, test := range tests {
		isFailed := isInitContainerFailed(test.status)
		assert.Equal(t, test.isFailed, isFailed, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGetBackoffKey(t *testing.T) {
	testSpecs := map[string]v1.PodSpec{
		"empty resources": {
			Containers: []v1.Container{{
				Name:  "test_container",
				Image: "foo/image:v1",
			}},
		},
		"with resources": {
			Containers: []v1.Container{{
				Name:  "test_container",
				Image: "foo/image:v1",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("100Mi"),
					},
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				},
			}},
		},
	}

	for name, spec := range testSpecs {
		t.Run(name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test_pod",
					Namespace: "test_pod_namespace",
					UID:       "test_pod_uid",
				},
				Spec: spec,
			}
			secondContainer := v1.Container{
				Name:  "second_container",
				Image: "registry.k8s.io/pause",
			}
			pod.Spec.Containers = append(pod.Spec.Containers, secondContainer)
			originalKey := GetBackoffKey(pod, &pod.Spec.Containers[0])

			podCopy := pod.DeepCopy()
			podCopy.Spec.ActiveDeadlineSeconds = ptr.To[int64](1)
			assert.Equal(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[0]),
				"Unrelated change should not change the key")

			podCopy = pod.DeepCopy()
			assert.NotEqual(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[1]),
				"Different container change should change the key")

			podCopy = pod.DeepCopy()
			podCopy.Name = "other-pod"
			assert.NotEqual(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[0]),
				"Different pod name should change the key")

			podCopy = pod.DeepCopy()
			podCopy.Namespace = "other-namespace"
			assert.NotEqual(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[0]),
				"Different pod namespace should change the key")

			podCopy = pod.DeepCopy()
			podCopy.Spec.Containers[0].Image = "foo/image:v2"
			assert.NotEqual(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[0]),
				"Updating the container image should change the key")

			podCopy = pod.DeepCopy()
			c := &podCopy.Spec.Containers[0]
			if c.Resources.Requests == nil {
				c.Resources.Requests = v1.ResourceList{}
			}
			c.Resources.Requests[v1.ResourceCPU] = resource.MustParse("200m")
			assert.NotEqual(t, originalKey, GetBackoffKey(podCopy, &podCopy.Spec.Containers[0]),
				"Updating the resources should change the key")
		})
	}
}

func TestToKubeContainer(t *testing.T) {
	tCtx := ktesting.Init(t)
	c := &runtimeapi.Container{
		Id: "test-id",
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    "test-name",
			Attempt: 1,
		},
		Image:    &runtimeapi.ImageSpec{Image: "test-image"},
		ImageId:  "test-image-id",
		ImageRef: "test-image-ref",
		State:    runtimeapi.ContainerState_CONTAINER_RUNNING,
		Annotations: map[string]string{
			containerHashLabel: "1234",
		},
	}
	expect := &kubecontainer.Container{
		ID: kubecontainer.ContainerID{
			Type: runtimetesting.FakeRuntimeName,
			ID:   "test-id",
		},
		Name:                "test-name",
		ImageID:             "test-image-id",
		ImageRef:            "test-image-ref",
		Image:               "test-image",
		ImageRuntimeHandler: "",
		Hash:                uint64(0x1234),
		State:               kubecontainer.ContainerStateRunning,
	}

	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)
	got, err := m.toKubeContainer(tCtx, c)
	assert.NoError(t, err)
	assert.Equal(t, expect, got)

	// unable to convert a nil pointer to a runtime container
	_, err = m.toKubeContainer(tCtx, nil)
	assert.Error(t, err)
	_, err = m.sandboxToKubeContainer(nil)
	assert.Error(t, err)
}

func TestToKubeContainerWithRuntimeHandlerInImageSpecCri(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, true)
	tCtx := ktesting.Init(t)
	c := &runtimeapi.Container{
		Id: "test-id",
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    "test-name",
			Attempt: 1,
		},
		Image:    &runtimeapi.ImageSpec{Image: "test-image", RuntimeHandler: "test-runtimeHandler"},
		ImageId:  "test-image-id",
		ImageRef: "test-image-ref",
		State:    runtimeapi.ContainerState_CONTAINER_RUNNING,
		Annotations: map[string]string{
			containerHashLabel: "1234",
		},
	}
	expect := &kubecontainer.Container{
		ID: kubecontainer.ContainerID{
			Type: runtimetesting.FakeRuntimeName,
			ID:   "test-id",
		},
		Name:                "test-name",
		ImageID:             "test-image-id",
		ImageRef:            "test-image-ref",
		Image:               "test-image",
		ImageRuntimeHandler: "test-runtimeHandler",
		Hash:                uint64(0x1234),
		State:               kubecontainer.ContainerStateRunning,
	}

	_, _, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)
	got, err := m.toKubeContainer(tCtx, c)
	assert.NoError(t, err)
	assert.Equal(t, expect, got)

	// unable to convert a nil pointer to a runtime container
	_, err = m.toKubeContainer(tCtx, nil)
	assert.Error(t, err)
	_, err = m.sandboxToKubeContainer(nil)
	assert.Error(t, err)
}

func TestGetImageUser(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, i, m, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	type image struct {
		name     string
		uid      *runtimeapi.Int64Value
		username string
	}

	type imageUserValues struct {
		// getImageUser can return (*int64)(nil) so comparing with *uid will break
		// type cannot be *int64 as Golang does not allow to take the address of a numeric constant"
		uid      interface{}
		username string
		err      error
	}

	tests := []struct {
		description             string
		originalImage           image
		expectedImageUserValues imageUserValues
	}{
		{
			"image without username and uid should return (new(int64), \"\", nil)",
			image{
				name:     "test-image-ref1",
				uid:      (*runtimeapi.Int64Value)(nil),
				username: "",
			},
			imageUserValues{
				uid:      int64(0),
				username: "",
				err:      nil,
			},
		},
		{
			"image with username and no uid should return ((*int64)nil, imageStatus.Username, nil)",
			image{
				name:     "test-image-ref2",
				uid:      (*runtimeapi.Int64Value)(nil),
				username: "testUser",
			},
			imageUserValues{
				uid:      (*int64)(nil),
				username: "testUser",
				err:      nil,
			},
		},
		{
			"image with uid should return (*int64, \"\", nil)",
			image{
				name: "test-image-ref3",
				uid: &runtimeapi.Int64Value{
					Value: 2,
				},
				username: "whatever",
			},
			imageUserValues{
				uid:      int64(2),
				username: "",
				err:      nil,
			},
		},
	}

	i.SetFakeImages([]string{"test-image-ref1", "test-image-ref2", "test-image-ref3"})
	for j, test := range tests {
		tCtx := ktesting.Init(t)
		i.Images[test.originalImage.name].Username = test.originalImage.username
		i.Images[test.originalImage.name].Uid = test.originalImage.uid

		uid, username, err := m.getImageUser(tCtx, test.originalImage.name)
		assert.NoError(t, err, "TestCase[%d]", j)

		if test.expectedImageUserValues.uid == (*int64)(nil) {
			assert.Equal(t, test.expectedImageUserValues.uid, uid, "TestCase[%d]", j)
		} else {
			assert.Equal(t, test.expectedImageUserValues.uid, *uid, "TestCase[%d]", j)
		}
		assert.Equal(t, test.expectedImageUserValues.username, username, "TestCase[%d]", j)
	}
}

func TestToRuntimeProtocol(t *testing.T) {
	tCtx := ktesting.Init(t)
	logger := klog.FromContext(tCtx)
	for _, test := range []struct {
		name     string
		protocol string
		expected runtimeapi.Protocol
	}{
		{
			name:     "TCP protocol",
			protocol: "TCP",
			expected: runtimeapi.Protocol_TCP,
		},
		{
			name:     "UDP protocol",
			protocol: "UDP",
			expected: runtimeapi.Protocol_UDP,
		},
		{
			name:     "SCTP protocol",
			protocol: "SCTP",
			expected: runtimeapi.Protocol_SCTP,
		},
		{
			name:     "unknown protocol",
			protocol: "unknown",
			expected: runtimeapi.Protocol_TCP,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if result := toRuntimeProtocol(logger, v1.Protocol(test.protocol)); result != test.expected {
				t.Errorf("expected %d but got %d", test.expected, result)
			}
		})
	}
}

func TestToKubeContainerState(t *testing.T) {
	for _, test := range []struct {
		name     string
		state    int32
		expected kubecontainer.State
	}{
		{
			name:     "container created",
			state:    0,
			expected: kubecontainer.ContainerStateCreated,
		},
		{
			name:     "container running",
			state:    1,
			expected: kubecontainer.ContainerStateRunning,
		},
		{
			name:     "container exited",
			state:    2,
			expected: kubecontainer.ContainerStateExited,
		},
		{
			name:     "unknown state",
			state:    3,
			expected: kubecontainer.ContainerStateUnknown,
		},
		{
			name:     "not supported state",
			state:    4,
			expected: kubecontainer.ContainerStateUnknown,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			if result := toKubeContainerState(runtimeapi.ContainerState(test.state)); result != test.expected {
				t.Errorf("expected %s but got %s", test.expected, result)
			}
		})
	}
}

func TestGetAppArmorProfile(t *testing.T) {
	tests := []struct {
		name               string
		podProfile         *v1.AppArmorProfile
		expectedProfile    *runtimeapi.SecurityProfile
		expectedOldProfile string
		expectError        bool
	}{{
		name:            "no appArmor",
		expectedProfile: nil,
	}, {
		name:       "runtime default",
		podProfile: &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeRuntimeDefault},
		expectedProfile: &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
		},
		expectedOldProfile: "runtime/default",
	}, {
		name:       "unconfined",
		podProfile: &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeUnconfined},
		expectedProfile: &runtimeapi.SecurityProfile{
			ProfileType: runtimeapi.SecurityProfile_Unconfined,
		},
		expectedOldProfile: "unconfined",
	}, {
		name: "localhost",
		podProfile: &v1.AppArmorProfile{
			Type:             v1.AppArmorProfileTypeLocalhost,
			LocalhostProfile: ptr.To("test"),
		},
		expectedProfile: &runtimeapi.SecurityProfile{
			ProfileType:  runtimeapi.SecurityProfile_Localhost,
			LocalhostRef: "test",
		},
		expectedOldProfile: "localhost/test",
	}, {
		name: "invalid localhost",
		podProfile: &v1.AppArmorProfile{
			Type: v1.AppArmorProfileTypeLocalhost,
		},
		expectError: true,
	}, {
		name: "invalid type",
		podProfile: &v1.AppArmorProfile{
			Type: "foo",
		},
		expectError: true,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
				},
				Spec: v1.PodSpec{
					SecurityContext: &v1.PodSecurityContext{
						AppArmorProfile: test.podProfile,
					},
					Containers: []v1.Container{{Name: "foo"}},
				},
			}

			actual, actualOld, err := getAppArmorProfile(&pod, &pod.Spec.Containers[0])

			if test.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}

			assert.Equal(t, test.expectedProfile, actual, "AppArmor profile")
			assert.Equal(t, test.expectedOldProfile, actualOld, "old (deprecated) profile string")
		})
	}
}

func TestMergeResourceConfig(t *testing.T) {
	tests := []struct {
		name     string
		source   *cm.ResourceConfig
		update   *cm.ResourceConfig
		expected *cm.ResourceConfig
	}{
		{
			name:   "merge all fields",
			source: &cm.ResourceConfig{Memory: ptr.To[int64](1024), CPUShares: ptr.To[uint64](2)},
			update: &cm.ResourceConfig{Memory: ptr.To[int64](2048), CPUQuota: ptr.To[int64](5000)},
			expected: &cm.ResourceConfig{
				Memory:    ptr.To[int64](2048),
				CPUShares: ptr.To[uint64](2),
				CPUQuota:  ptr.To[int64](5000),
			},
		},
		{
			name:   "merge HugePageLimit and Unified",
			source: &cm.ResourceConfig{HugePageLimit: map[int64]int64{2048: 1024}, Unified: map[string]string{"key1": "value1"}},
			update: &cm.ResourceConfig{HugePageLimit: map[int64]int64{4096: 2048}, Unified: map[string]string{"key1": "newValue1", "key2": "value2"}},
			expected: &cm.ResourceConfig{
				HugePageLimit: map[int64]int64{2048: 1024, 4096: 2048},
				Unified:       map[string]string{"key1": "newValue1", "key2": "value2"},
			},
		},
		{
			name:   "update nil source",
			source: nil,
			update: &cm.ResourceConfig{Memory: ptr.To[int64](4096)},
			expected: &cm.ResourceConfig{
				Memory: ptr.To[int64](4096),
			},
		},
		{
			name:   "update nil update",
			source: &cm.ResourceConfig{Memory: ptr.To[int64](1024)},
			update: nil,
			expected: &cm.ResourceConfig{
				Memory: ptr.To[int64](1024),
			},
		},
		{
			name:   "update empty source",
			source: &cm.ResourceConfig{},
			update: &cm.ResourceConfig{Memory: ptr.To[int64](8192)},
			expected: &cm.ResourceConfig{
				Memory: ptr.To[int64](8192),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			merged := mergeResourceConfig(tt.source, tt.update)

			assert.Equal(t, tt.expected, merged)
		})
	}
}

func TestConvertResourceConfigToLinuxContainerResources(t *testing.T) {
	resCfg := &cm.ResourceConfig{
		Memory:        ptr.To[int64](2048),
		CPUShares:     ptr.To[uint64](2),
		CPUPeriod:     ptr.To[uint64](10000),
		CPUQuota:      ptr.To[int64](5000),
		HugePageLimit: map[int64]int64{4096: 2048},
		Unified:       map[string]string{"key1": "value1"},
	}

	lcr := convertResourceConfigToLinuxContainerResources(resCfg)

	assert.Equal(t, int64(*resCfg.CPUPeriod), lcr.CpuPeriod)
	assert.Equal(t, *resCfg.CPUQuota, lcr.CpuQuota)
	assert.Equal(t, int64(*resCfg.CPUShares), lcr.CpuShares)
	assert.Equal(t, *resCfg.Memory, lcr.MemoryLimitInBytes)
	assert.Equal(t, resCfg.Unified, lcr.Unified)
}
