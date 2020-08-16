// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func makeExpectedConfig(m *kubeGenericRuntimeManager, pod *v1.Pod, containerIndex int) *runtimeapi.ContainerConfig {
	container := &pod.Spec.Containers[containerIndex]
	podIP := ""
	restartCount := 0
	opts, _, _ := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP, []string{podIP})
	containerLogsPath := buildContainerLogsPath(container.Name, restartCount)
	restartCountUint32 := uint32(restartCount)
	envs := make([]*runtimeapi.KeyValue, len(opts.Envs))

	expectedConfig := &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    container.Name,
			Attempt: restartCountUint32,
		},
		Image:       &runtimeapi.ImageSpec{Image: container.Image},
		Command:     container.Command,
		Args:        []string(nil),
		WorkingDir:  container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount, opts),
		Devices:     makeDevices(opts),
		Mounts:      m.makeMounts(opts, container),
		LogPath:     containerLogsPath,
		Stdin:       container.Stdin,
		StdinOnce:   container.StdinOnce,
		Tty:         container.TTY,
		Linux:       m.generateLinuxContainerConfig(container, pod, new(int64), "", nil),
		Envs:        envs,
	}
	return expectedConfig
}

func TestGenerateContainerConfig(t *testing.T) {
	_, imageService, m, err := createTestRuntimeManager()
	assert.NoError(t, err)

	runAsUser := int64(1000)
	runAsGroup := int64(2000)
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
					SecurityContext: &v1.SecurityContext{
						RunAsUser:  &runAsUser,
						RunAsGroup: &runAsGroup,
					},
				},
			},
		},
	}

	expectedConfig := makeExpectedConfig(m, pod, 0)
	containerConfig, _, err := m.generateContainerConfig(&pod.Spec.Containers[0], pod, 0, "", pod.Spec.Containers[0].Image, []string{}, nil)
	assert.NoError(t, err)
	assert.Equal(t, expectedConfig, containerConfig, "generate container config for kubelet runtime v1.")
	assert.Equal(t, runAsUser, containerConfig.GetLinux().GetSecurityContext().GetRunAsUser().GetValue(), "RunAsUser should be set")
	assert.Equal(t, runAsGroup, containerConfig.GetLinux().GetSecurityContext().GetRunAsGroup().GetValue(), "RunAsGroup should be set")

	runAsRoot := int64(0)
	runAsNonRootTrue := true
	podWithContainerSecurityContext := &v1.Pod{
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
					SecurityContext: &v1.SecurityContext{
						RunAsNonRoot: &runAsNonRootTrue,
						RunAsUser:    &runAsRoot,
					},
				},
			},
		},
	}

	_, _, err = m.generateContainerConfig(&podWithContainerSecurityContext.Spec.Containers[0], podWithContainerSecurityContext, 0, "", podWithContainerSecurityContext.Spec.Containers[0].Image, []string{}, nil)
	assert.Error(t, err)

	imageID, _ := imageService.PullImage(&runtimeapi.ImageSpec{Image: "busybox"}, nil, nil)
	image, _ := imageService.ImageStatus(&runtimeapi.ImageSpec{Image: imageID})

	image.Uid = nil
	image.Username = "test"

	podWithContainerSecurityContext.Spec.Containers[0].SecurityContext.RunAsUser = nil
	podWithContainerSecurityContext.Spec.Containers[0].SecurityContext.RunAsNonRoot = &runAsNonRootTrue

	_, _, err = m.generateContainerConfig(&podWithContainerSecurityContext.Spec.Containers[0], podWithContainerSecurityContext, 0, "", podWithContainerSecurityContext.Spec.Containers[0].Image, []string{}, nil)
	assert.Error(t, err, "RunAsNonRoot should fail for non-numeric username")
}

func TestGetHugepageLimitsFromResources(t *testing.T) {
	var baseHugepage []*runtimeapi.HugepageLimit

	// For each page size, limit to 0.
	for _, pageSize := range cgroupfs.HugePageSizes {
		baseHugepage = append(baseHugepage, &runtimeapi.HugepageLimit{
			PageSize: pageSize,
			Limit:    uint64(0),
		})
	}

	tests := []struct {
		name      string
		resources v1.ResourceRequirements
		expected  []*runtimeapi.HugepageLimit
	}{
		{
			name: "Success2MB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2Mi": resource.MustParse("2Mi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
			},
		},
		{
			name: "Success1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1Gi": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
		},
		{
			name: "Skip2MB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-2MB": resource.MustParse("2Mi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    0,
				},
			},
		},
		{
			name: "Skip1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"hugepages-1GB": resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "1GB",
					Limit:    0,
				},
			},
		},
		{
			name: "Success2MBand1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse("0"),
					"hugepages-2Mi":                 resource.MustParse("2Mi"),
					"hugepages-1Gi":                 resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    2097152,
				},
				{
					PageSize: "1GB",
					Limit:    2147483648,
				},
			},
		},
		{
			name: "Skip2MBand1GB",
			resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse("0"),
					"hugepages-2MB":                 resource.MustParse("2Mi"),
					"hugepages-1GB":                 resource.MustParse("2Gi"),
				},
			},
			expected: []*runtimeapi.HugepageLimit{
				{
					PageSize: "2MB",
					Limit:    0,
				},
				{
					PageSize: "1GB",
					Limit:    0,
				},
			},
		},
	}

	for _, test := range tests {
		// Validate if machine supports hugepage size that used in test case.
		machineHugepageSupport := true
		for _, hugepageLimit := range test.expected {
			hugepageSupport := false
			for _, pageSize := range cgroupfs.HugePageSizes {
				if pageSize == hugepageLimit.PageSize {
					hugepageSupport = true
					break
				}
			}

			if !hugepageSupport {
				machineHugepageSupport = false
				break
			}
		}

		// Case of machine can't support hugepage size
		if !machineHugepageSupport {
			continue
		}

		expectedHugepages := baseHugepage
		for _, hugepage := range test.expected {
			for _, expectedHugepage := range expectedHugepages {
				if expectedHugepage.PageSize == hugepage.PageSize {
					expectedHugepage.Limit = hugepage.Limit
				}
			}
		}

		results := GetHugepageLimitsFromResources(test.resources)
		if !reflect.DeepEqual(expectedHugepages, results) {
			t.Errorf("%s test failed. Expected %v but got %v", test.name, expectedHugepages, results)
		}

		for _, hugepage := range baseHugepage {
			hugepage.Limit = uint64(0)
		}
	}
}

func TestGenerateLinuxContainerConfigNamespaces(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()
	_, _, m, err := createTestRuntimeManager()
	if err != nil {
		t.Fatalf("error creating test RuntimeManager: %v", err)
	}

	for _, tc := range []struct {
		name   string
		pod    *v1.Pod
		target *kubecontainer.ContainerID
		want   *runtimeapi.NamespaceOption
	}{
		{
			"Default namespaces",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
				},
			},
			nil,
			&runtimeapi.NamespaceOption{
				Pid: runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		{
			"PID Namespace POD",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
			nil,
			&runtimeapi.NamespaceOption{
				Pid: runtimeapi.NamespaceMode_POD,
			},
		},
		{
			"PID Namespace TARGET",
			&v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "test"},
					},
				},
			},
			&kubecontainer.ContainerID{Type: "docker", ID: "really-long-id-string"},
			&runtimeapi.NamespaceOption{
				Pid:      runtimeapi.NamespaceMode_TARGET,
				TargetId: "really-long-id-string",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := m.generateLinuxContainerConfig(&tc.pod.Spec.Containers[0], tc.pod, nil, "", tc.target)
			if diff := cmp.Diff(tc.want, got.SecurityContext.NamespaceOptions); diff != "" {
				t.Errorf("%v: diff (-want +got):\n%v", t.Name(), diff)
			}
		})
	}
}
