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
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	runtimetesting "k8s.io/cri-api/pkg/apis/testing"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilpointer "k8s.io/utils/pointer"
)

func TestStableKey(t *testing.T) {
	container := &v1.Container{
		Name:  "test_container",
		Image: "foo/image:v1",
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test_pod",
			Namespace: "test_pod_namespace",
			UID:       "test_pod_uid",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{*container},
		},
	}
	oldKey := getStableKey(pod, container)

	// Updating the container image should change the key.
	container.Image = "foo/image:v2"
	newKey := getStableKey(pod, container)
	assert.NotEqual(t, oldKey, newKey)
}

func TestToKubeContainer(t *testing.T) {
	c := &runtimeapi.Container{
		Id: "test-id",
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    "test-name",
			Attempt: 1,
		},
		Image:    &runtimeapi.ImageSpec{Image: "test-image"},
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
		Name:    "test-name",
		ImageID: "test-image-ref",
		Image:   "test-image",
		Hash:    uint64(0x1234),
		State:   kubecontainer.ContainerStateRunning,
	}

	_, _, m, err := createTestRuntimeManager()
	assert.NoError(t, err)
	got, err := m.toKubeContainer(c)
	assert.NoError(t, err)
	assert.Equal(t, expect, got)
}

func TestGetImageUser(t *testing.T) {
	_, i, m, err := createTestRuntimeManager()
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
		i.Images[test.originalImage.name].Username = test.originalImage.username
		i.Images[test.originalImage.name].Uid = test.originalImage.uid

		uid, username, err := m.getImageUser(test.originalImage.name)
		assert.NoError(t, err, "TestCase[%d]", j)

		if test.expectedImageUserValues.uid == (*int64)(nil) {
			assert.Equal(t, test.expectedImageUserValues.uid, uid, "TestCase[%d]", j)
		} else {
			assert.Equal(t, test.expectedImageUserValues.uid, *uid, "TestCase[%d]", j)
		}
		assert.Equal(t, test.expectedImageUserValues.username, username, "TestCase[%d]", j)
	}
}

func TestFieldProfile(t *testing.T) {
	tests := []struct {
		description     string
		scmpProfile     *v1.SeccompProfile
		rootPath        string
		expectedProfile string
	}{
		{
			description:     "no seccompProfile should return empty",
			expectedProfile: "",
		},
		{
			description: "type localhost without profile should return empty",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeLocalhost,
			},
			expectedProfile: "",
		},
		{
			description: "unknown type should return empty",
			scmpProfile: &v1.SeccompProfile{
				Type: "",
			},
			expectedProfile: "",
		},
		{
			description: "SeccompProfileTypeRuntimeDefault should return runtime/default",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeRuntimeDefault,
			},
			expectedProfile: "runtime/default",
		},
		{
			description: "SeccompProfileTypeUnconfined should return unconfined",
			scmpProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeUnconfined,
			},
			expectedProfile: "unconfined",
		},
		{
			description: "SeccompProfileTypeLocalhost should return unconfined",
			scmpProfile: &v1.SeccompProfile{
				Type:             v1.SeccompProfileTypeLocalhost,
				LocalhostProfile: utilpointer.StringPtr("profile.json"),
			},
			rootPath:        "/test/",
			expectedProfile: "localhost//test/profile.json",
		},
	}

	for i, test := range tests {
		seccompProfile := fieldProfile(test.scmpProfile, test.rootPath)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGetSeccompProfilePath(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile string
	}{
		{
			description:     "no seccomp should return empty",
			expectedProfile: "",
		},
		{
			description:     "annotations: no seccomp with containerName should return empty",
			containerName:   "container1",
			expectedProfile: "",
		},
		{
			description: "annotations: pod runtime/default seccomp profile should return runtime/default",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.SeccompProfileRuntimeDefault,
			},
			expectedProfile: "runtime/default",
		},
		{
			description: "annotations: pod docker/default seccomp profile should return docker/default",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.DeprecatedSeccompProfileDockerDefault,
			},
			expectedProfile: "docker/default",
		},
		{
			description: "annotations: pod runtime/default seccomp profile with containerName should return runtime/default",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.SeccompProfileRuntimeDefault,
			},
			containerName:   "container1",
			expectedProfile: "runtime/default",
		},
		{
			description: "annotations: pod docker/default seccomp profile with containerName should return docker/default",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.DeprecatedSeccompProfileDockerDefault,
			},
			containerName:   "container1",
			expectedProfile: "docker/default",
		},
		{
			description: "annotations: pod unconfined seccomp profile should return unconfined",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.SeccompProfileNameUnconfined,
			},
			expectedProfile: "unconfined",
		},
		{
			description: "annotations: pod unconfined seccomp profile with containerName should return unconfined",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: v1.SeccompProfileNameUnconfined,
			},
			containerName:   "container1",
			expectedProfile: "unconfined",
		},
		{
			description: "annotations: pod localhost seccomp profile should return local profile path",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/chmod.json",
			},
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "chmod.json"),
		},
		{
			description: "annotations: pod localhost seccomp profile with containerName should return local profile path",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/chmod.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "chmod.json"),
		},
		{
			description: "annotations: container localhost seccomp profile with containerName should return local profile path",
			annotation: map[string]string{
				v1.SeccompContainerAnnotationKeyPrefix + "container1": "localhost/chmod.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "chmod.json"),
		},
		{
			description: "annotations: container localhost seccomp profile should override pod profile",
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey:                            v1.SeccompProfileNameUnconfined,
				v1.SeccompContainerAnnotationKeyPrefix + "container1": "localhost/chmod.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "chmod.json"),
		},
		{
			description: "annotations: container localhost seccomp profile with unmatched containerName should return empty",
			annotation: map[string]string{
				v1.SeccompContainerAnnotationKeyPrefix + "container1": "localhost/chmod.json",
			},
			containerName:   "container2",
			expectedProfile: "",
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: "unconfined",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "filename"),
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns empty",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: "",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns empty",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: "",
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "filename2"),
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: "runtime/default",
		},
		{
			description: "prioritise container field over container annotation, pod field and pod annotation",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			containerSc: &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-cont-profile.json")}},
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey:                            "localhost/annota-pod-profile.json",
				v1.SeccompContainerAnnotationKeyPrefix + "container1": "localhost/annota-cont-profile.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "field-cont-profile.json"),
		},
		{
			description: "prioritise container annotation over pod field",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey:                            "localhost/annota-pod-profile.json",
				v1.SeccompContainerAnnotationKeyPrefix + "container1": "localhost/annota-cont-profile.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "annota-cont-profile.json"),
		},
		{
			description: "prioritise pod field over pod annotation",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			annotation: map[string]string{
				v1.SeccompPodAnnotationKey: "localhost/annota-pod-profile.json",
			},
			containerName:   "container1",
			expectedProfile: "localhost/" + filepath.Join(fakeSeccompProfileRoot, "field-pod-profile.json"),
		},
	}

	for i, test := range tests {
		seccompProfile := m.getSeccompProfilePath(test.annotation, test.containerName, test.podSc, test.containerSc)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func TestGetSeccompProfile(t *testing.T) {
	_, _, m, err := createTestRuntimeManager()
	require.NoError(t, err)

	unconfinedProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_Unconfined,
	}

	runtimeDefaultProfile := &runtimeapi.SecurityProfile{
		ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
	}

	tests := []struct {
		description     string
		annotation      map[string]string
		podSc           *v1.PodSecurityContext
		containerSc     *v1.SecurityContext
		containerName   string
		expectedProfile *runtimeapi.SecurityProfile
	}{
		{
			description:     "no seccomp should return unconfined",
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "pod seccomp profile set to unconfined returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to unconfined returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeRuntimeDefault returns runtime/default",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description: "pod seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: filepath.Join(fakeSeccompProfileRoot, "filename"),
			},
		},
		{
			description:     "pod seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
		},
		{
			description:     "container seccomp profile set to SeccompProfileTypeLocalhost with empty LocalhostProfile returns unconfined",
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost}},
			expectedProfile: unconfinedProfile,
		},
		{
			description: "container seccomp profile set to SeccompProfileTypeLocalhost returns 'localhost/' + LocalhostProfile",
			containerSc: &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("filename2")}},
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: filepath.Join(fakeSeccompProfileRoot, "filename2"),
			},
		},
		{
			description:     "prioritise container field over pod field",
			podSc:           &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}},
			containerSc:     &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault}},
			expectedProfile: runtimeDefaultProfile,
		},
		{
			description:   "prioritise container field over pod field",
			podSc:         &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-pod-profile.json")}},
			containerSc:   &v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: getLocal("field-cont-profile.json")}},
			containerName: "container1",
			expectedProfile: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: filepath.Join(fakeSeccompProfileRoot, "field-cont-profile.json"),
			},
		},
	}

	for i, test := range tests {
		seccompProfile := m.getSeccompProfile(test.annotation, test.containerName, test.podSc, test.containerSc)
		assert.Equal(t, test.expectedProfile, seccompProfile, "TestCase[%d]: %s", i, test.description)
	}
}

func getLocal(v string) *string {
	return &v
}

func TestNamespacesForPod(t *testing.T) {
	for desc, test := range map[string]struct {
		input    *v1.Pod
		expected *runtimeapi.NamespaceOption
	}{
		"nil pod -> default v1 namespaces": {
			nil,
			&runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		"v1.Pod default namespaces": {
			&v1.Pod{},
			&runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
		"Host Namespaces": {
			&v1.Pod{
				Spec: v1.PodSpec{
					HostIPC:     true,
					HostNetwork: true,
					HostPID:     true,
				},
			},
			&runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_NODE,
				Network: runtimeapi.NamespaceMode_NODE,
				Pid:     runtimeapi.NamespaceMode_NODE,
			},
		},
		"Shared Process Namespace (feature enabled)": {
			&v1.Pod{
				Spec: v1.PodSpec{
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
			&runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_POD,
			},
		},
		"Shared Process Namespace, redundant flag (feature enabled)": {
			&v1.Pod{
				Spec: v1.PodSpec{
					ShareProcessNamespace: &[]bool{false}[0],
				},
			},
			&runtimeapi.NamespaceOption{
				Ipc:     runtimeapi.NamespaceMode_POD,
				Network: runtimeapi.NamespaceMode_POD,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
			},
		},
	} {
		t.Logf("TestCase: %s", desc)
		actual := namespacesForPod(test.input)
		assert.Equal(t, test.expected, actual)
	}
}
