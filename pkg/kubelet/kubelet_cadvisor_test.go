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

package kubelet

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestGetContainerInfo(t *testing.T) {
	cadvisorApiFailure := fmt.Errorf("cAdvisor failure")
	runtimeError := fmt.Errorf("List containers error")
	tests := []struct {
		name                      string
		containerID               string
		containerPath             string
		cadvisorContainerInfo     cadvisorapi.ContainerInfo
		runtimeError              error
		podList                   []*kubecontainertest.FakePod
		requestedPodFullName      string
		requestedPodUid           types.UID
		requestedContainerName    string
		expectDockerContainerCall bool
		mockError                 error
		expectedError             error
		expectStats               bool
	}{
		{
			name:          "get container info",
			containerID:   "ab2cdf",
			containerPath: "/docker/ab2cdf",
			cadvisorContainerInfo: cadvisorapi.ContainerInfo{
				ContainerReference: cadvisorapi.ContainerReference{
					Name: "/docker/ab2cdf",
				},
			},
			runtimeError: nil,
			podList: []*kubecontainertest.FakePod{
				{
					Pod: &kubecontainer.Pod{
						ID:        "12345678",
						Name:      "qux",
						Namespace: "ns",
						Containers: []*kubecontainer.Container{
							{
								Name: "foo",
								ID:   kubecontainer.ContainerID{Type: "test", ID: "ab2cdf"},
							},
						},
					},
				},
			},
			requestedPodFullName:      "qux_ns",
			requestedPodUid:           "",
			requestedContainerName:    "foo",
			expectDockerContainerCall: true,
			mockError:                 nil,
			expectedError:             nil,
			expectStats:               true,
		},
		{
			name:                  "get container info when cadvisor failed",
			containerID:           "ab2cdf",
			containerPath:         "/docker/ab2cdf",
			cadvisorContainerInfo: cadvisorapi.ContainerInfo{},
			runtimeError:          nil,
			podList: []*kubecontainertest.FakePod{
				{
					Pod: &kubecontainer.Pod{
						ID:        "uuid",
						Name:      "qux",
						Namespace: "ns",
						Containers: []*kubecontainer.Container{
							{
								Name: "foo",
								ID:   kubecontainer.ContainerID{Type: "test", ID: "ab2cdf"},
							},
						},
					},
				},
			},
			requestedPodFullName:      "qux_ns",
			requestedPodUid:           "uuid",
			requestedContainerName:    "foo",
			expectDockerContainerCall: true,
			mockError:                 cadvisorApiFailure,
			expectedError:             cadvisorApiFailure,
			expectStats:               false,
		},
		{
			name:                      "get container info on non-existent container",
			containerID:               "",
			containerPath:             "",
			cadvisorContainerInfo:     cadvisorapi.ContainerInfo{},
			runtimeError:              nil,
			podList:                   []*kubecontainertest.FakePod{},
			requestedPodFullName:      "qux",
			requestedPodUid:           "",
			requestedContainerName:    "foo",
			expectDockerContainerCall: false,
			mockError:                 nil,
			expectedError:             kubecontainer.ErrContainerNotFound,
			expectStats:               false,
		},
		{
			name:                   "get container info when container runtime failed",
			containerID:            "",
			containerPath:          "",
			cadvisorContainerInfo:  cadvisorapi.ContainerInfo{},
			runtimeError:           runtimeError,
			podList:                []*kubecontainertest.FakePod{},
			requestedPodFullName:   "qux",
			requestedPodUid:        "",
			requestedContainerName: "foo",
			mockError:              nil,
			expectedError:          runtimeError,
			expectStats:            false,
		},
		{
			name:                   "get container info with no containers",
			containerID:            "",
			containerPath:          "",
			cadvisorContainerInfo:  cadvisorapi.ContainerInfo{},
			runtimeError:           nil,
			podList:                []*kubecontainertest.FakePod{},
			requestedPodFullName:   "qux_ns",
			requestedPodUid:        "",
			requestedContainerName: "foo",
			mockError:              nil,
			expectedError:          kubecontainer.ErrContainerNotFound,
			expectStats:            false,
		},
		{
			name:                  "get container info with no matching containers",
			containerID:           "",
			containerPath:         "",
			cadvisorContainerInfo: cadvisorapi.ContainerInfo{},
			runtimeError:          nil,
			podList: []*kubecontainertest.FakePod{
				{
					Pod: &kubecontainer.Pod{
						ID:        "12345678",
						Name:      "qux",
						Namespace: "ns",
						Containers: []*kubecontainer.Container{
							{
								Name: "bar",
								ID:   kubecontainer.ContainerID{Type: "test", ID: "fakeID"},
							},
						},
					},
				},
			},
			requestedPodFullName:   "qux_ns",
			requestedPodUid:        "",
			requestedContainerName: "foo",
			mockError:              nil,
			expectedError:          kubecontainer.ErrContainerNotFound,
			expectStats:            false,
		},
	}

	for _, tc := range tests {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnablec */)
		defer testKubelet.Cleanup()
		fakeRuntime := testKubelet.fakeRuntime
		kubelet := testKubelet.kubelet
		cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
		mockCadvisor := testKubelet.fakeCadvisor
		if tc.expectDockerContainerCall {
			mockCadvisor.On("DockerContainer", tc.containerID, cadvisorReq).Return(tc.cadvisorContainerInfo, tc.mockError)
		}
		fakeRuntime.Err = tc.runtimeError
		fakeRuntime.PodList = tc.podList

		stats, err := kubelet.GetContainerInfo(tc.requestedPodFullName, tc.requestedPodUid, tc.requestedContainerName, cadvisorReq)
		assert.Equal(t, tc.expectedError, err)

		if tc.expectStats {
			require.NotNil(t, stats)
		}
		mockCadvisor.AssertExpectations(t)
	}
}

func TestGetRawContainerInfoRoot(t *testing.T) {
	containerPath := "/"
	containerInfo := &cadvisorapi.ContainerInfo{
		ContainerReference: cadvisorapi.ContainerReference{
			Name: containerPath,
		},
	}
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	_, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, false)
	assert.NoError(t, err)
	mockCadvisor.AssertExpectations(t)
}

func TestGetRawContainerInfoSubcontainers(t *testing.T) {
	containerPath := "/kubelet"
	containerInfo := map[string]*cadvisorapi.ContainerInfo{
		containerPath: {
			ContainerReference: cadvisorapi.ContainerReference{
				Name: containerPath,
			},
		},
		"/kubelet/sub": {
			ContainerReference: cadvisorapi.ContainerReference{
				Name: "/kubelet/sub",
			},
		},
	}
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor.On("SubcontainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	result, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, true)
	assert.NoError(t, err)
	assert.Len(t, result, 2)
	mockCadvisor.AssertExpectations(t)
}

func TestHasDedicatedImageFs(t *testing.T) {
	testCases := map[string]struct {
		imageFsInfo cadvisorapiv2.FsInfo
		rootFsInfo  cadvisorapiv2.FsInfo
		expected    bool
	}{
		"has-dedicated-image-fs": {
			imageFsInfo: cadvisorapiv2.FsInfo{Device: "123"},
			rootFsInfo:  cadvisorapiv2.FsInfo{Device: "456"},
			expected:    true,
		},
		"has-unified-image-fs": {
			imageFsInfo: cadvisorapiv2.FsInfo{Device: "123"},
			rootFsInfo:  cadvisorapiv2.FsInfo{Device: "123"},
			expected:    false,
		},
	}
	for testName, testCase := range testCases {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		defer testKubelet.Cleanup()
		kubelet := testKubelet.kubelet
		mockCadvisor := testKubelet.fakeCadvisor
		mockCadvisor.On("Start").Return(nil)
		mockCadvisor.On("ImagesFsInfo").Return(testCase.imageFsInfo, nil)
		mockCadvisor.On("RootFsInfo").Return(testCase.rootFsInfo, nil)
		actual, err := kubelet.HasDedicatedImageFs()
		assert.NoError(t, err, "test [%s]", testName)
		assert.Equal(t, testCase.expected, actual, "test [%s]", testName)
	}
}
