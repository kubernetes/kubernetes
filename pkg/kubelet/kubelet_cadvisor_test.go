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

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestGetContainerInfo(t *testing.T) {
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
	containerInfo := cadvisorapi.ContainerInfo{
		ContainerReference: cadvisorapi.ContainerReference{
			Name: containerPath,
		},
	}

	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	fakeRuntime := testKubelet.fakeRuntime
	kubelet := testKubelet.kubelet
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)
	fakeRuntime.PodList = []*kubecontainertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        "12345678",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{
					Name: "foo",
					ID:   kubecontainer.ContainerID{Type: "test", ID: containerID},
				},
			},
		}},
	}
	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", cadvisorReq)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetRawContainerInfoRoot(t *testing.T) {
	containerPath := "/"
	containerInfo := &cadvisorapi.ContainerInfo{
		ContainerReference: cadvisorapi.ContainerReference{
			Name: containerPath,
		},
	}
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	_, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
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
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor.On("SubcontainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	result, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, true)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("Expected 2 elements, received: %#v", result)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWhenCadvisorFailed(t *testing.T) {
	containerID := "ab2cdf"
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	cadvisorApiFailure := fmt.Errorf("cAdvisor failure")
	containerInfo := cadvisorapi.ContainerInfo{}
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, cadvisorApiFailure)
	fakeRuntime.PodList = []*kubecontainertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        "uuid",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{Name: "foo",
					ID: kubecontainer.ContainerID{Type: "test", ID: containerID},
				},
			},
		}},
	}
	stats, err := kubelet.GetContainerInfo("qux_ns", "uuid", "foo", cadvisorReq)
	if stats != nil {
		t.Errorf("non-nil stats on error")
	}
	if err == nil {
		t.Errorf("expect error but received nil error")
		return
	}
	if err.Error() != cadvisorApiFailure.Error() {
		t.Errorf("wrong error message. expect %v, got %v", cadvisorApiFailure, err)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoOnNonExistContainer(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*kubecontainertest.FakePod{}

	stats, _ := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWhenContainerRuntimeFailed(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	expectedErr := fmt.Errorf("List containers error")
	fakeRuntime.Err = expectedErr

	stats, err := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if err == nil {
		t.Errorf("expected error from dockertools, got none")
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("expected error %v got %v", expectedErr.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools failed")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithNoContainers(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor

	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", nil)
	if err == nil {
		t.Errorf("expected error from cadvisor client, got none")
	}
	if err != kubecontainer.ErrContainerNotFound {
		t.Errorf("expected error %v, got %v", kubecontainer.ErrContainerNotFound.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools returned no containers")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithNoMatchingContainers(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	fakeRuntime := testKubelet.fakeRuntime
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime.PodList = []*kubecontainertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        "12345678",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{Name: "bar",
					ID: kubecontainer.ContainerID{Type: "test", ID: "fakeID"},
				},
			}},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", nil)
	if err == nil {
		t.Errorf("Expected error from cadvisor client, got none")
	}
	if err != kubecontainer.ErrContainerNotFound {
		t.Errorf("Expected error %v, got %v", kubecontainer.ErrContainerNotFound.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools returned no containers")
	}
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
		kubelet := testKubelet.kubelet
		mockCadvisor := testKubelet.fakeCadvisor
		mockCadvisor.On("Start").Return(nil)
		mockCadvisor.On("ImagesFsInfo").Return(testCase.imageFsInfo, nil)
		mockCadvisor.On("RootFsInfo").Return(testCase.rootFsInfo, nil)
		actual, err := kubelet.HasDedicatedImageFs()
		if err != nil {
			t.Errorf("case: %s, unexpected error: %v", testName, err)
		}
		if actual != testCase.expected {
			t.Errorf("case: %s, expected: %v, actual: %v", testName, testCase.expected, actual)
		}
	}
}
