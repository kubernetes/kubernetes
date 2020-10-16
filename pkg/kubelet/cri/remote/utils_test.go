/*
Copyright 2021 The Kubernetes Authors.

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

package remote

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestVerifySandboxStatus(t *testing.T) {

	testcases := []struct {
		description      string
		podSandboxStatus *runtimeapi.PodSandboxStatus
		expectedResult   error
	}{{
		description:      "verifySandboxStatus succeeds",
		podSandboxStatus: getPodSandboxStatus(true, true, true, true),
		expectedResult:   nil,
	}, {
		description:      "verifySandboxStatus fails due to lack of Id",
		podSandboxStatus: getPodSandboxStatus(false, true, true, true),
		expectedResult:   fmt.Errorf("status.Id is not set"),
	}, {
		description:      "verifySandboxStatus fails due to lack of metadata",
		podSandboxStatus: getPodSandboxStatus(true, false, true, true),
		expectedResult:   fmt.Errorf("status.Metadata is not set"),
	}, {
		description:      "verifySandboxStatus fails due to lack of CreatedAt",
		podSandboxStatus: getPodSandboxStatus(true, true, false, true),
		expectedResult:   fmt.Errorf("status.CreatedAt is not set"),
	}, {
		description:      "verifySandboxStatus fails due to lack of name",
		podSandboxStatus: getPodSandboxStatus(true, true, true, false),
		expectedResult:   fmt.Errorf("metadata.Name, metadata.Namespace or metadata.Uid is not in metadata \"&PodSandboxMetadata{Name:,Uid:100,Namespace:foo-ns,Attempt:10,}\""),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			result := verifySandboxStatus(tc.podSandboxStatus)
			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func getPodSandboxStatus(validId, validMetadata, validCreatedAt, validName bool) *runtimeapi.PodSandboxStatus {
	id := "12345678"
	metadata := &runtimeapi.PodSandboxMetadata{
		Name:          "testname",
		Uid:           "100",
		Namespace:     "foo-ns",
		Attempt:       10,
		XXX_sizecache: 100,
	}
	createAt := int64(6000)

	if !validId {
		id = ""
	}
	if !validMetadata {
		metadata = nil
	}
	if !validCreatedAt {
		createAt = 0
	}
	if !validName {
		metadata.Name = ""
	}
	return &runtimeapi.PodSandboxStatus{
		Id:        id,
		Metadata:  metadata,
		State:     runtimeapi.PodSandboxState_SANDBOX_READY,
		CreatedAt: createAt,
		Network: &runtimeapi.PodSandboxNetworkStatus{
			Ip:            "192.168.1.100",
			XXX_sizecache: 100,
		},
		Linux: &runtimeapi.LinuxPodSandboxStatus{
			XXX_sizecache: 100,
		},
		RuntimeHandler: "foo-header",
		XXX_sizecache:  100,
	}
}

func TestVerifyContainerStatus(t *testing.T) {

	testcases := []struct {
		description     string
		containerStatus *runtimeapi.ContainerStatus
		expectedResult  error
	}{{
		description:     "verifyContainerStatus succeeds",
		containerStatus: getContainerStatus(true, true, true, true, true, true),
		expectedResult:  nil,
	}, {
		description:     "verifyContainerStatus fails due to lack of Id",
		containerStatus: getContainerStatus(false, true, true, true, true, true),
		expectedResult:  fmt.Errorf("status.Id is not set"),
	}, {
		description:     "verifyContainerStatus fails due to lack of metadata",
		containerStatus: getContainerStatus(true, false, true, true, true, true),
		expectedResult:  fmt.Errorf("status.Metadata is not set"),
	}, {
		description:     "verifyContainerStatus fails due to lack of CreatedAt",
		containerStatus: getContainerStatus(true, true, false, true, true, true),
		expectedResult:  fmt.Errorf("status.CreatedAt is not set"),
	}, {
		description:     "verifyContainerStatus fails due to lack of ImageRef",
		containerStatus: getContainerStatus(true, true, true, false, true, true),
		expectedResult:  fmt.Errorf("status.ImageRef is not set"),
	}, {
		description:     "verifyContainerStatus fails due to lack of Image",
		containerStatus: getContainerStatus(true, true, true, true, false, true),
		expectedResult:  fmt.Errorf("status.Image is not set"),
	}, {
		description:     "verifyContainerStatus fails due to lack of name",
		containerStatus: getContainerStatus(true, true, true, true, true, false),
		expectedResult:  fmt.Errorf("metadata.Name is not in metadata \"&ContainerMetadata{Name:,Attempt:10,}\""),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			result := verifyContainerStatus(tc.containerStatus)
			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func getContainerStatus(validId, validMetadata, validCreatedAt, validImageRef, validImage, validName bool) *runtimeapi.ContainerStatus {
	id := "12345678"
	metadata := &runtimeapi.ContainerMetadata{
		Name:                 "testname",
		Attempt:              10,
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_sizecache:        100,
	}
	createAt := int64(6000)
	imageRef := "123456789"
	image := &runtimeapi.ImageSpec{
		Image:                "testImage",
		Annotations:          map[string]string{},
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_sizecache:        100,
	}

	if !validId {
		id = ""
	}
	if !validMetadata {
		metadata = nil
	}
	if !validCreatedAt {
		createAt = 0
	}
	if !validImageRef {
		imageRef = ""
	}
	if !validImage {
		image = nil
	}
	if !validName {
		metadata.Name = ""
	}

	return &runtimeapi.ContainerStatus{
		Id:                   id,
		Metadata:             metadata,
		State:                runtimeapi.ContainerState_CONTAINER_RUNNING,
		CreatedAt:            createAt,
		StartedAt:            6000,
		FinishedAt:           0,
		ExitCode:             0,
		Image:                image,
		ImageRef:             imageRef,
		Reason:               "",
		Message:              "Running",
		Labels:               map[string]string{},
		Annotations:          map[string]string{},
		Mounts:               []*runtimeapi.Mount{},
		LogPath:              "",
		XXX_NoUnkeyedLiteral: struct{}{},
		XXX_sizecache:        100,
	}
}
