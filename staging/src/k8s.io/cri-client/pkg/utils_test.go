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

package cri

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func makePodSandboxMetadata(name, namespace, uid string) *runtimeapi.PodSandboxMetadata {
	return &runtimeapi.PodSandboxMetadata{
		Name:      name,
		Namespace: namespace,
		Uid:       uid,
	}
}

func TestVerifySandboxStatus(t *testing.T) {
	ct := int64(1)
	metaWithoutName := makePodSandboxMetadata("", "bar", "1")
	metaWithoutNamespace := makePodSandboxMetadata("foo", "", "1")
	metaWithoutUid := makePodSandboxMetadata("foo", "bar", "")

	statuses := []struct {
		input    *runtimeapi.PodSandboxStatus
		expected error
	}{
		{
			input: &runtimeapi.PodSandboxStatus{
				CreatedAt: ct,
				Metadata:  makePodSandboxMetadata("foo", "bar", "1"),
			},
			expected: fmt.Errorf("status.Id is not set"),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:        "1",
				CreatedAt: ct,
			},
			expected: fmt.Errorf("status.Metadata is not set"),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:        "2",
				CreatedAt: ct,
				Metadata:  metaWithoutName,
			},
			expected: fmt.Errorf("metadata.Name, metadata.Namespace or metadata.Uid is not in metadata %q", metaWithoutName),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:        "3",
				CreatedAt: ct,
				Metadata:  metaWithoutNamespace,
			},
			expected: fmt.Errorf("metadata.Name, metadata.Namespace or metadata.Uid is not in metadata %q", metaWithoutNamespace),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:        "4",
				CreatedAt: ct,
				Metadata:  metaWithoutUid,
			},
			expected: fmt.Errorf("metadata.Name, metadata.Namespace or metadata.Uid is not in metadata %q", metaWithoutUid),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:       "5",
				Metadata: makePodSandboxMetadata("foo", "bar", "1"),
			},
			expected: fmt.Errorf("status.CreatedAt is not set"),
		},
		{
			input: &runtimeapi.PodSandboxStatus{
				Id:        "6",
				CreatedAt: ct,
				Metadata:  makePodSandboxMetadata("foo", "bar", "1"),
			},
			expected: nil,
		},
	}

	for _, status := range statuses {
		actual := verifySandboxStatus(status.input)
		if actual != nil {
			assert.EqualError(t, actual, status.expected.Error())
		} else {
			assert.Nil(t, status.expected)
		}
	}
}

func TestVerifyContainerStatus(t *testing.T) {
	meta := &runtimeapi.ContainerMetadata{Name: "cname", Attempt: 3}
	metaWithoutName := &runtimeapi.ContainerMetadata{Attempt: 3}
	imageSpec := &runtimeapi.ImageSpec{Image: "fimage"}
	imageSpecWithoutImage := &runtimeapi.ImageSpec{}

	statuses := []struct {
		input    *runtimeapi.ContainerStatus
		expected error
	}{
		{
			input:    &runtimeapi.ContainerStatus{},
			expected: fmt.Errorf("status.Id is not set"),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id: "1",
			},
			expected: fmt.Errorf("status.Metadata is not set"),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id:       "2",
				Metadata: metaWithoutName,
			},
			expected: fmt.Errorf("metadata.Name is not in metadata %q", metaWithoutName),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id:       "3",
				Metadata: meta,
			},
			expected: fmt.Errorf("status.CreatedAt is not set"),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id:        "4",
				Metadata:  meta,
				CreatedAt: 1,
				Image:     imageSpecWithoutImage,
			},
			expected: fmt.Errorf("status.Image is not set"),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id:        "5",
				Metadata:  meta,
				Image:     imageSpec,
				CreatedAt: 1,
			},
			expected: fmt.Errorf("status.ImageRef is not set"),
		},
		{
			input: &runtimeapi.ContainerStatus{
				Id:        "5",
				Metadata:  meta,
				Image:     imageSpec,
				CreatedAt: 1,
				ImageRef:  "Ref-1",
			},
			expected: nil,
		},
	}
	for _, status := range statuses {
		actual := verifyContainerStatus(status.input)
		if actual != nil {
			assert.EqualError(t, actual, status.expected.Error())
		} else {
			assert.Nil(t, status.expected)
		}
	}
}
