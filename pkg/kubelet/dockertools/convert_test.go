/*
Copyright 2015 The Kubernetes Authors.

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

package dockertools

import (
	"reflect"
	"testing"

	dockertypes "github.com/docker/engine-api/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestMapState(t *testing.T) {
	testCases := []struct {
		input    string
		expected kubecontainer.ContainerState
	}{
		{input: "Up 5 hours", expected: kubecontainer.ContainerStateRunning},
		{input: "Exited (0) 2 hours ago", expected: kubecontainer.ContainerStateExited},
		{input: "Created", expected: kubecontainer.ContainerStateUnknown},
		{input: "Random string", expected: kubecontainer.ContainerStateUnknown},
	}

	for i, test := range testCases {
		if actual := mapState(test.input); actual != test.expected {
			t.Errorf("Test[%d]: expected %q, got %q", i, test.expected, actual)
		}
	}
}

func TestToRuntimeContainer(t *testing.T) {
	original := &dockertypes.Container{
		ID:     "ab2cdf",
		Image:  "bar_image",
		Names:  []string{"/k8s_bar.5678_foo_ns_1234_42"},
		Status: "Up 5 hours",
	}
	expected := &kubecontainer.Container{
		ID:    kubecontainer.ContainerID{Type: "docker", ID: "ab2cdf"},
		Name:  "bar",
		Image: "bar_image",
		Hash:  0x5678,
		State: kubecontainer.ContainerStateRunning,
	}

	actual, err := toRuntimeContainer(original)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestToRuntimeImage(t *testing.T) {
	original := &dockertypes.Image{
		ID:          "aeeea",
		RepoTags:    []string{"abc", "def"},
		RepoDigests: []string{"123", "456"},
		VirtualSize: 1234,
	}
	expected := &kubecontainer.Image{
		ID:          "aeeea",
		RepoTags:    []string{"abc", "def"},
		RepoDigests: []string{"123", "456"},
		Size:        1234,
	}

	actual, err := toRuntimeImage(original)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}
