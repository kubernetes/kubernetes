/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	docker "github.com/fsouza/go-dockerclient"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestToRuntimeContainer(t *testing.T) {
	original := &docker.APIContainers{
		ID:      "ab2cdf",
		Image:   "bar_image",
		Created: 12345,
		Names:   []string{"/k8s_bar.5678_foo_ns_1234_42"},
	}
	expected := &kubecontainer.Container{
		ID:      kubecontainer.ContainerID{"docker", "ab2cdf"},
		Name:    "bar",
		Image:   "bar_image",
		Hash:    0x5678,
		Created: 12345,
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
	original := &docker.APIImages{
		ID:          "aeeea",
		RepoTags:    []string{"abc", "def"},
		VirtualSize: 1234,
	}
	expected := &kubecontainer.Image{
		ID:   "aeeea",
		Tags: []string{"abc", "def"},
		Size: 1234,
	}

	actual, err := toRuntimeImage(original)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}
