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

package dockershim

import (
	"testing"

	dockertypes "github.com/docker/engine-api/types"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

func TestCreateSandbox(t *testing.T) {
	ds, fakeDocker := newTestDockerSevice()
	name := "FOO"
	namespace := "BAR"
	uid := "1"
	config := &runtimeApi.PodSandboxConfig{
		Metadata: &runtimeApi.PodSandboxMetadata{
			Name:      &name,
			Namespace: &namespace,
			Uid:       &uid,
		},
	}
	id, err := ds.CreatePodSandbox(config)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if err := fakeDocker.AssertStarted([]string{id}); err != nil {
		t.Errorf("%v", err)
	}

	// List running containers and verify that there is one (and only one)
	// running container that we just created.
	containers, err := fakeDocker.ListContainers(dockertypes.ContainerListOptions{All: false})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(containers) != 1 {
		t.Errorf("More than one running containers: %+v", containers)
	}
	if containers[0].ID != id {
		t.Errorf("Expected id %q, got %v", id, containers[0].ID)
	}
}
