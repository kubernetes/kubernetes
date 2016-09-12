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
	"fmt"
	"testing"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/stretchr/testify/assert"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// A helper to create a basic config.
func makeSandboxConfig(name, namespace, uid string, attempt uint32) *runtimeApi.PodSandboxConfig {
	return &runtimeApi.PodSandboxConfig{
		Metadata: &runtimeApi.PodSandboxMetadata{
			Name:      &name,
			Namespace: &namespace,
			Uid:       &uid,
			Attempt:   &attempt,
		},
	}
}

// TestRunSandbox tests that RunSandbox creates and starts a container
// acting a the sandbox for the pod.
func TestRunSandbox(t *testing.T) {
	ds, fakeDocker := newTestDockerSevice()
	config := makeSandboxConfig("foo", "bar", "1", 0)
	id, err := ds.RunPodSandbox(config)
	assert.NoError(t, err)
	assert.NoError(t, fakeDocker.AssertStarted([]string{id}))

	// List running containers and verify that there is one (and only one)
	// running container that we just created.
	containers, err := fakeDocker.ListContainers(dockertypes.ContainerListOptions{All: false})
	assert.NoError(t, err)
	assert.Len(t, containers, 1)
	assert.Equal(t, id, containers[0].ID)
}

// TestListSandboxes creates several sandboxes and then list them to check
// whether the correct metadatas, states, and labels are returned.
func TestListSandboxes(t *testing.T) {
	ds, _ := newTestDockerSevice()
	name, namespace := "foo", "bar"
	configs := []*runtimeApi.PodSandboxConfig{}
	for i := 0; i < 3; i++ {
		c := makeSandboxConfig(fmt.Sprintf("%s%d", name, i),
			fmt.Sprintf("%s%d", namespace, i), fmt.Sprintf("%d", i), 0)
		configs = append(configs, c)
	}

	expected := []*runtimeApi.PodSandbox{}
	state := runtimeApi.PodSandBoxState_READY
	var createdAt int64 = 0
	for i := range configs {
		id, err := ds.RunPodSandbox(configs[i])
		assert.NoError(t, err)
		// Prepend to the expected list because ListPodSandbox returns
		// the most recent sandbox first.
		expected = append([]*runtimeApi.PodSandbox{{
			Metadata:  configs[i].Metadata,
			Id:        &id,
			State:     &state,
			Labels:    map[string]string{containerTypeLabelKey: containerTypeLabelSandbox},
			CreatedAt: &createdAt,
		}}, expected...)
	}
	sandboxes, err := ds.ListPodSandbox(nil)
	assert.NoError(t, err)
	assert.Len(t, sandboxes, len(expected))
	assert.Equal(t, expected, sandboxes)
}
