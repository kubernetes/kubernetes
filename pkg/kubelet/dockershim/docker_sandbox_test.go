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
	"os"
	"testing"
	"time"

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

// TestSandboxStatus tests the basic lifecycle operations and verify that
// the status returned reflects the operations performed.
func TestSandboxStatus(t *testing.T) {
	ds, fakeDocker := newTestDockerSevice()
	fClock := fakeDocker.Clock
	config := makeSandboxConfig("foo", "bar", "1", 0)

	// TODO: The following variables depend on the internal
	// implementation of FakeDockerClient, and should be fixed.
	fakeIP := "2.3.4.5"
	fakeNS := fmt.Sprintf("/proc/%d/ns/net", os.Getpid())

	state := runtimeApi.PodSandBoxState_READY
	ct := int64(0)
	expected := &runtimeApi.PodSandboxStatus{
		State:     &state,
		CreatedAt: &ct,
		Metadata:  config.Metadata,
		Labels:    map[string]string{containerTypeLabelKey: containerTypeLabelSandbox},
		Network:   &runtimeApi.PodSandboxNetworkStatus{Ip: &fakeIP},
		Linux:     &runtimeApi.LinuxPodSandboxStatus{Namespaces: &runtimeApi.Namespace{Network: &fakeNS}},
	}

	// Create the sandbox.
	fClock.SetTime(time.Now())
	*expected.CreatedAt = fClock.Now().Unix()
	id, err := ds.RunPodSandbox(config)
	expected.Id = &id // ID is only known after the creation.
	status, err := ds.PodSandboxStatus(id)
	assert.NoError(t, err)
	assert.Equal(t, expected, status)

	// Stop the sandbox.
	*expected.State = runtimeApi.PodSandBoxState_NOTREADY
	err = ds.StopPodSandbox(id)
	assert.NoError(t, err)
	status, err = ds.PodSandboxStatus(id)
	assert.Equal(t, expected, status)

	// Remove the container.
	err = ds.RemovePodSandbox(id)
	assert.NoError(t, err)
	status, err = ds.PodSandboxStatus(id)
	assert.Error(t, err, fmt.Sprintf("status of sandbox: %+v", status))
}
