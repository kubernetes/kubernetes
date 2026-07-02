/*
Copyright The Kubernetes Authors.

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

package testing

import (
	"testing"

	"google.golang.org/protobuf/proto"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func TestRestorePodKeepsPodsSeparate(t *testing.T) {
	firstMetadata := &runtimeapi.PodSandboxMetadata{Name: "worker", Namespace: "team-a", Uid: "uid-a", Attempt: 1}
	for _, tc := range []struct {
		name     string
		metadata *runtimeapi.PodSandboxMetadata
	}{
		{name: "different pod name", metadata: &runtimeapi.PodSandboxMetadata{Name: "other", Namespace: "team-a", Uid: "uid-a", Attempt: 1}},
		{name: "different namespace", metadata: &runtimeapi.PodSandboxMetadata{Name: "worker", Namespace: "team-b", Uid: "uid-a", Attempt: 1}},
		{name: "different pod UID", metadata: &runtimeapi.PodSandboxMetadata{Name: "worker", Namespace: "team-a", Uid: "uid-b", Attempt: 1}},
		{name: "different sandbox attempt", metadata: &runtimeapi.PodSandboxMetadata{Name: "worker", Namespace: "team-a", Uid: "uid-a", Attempt: 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			runtime := NewFakeRuntimeService()
			requests := make([]*runtimeapi.RestorePodRequest, 2)
			responses := make([]*runtimeapi.RestorePodResponse, 2)
			for i, metadata := range []*runtimeapi.PodSandboxMetadata{firstMetadata, tc.metadata} {
				identity := []string{"first", "second"}[i]
				request := &runtimeapi.RestorePodRequest{
					Config: &runtimeapi.PodSandboxConfig{
						Metadata:    metadata,
						Labels:      map[string]string{"pod": identity},
						Annotations: map[string]string{"sandbox": identity},
					},
					RuntimeHandler: "test-handler",
				}
				for j, name := range []string{"app", "sidecar"} {
					request.ContainerConfigs = append(request.ContainerConfigs, &runtimeapi.ContainerConfig{
						Metadata:    &runtimeapi.ContainerMetadata{Name: name, Attempt: uint32(j + 2)},
						Image:       &runtimeapi.ImageSpec{Image: identity + "-" + name + ":latest"},
						Labels:      map[string]string{"pod": identity},
						Annotations: map[string]string{"container": identity},
						Linux:       &runtimeapi.LinuxContainerConfig{Resources: &runtimeapi.LinuxContainerResources{CpuQuota: int64(i + 100)}},
					})
				}
				response, err := runtime.RestorePod(ctx, request)
				if err != nil {
					t.Fatalf("restore pod %d: %v", i, err)
				}
				if response.GetPodSandboxId() == "" || len(response.GetRestoredContainers()) != len(request.ContainerConfigs) {
					t.Fatalf("restore pod %d returned incomplete response: %v", i, response)
				}
				requests[i], responses[i] = request, response
			}

			if responses[0].PodSandboxId == responses[1].PodSandboxId {
				t.Errorf("restores share sandbox ID %q", responses[0].PodSandboxId)
			}
			seenContainers := make(map[string]bool)
			for i, response := range responses {
				sandboxStatus, err := runtime.PodSandboxStatus(ctx, response.PodSandboxId, false)
				if err != nil {
					t.Fatalf("sandbox status for pod %d: %v", i, err)
				}
				if !proto.Equal(sandboxStatus.Status.Metadata, requests[i].Config.Metadata) || sandboxStatus.Status.State != runtimeapi.PodSandboxState_SANDBOX_READY {
					t.Errorf("sandbox status for pod %d: %v", i, sandboxStatus)
				}
				for j, restored := range response.RestoredContainers {
					config := requests[i].ContainerConfigs[j]
					if restored.ContainerId == "" || seenContainers[restored.ContainerId] {
						t.Errorf("pod %d has empty or reused container ID %q", i, restored.ContainerId)
					}
					seenContainers[restored.ContainerId] = true
					if restored.Name != config.Metadata.Name {
						t.Errorf("restored container name = %q, want %q", restored.Name, config.Metadata.Name)
					}
					status, err := runtime.ContainerStatus(ctx, restored.ContainerId, false)
					if err != nil {
						t.Fatalf("container status for pod %d container %q: %v", i, restored.Name, err)
					}
					if !proto.Equal(status.Status.Metadata, config.Metadata) || !proto.Equal(status.Status.Image, config.Image) || status.Status.State != runtimeapi.ContainerState_CONTAINER_CREATED {
						t.Errorf("container status for pod %d container %q: %v", i, restored.Name, status)
					}
					container := runtime.Containers[restored.ContainerId]
					if container.SandboxID != response.PodSandboxId || !proto.Equal(container.LinuxResources, config.Linux.Resources) {
						t.Errorf("container %q belongs to sandbox %q with resources %v", restored.ContainerId, container.SandboxID, container.LinuxResources)
					}
					id, err := runtime.GetContainerID(response.PodSandboxId, restored.Name, config.Metadata.Attempt)
					if err != nil || id != restored.ContainerId {
						t.Errorf("container lookup for pod %d container %q = %q, %v", i, restored.Name, id, err)
					}
					if err := runtime.StartContainer(ctx, restored.ContainerId); err != nil {
						t.Fatalf("start restored container %q: %v", restored.ContainerId, err)
					}
				}
			}
			if len(runtime.Sandboxes) != 2 || len(runtime.Containers) != 4 {
				t.Errorf("runtime has %d sandboxes and %d containers, want 2 and 4", len(runtime.Sandboxes), len(runtime.Containers))
			}

			for _, restored := range responses[0].RestoredContainers {
				if err := runtime.StopContainer(ctx, restored.ContainerId, 0); err != nil {
					t.Fatalf("stop first pod container %q: %v", restored.ContainerId, err)
				}
				if err := runtime.RemoveContainer(ctx, restored.ContainerId); err != nil {
					t.Fatalf("remove first pod container %q: %v", restored.ContainerId, err)
				}
			}
			if err := runtime.StopPodSandbox(ctx, responses[0].PodSandboxId); err != nil {
				t.Fatalf("stop first pod sandbox: %v", err)
			}
			status, err := runtime.PodSandboxStatus(ctx, responses[1].PodSandboxId, false)
			if err != nil || status.GetStatus().GetState() != runtimeapi.PodSandboxState_SANDBOX_READY {
				t.Errorf("stopping first sandbox changed second sandbox: %v, %v", status, err)
			}
			if err := runtime.RemovePodSandbox(ctx, responses[0].PodSandboxId); err != nil {
				t.Fatalf("remove first pod sandbox: %v", err)
			}
			if _, err := runtime.PodSandboxStatus(ctx, responses[0].PodSandboxId, false); err == nil {
				t.Error("first pod sandbox still exists after removal")
			}
			status, err = runtime.PodSandboxStatus(ctx, responses[1].PodSandboxId, false)
			if err != nil || !proto.Equal(status.GetStatus().GetMetadata(), requests[1].Config.Metadata) {
				t.Errorf("removing first sandbox changed second sandbox: %v, %v", status, err)
			}
			for j, restored := range responses[1].RestoredContainers {
				status, err := runtime.ContainerStatus(ctx, restored.ContainerId, false)
				if err != nil || status.GetStatus().GetState() != runtimeapi.ContainerState_CONTAINER_RUNNING {
					t.Errorf("cleanup of first pod changed second pod container %q: %v, %v", restored.Name, status, err)
				}
				id, err := runtime.GetContainerID(responses[1].PodSandboxId, restored.Name, requests[1].ContainerConfigs[j].Metadata.Attempt)
				if err != nil || id != restored.ContainerId {
					t.Errorf("remaining container lookup for %q = %q, %v", restored.Name, id, err)
				}
			}
			if len(runtime.Sandboxes) != 1 || len(runtime.Containers) != 2 {
				t.Errorf("after cleanup runtime has %d sandboxes and %d containers, want 1 and 2", len(runtime.Sandboxes), len(runtime.Containers))
			}
		})
	}
}

func TestRestorePodPreservesSandboxConfig(t *testing.T) {
	ctx := t.Context()
	runtime := NewFakeRuntimeService()
	config := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{Name: "worker", Namespace: "team-a", Uid: "uid-a", Attempt: 1},
		Linux: &runtimeapi.LinuxPodSandboxConfig{SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{
			NamespaceOptions: &runtimeapi.NamespaceOption{
				Network: runtimeapi.NamespaceMode_NODE,
				Pid:     runtimeapi.NamespaceMode_CONTAINER,
				Ipc:     runtimeapi.NamespaceMode_NODE,
			},
		}},
	}
	const handler = "restore-handler"
	response, err := runtime.RestorePod(ctx, &runtimeapi.RestorePodRequest{Config: config, RuntimeHandler: handler})
	if err != nil {
		t.Fatalf("restore pod: %v", err)
	}
	status, err := runtime.PodSandboxStatus(ctx, response.PodSandboxId, false)
	if err != nil {
		t.Fatalf("sandbox status: %v", err)
	}
	if !proto.Equal(status.GetStatus().GetLinux().GetNamespaces().GetOptions(), config.Linux.SecurityContext.NamespaceOptions) {
		t.Errorf("restored sandbox namespaces = %v, want %v", status.GetStatus().GetLinux().GetNamespaces().GetOptions(), config.Linux.SecurityContext.NamespaceOptions)
	}
	if status.GetStatus().GetRuntimeHandler() != handler {
		t.Errorf("restored sandbox runtime handler = %q, want %q", status.GetStatus().GetRuntimeHandler(), handler)
	}
}
