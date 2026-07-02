/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"testing"

	"google.golang.org/protobuf/proto"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

// TestCreateContainerSandboxConfigReflectsKubeletView documents that
// CreateContainerRequest.sandbox_config may differ from the config used at
// RunPodSandbox (see CRI api.proto). The fake records the latest sandbox_config
// for test assertions.
func TestCreateContainerSandboxConfigReflectsKubeletView(t *testing.T) {
	ctx := context.Background()
	r := NewFakeRuntimeService()

	md := &runtimeapi.PodSandboxMetadata{Name: "pod", Uid: "uid", Namespace: "ns", Attempt: 0}
	runCfg := &runtimeapi.PodSandboxConfig{
		Metadata: md,
		Annotations: map[string]string{
			"key": "value-at-run-pod-sandbox",
		},
		Linux: &runtimeapi.LinuxPodSandboxConfig{},
	}

	sandboxID, err := r.RunPodSandbox(ctx, runCfg, "")
	if err != nil {
		t.Fatalf("RunPodSandbox: %v", err)
	}

	createCfg := proto.Clone(runCfg).(*runtimeapi.PodSandboxConfig)
	createCfg.Annotations = map[string]string{
		"key": "value-at-create-container",
	}

	_, err = r.CreateContainer(ctx, sandboxID, &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{Name: "ctr", Attempt: 0},
		Image:    &runtimeapi.ImageSpec{Image: "busybox"},
		Labels:   map[string]string{},
	}, createCfg)
	if err != nil {
		t.Fatalf("CreateContainer: %v", err)
	}

	r.Lock()
	last := r.LastCreateContainerSandboxConfig
	r.Unlock()
	if last == nil {
		t.Fatal("expected LastCreateContainerSandboxConfig to be set")
	}
	if got := last.Annotations["key"]; got != "value-at-create-container" {
		t.Fatalf("expected kubelet view of annotations at CreateContainer, got %q", got)
	}
}

func TestCreateContainerNilSandboxConfigClearsLast(t *testing.T) {
	ctx := context.Background()
	r := NewFakeRuntimeService()
	md := &runtimeapi.PodSandboxMetadata{Name: "pod", Uid: "uid2", Namespace: "ns", Attempt: 0}
	runCfg := &runtimeapi.PodSandboxConfig{
		Metadata: md,
		Linux:    &runtimeapi.LinuxPodSandboxConfig{},
	}
	sandboxID, err := r.RunPodSandbox(ctx, runCfg, "")
	if err != nil {
		t.Fatalf("RunPodSandbox: %v", err)
	}

	if _, err := r.CreateContainer(ctx, sandboxID, &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{Name: "c1", Attempt: 0},
		Image:    &runtimeapi.ImageSpec{Image: "img"},
		Labels:   map[string]string{},
	}, runCfg); err != nil {
		t.Fatalf("CreateContainer: %v", err)
	}
	if _, err := r.CreateContainer(ctx, sandboxID, &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{Name: "c2", Attempt: 0},
		Image:    &runtimeapi.ImageSpec{Image: "img"},
		Labels:   map[string]string{},
	}, nil); err != nil {
		t.Fatalf("CreateContainer: %v", err)
	}

	r.Lock()
	last := r.LastCreateContainerSandboxConfig
	r.Unlock()
	if last != nil {
		t.Fatalf("expected nil last config after CreateContainer with nil sandbox_config, got %+v", last)
	}
}
