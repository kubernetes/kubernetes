//go:build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package cadvisor

import (
	"context"
	"testing"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

// fakeEnumerator is a ContainerEnumerator test double returning a fixed set
// of CRI containers.
type fakeEnumerator struct {
	containers []*runtimeapi.Container
}

func (f *fakeEnumerator) ListContainers(context.Context, *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	return f.containers, nil
}

// TestGetRequestedContainersInfoWindows verifies that GetRequestedContainersInfo
// surfaces one minimal per-container ContainerInfo keyed by the CRI container ID,
// with the labels/alias the metrics path needs so the Windows OOM watcher's
// recordOOMKill(containerID) is read back by OOMEventsForContainer(info.Name).
func TestGetRequestedContainersInfoWindows(t *testing.T) {
	client := &cadvisorClient{
		winStatsClient: nil,
		containerEnumerator: &fakeEnumerator{
			containers: []*runtimeapi.Container{
				{
					Id:        "cid-1",
					CreatedAt: 1234,
					Metadata:  &runtimeapi.ContainerMetadata{Name: "app"},
					Labels:    map[string]string{"io.kubernetes.container.name": "app"},
				},
				{
					Id:       "cid-2",
					Metadata: &runtimeapi.ContainerMetadata{Name: "sidecar"},
					Labels:   map[string]string{"io.kubernetes.container.name": "sidecar"},
				},
			},
		},
	}

	infos, err := client.GetRequestedContainersInfo("/", cadvisorapi.RequestOptions{})
	if err != nil {
		t.Fatalf("GetRequestedContainersInfo() error = %v", err)
	}
	if len(infos) != 2 {
		t.Fatalf("GetRequestedContainersInfo() returned %d infos, want 2", len(infos))
	}

	for id, want := range map[string]string{"cid-1": "app", "cid-2": "sidecar"} {
		info, ok := infos[id]
		if !ok {
			t.Errorf("no info for container id %q", id)
			continue
		}
		if info.Name != id {
			t.Errorf("info.Name = %q for %q, want the CRI container ID", info.Name, id)
		}
		if len(info.Stats) != 1 || info.Stats[0] == nil {
			t.Errorf("info.Stats = %v, want a single sample", info.Stats)
		}
		if info.Spec.Labels["io.kubernetes.container.name"] != want {
			t.Errorf("Spec.Labels did not carry the container name, got %v", info.Spec.Labels)
		}
	}

	// A client without an enumerator must keep the historic no-op behavior.
	infoZe, err := (&cadvisorClient{}).GetRequestedContainersInfo("/", cadvisorapi.RequestOptions{})
	if err != nil {
		t.Fatalf("GetRequestedContainersInfo() no-enumerator error = %v", err)
	}
	if infoZe != nil {
		t.Errorf("GetRequestedContainersInfo() = %v without enumerator, want nil", infoZe)
	}
}
