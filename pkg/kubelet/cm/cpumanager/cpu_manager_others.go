//go:build !windows

/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"context"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/utils/cpuset"
)

func (m *manager) updateContainerCPUSet(ctx context.Context, containerID string, cpus cpuset.CPUSet) error {
	// TODO: Consider adding a `ResourceConfigForContainer` helper in
	// helpers_linux.go similar to what exists for pods.
	// It would be better to pass the full container resources here instead of
	// this patch-like partial resources.

	return m.containerRuntime.UpdateContainerResources(
		ctx,
		containerID,
		&runtimeapi.ContainerResources{
			Linux: &runtimeapi.LinuxContainerResources{
				CpusetCpus: cpus.String(),
			},
		})
}
