// +build linux

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

package dockershim

import (
	"context"
	"fmt"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

// ContainerStats returns stats for a container stats request based on container id.
func (ds *dockerService) ContainerStats(_ context.Context, r *runtimeapi.ContainerStatsRequest) (*runtimeapi.ContainerStatsResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// ListContainerStats returns stats for a list container stats request based on a filter.
func (ds *dockerService) ListContainerStats(_ context.Context, r *runtimeapi.ListContainerStatsRequest) (*runtimeapi.ListContainerStatsResponse, error) {
	return nil, fmt.Errorf("not implemented")
}
