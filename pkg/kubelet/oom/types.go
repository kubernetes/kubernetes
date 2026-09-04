/*
Copyright 2019 The Kubernetes Authors.

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

package oom

import (
	"context"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

// containerStatusGetter is the subset of the CRI runtime service the Windows
// OOM watcher needs to observe container exit reasons. The kubelet's
// internalapi.RuntimeService satisfies it. It lives here (outside of any
// *_windows.go file) only so the unsupported platform stub can share it.
type containerStatusGetter interface {
	ListContainers(ctx context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error)
	ContainerStatus(ctx context.Context, containerID string, verbose bool) (*runtimeapi.ContainerStatusResponse, error)
}

// Watcher defines the interface of OOM watchers.
type Watcher interface {
	Start(ctx context.Context, ref *v1.ObjectReference) error
}
