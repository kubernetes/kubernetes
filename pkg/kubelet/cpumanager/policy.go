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
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
)

// Policy interface used by cpu manager, shall implement logic for cpu to pod
// assignment
type Policy interface {
	IsUnderPressure() bool
	Name() string
	Start(s state.State)
	RegisterContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error
	UnregisterContainer(s state.State, containerID string) error
}
