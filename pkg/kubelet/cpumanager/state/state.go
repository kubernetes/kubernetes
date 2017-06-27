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

package state

import (
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

type Reader interface {
	GetCPUSet(containerID string) (cpuset.CPUSet, bool)
	GetDefaultCPUSet() cpuset.CPUSet
	GetCPUSetOrDefault(containerID string) cpuset.CPUSet
}

type Writer interface {
	SetCPUSet(containerID string, cpuset cpuset.CPUSet)
	SetDefaultCPUSet(cpuset cpuset.CPUSet)
	Delete(containerID string)
}

type State interface {
	Reader
	Writer
}
