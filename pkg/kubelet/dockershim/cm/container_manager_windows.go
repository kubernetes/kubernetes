// +build windows

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

package cm

import (
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// no-op
type containerManager struct {
}

// NewContainerManager creates a new instance of ContainerManager
func NewContainerManager(_ string, _ libdocker.Interface) ContainerManager {
	return &containerManager{}
}

func (m *containerManager) Start() error {
	return nil
}
