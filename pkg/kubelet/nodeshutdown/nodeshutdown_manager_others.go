//go:build !linux
// +build !linux

/*
Copyright 2020 The Kubernetes Authors.

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

package nodeshutdown

import (
	"time"

	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

// Manager is a fake node shutdown manager for non linux platforms.
type Manager struct{}

// NewManager returns a fake node shutdown manager for non linux platforms.
func NewManager(getPodsFunc eviction.ActivePodsFunc, killPodFunc eviction.KillPodFunc, syncNodeStatus func(), shutdownGracePeriodRequested, shutdownGracePeriodCriticalPods time.Duration) (*Manager, lifecycle.PodAdmitHandler) {
	m := &Manager{}
	return m, m
}

// Admit returns a fake Pod admission which always returns true
func (m *Manager) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{Admit: true}
}

// Start is a no-op always returning nil for non linux platforms.
func (m *Manager) Start() error {
	return nil
}

// ShutdownStatus is a no-op always returning nil for non linux platforms.
func (m *Manager) ShutdownStatus() error {
	return nil
}
