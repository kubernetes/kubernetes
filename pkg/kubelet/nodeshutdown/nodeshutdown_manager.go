/*
Copyright 2021 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/utils/clock"
)

// Manager interface provides methods for Kubelet to manage node shutdown.
type Manager interface {
	Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult
	Start() error
	ShutdownStatus() error
}

// Config represents Manager configuration
type Config struct {
	Logger                           klog.Logger
	ProbeManager                     prober.Manager
	Recorder                         record.EventRecorder
	NodeRef                          *v1.ObjectReference
	GetPodsFunc                      eviction.ActivePodsFunc
	KillPodFunc                      eviction.KillPodFunc
	SyncNodeStatusFunc               func()
	ShutdownGracePeriodRequested     time.Duration
	ShutdownGracePeriodCriticalPods  time.Duration
	ShutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority
	StateDirectory                   string
	Clock                            clock.Clock
}

// managerStub is a fake node shutdown managerImpl .
type managerStub struct{}

// Admit returns a fake Pod admission which always returns true
func (managerStub) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	return lifecycle.PodAdmitResult{Admit: true}
}

// Start is a no-op always returning nil for non linux platforms.
func (managerStub) Start() error {
	return nil
}

// ShutdownStatus is a no-op always returning nil for non linux platforms.
func (managerStub) ShutdownStatus() error {
	return nil
}
