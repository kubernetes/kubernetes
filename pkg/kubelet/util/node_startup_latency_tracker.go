/*
Copyright 2023 The Kubernetes Authors.

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

package util

import (
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/clock"
)

type NodeStartupLatencyTracker interface {
	// This function may be called across Kubelet restart.
	RecordAttemptRegisterNode()
	// This function should not be called across Kubelet restart.
	RecordRegisteredNewNode()
	// This function may be called across Kubelet restart.
	RecordNodeReady()
}

type basicNodeStartupLatencyTracker struct {
	lock sync.Mutex

	bootTime                     time.Time
	kubeletStartTime             time.Time
	firstRegistrationAttemptTime time.Time
	firstRegisteredNewNodeTime   time.Time
	firstNodeReadyTime           time.Time

	// For testability
	clock clock.Clock
}

func NewNodeStartupLatencyTracker() NodeStartupLatencyTracker {
	bootTime, err := GetBootTime()
	if err != nil {
		bootTime = time.Time{}
	}
	return &basicNodeStartupLatencyTracker{
		bootTime:         bootTime,
		kubeletStartTime: time.Now(),
		clock:            clock.RealClock{},
	}
}

func (n *basicNodeStartupLatencyTracker) RecordAttemptRegisterNode() {
	n.lock.Lock()
	defer n.lock.Unlock()

	if !n.firstRegistrationAttemptTime.IsZero() {
		return
	}

	n.firstRegistrationAttemptTime = n.clock.Now()
}

func (n *basicNodeStartupLatencyTracker) RecordRegisteredNewNode() {
	n.lock.Lock()
	defer n.lock.Unlock()

	if n.firstRegistrationAttemptTime.IsZero() || !n.firstRegisteredNewNodeTime.IsZero() {
		return
	}

	n.firstRegisteredNewNodeTime = n.clock.Now()

	if !n.bootTime.IsZero() {
		metrics.NodeStartupPreKubeletDuration.Set(n.kubeletStartTime.Sub(n.bootTime).Seconds())
	}
	metrics.NodeStartupPreRegistrationDuration.Set(n.firstRegistrationAttemptTime.Sub(n.kubeletStartTime).Seconds())
	metrics.NodeStartupRegistrationDuration.Set(n.firstRegisteredNewNodeTime.Sub(n.firstRegistrationAttemptTime).Seconds())
}

func (n *basicNodeStartupLatencyTracker) RecordNodeReady() {
	n.lock.Lock()
	defer n.lock.Unlock()

	if n.firstRegisteredNewNodeTime.IsZero() || !n.firstNodeReadyTime.IsZero() {
		return
	}

	n.firstNodeReadyTime = n.clock.Now()

	metrics.NodeStartupPostRegistrationDuration.Set(n.firstNodeReadyTime.Sub(n.firstRegisteredNewNodeTime).Seconds())
	if !n.bootTime.IsZero() {
		metrics.NodeStartupDuration.Set(n.firstNodeReadyTime.Sub(n.bootTime).Seconds())
	}
}
