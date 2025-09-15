/*
Copyright 2014 The Kubernetes Authors.

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

package capabilities

import (
	"sync"
)

// Capabilities defines the set of capabilities available within the system.
// For now these are global.  Eventually they may be per-user
type Capabilities struct {
	AllowPrivileged bool

	// Pod sources from which to allow privileged capabilities like host networking, sharing the host
	// IPC namespace, and sharing the host PID namespace.
	PrivilegedSources PrivilegedSources

	// PerConnectionBandwidthLimitBytesPerSec limits the throughput of each connection (currently only used for proxy, exec, attach)
	PerConnectionBandwidthLimitBytesPerSec int64
}

// PrivilegedSources defines the pod sources allowed to make privileged requests for certain types
// of capabilities like host networking, sharing the host IPC namespace, and sharing the host PID namespace.
type PrivilegedSources struct {
	// List of pod sources for which using host network is allowed.
	HostNetworkSources []string

	// List of pod sources for which using host pid namespace is allowed.
	HostPIDSources []string

	// List of pod sources for which using host ipc is allowed.
	HostIPCSources []string
}

var capInstance struct {
	once         sync.Once
	lock         sync.Mutex
	capabilities *Capabilities
}

// Initialize the capability set.  This can only be done once per binary, subsequent calls are ignored.
func Initialize(c Capabilities) {
	// Only do this once
	capInstance.once.Do(func() {
		capInstance.capabilities = &c
	})
}

// Setup the capability set.  It wraps Initialize for improving usability.
func Setup(allowPrivileged bool, perConnectionBytesPerSec int64) {
	Initialize(Capabilities{
		AllowPrivileged:                        allowPrivileged,
		PerConnectionBandwidthLimitBytesPerSec: perConnectionBytesPerSec,
	})
}

// ResetForTest resets the capabilities to a given state for testing purposes.
// This function should only be called from tests.
func ResetForTest() {
	capInstance.lock.Lock()
	defer capInstance.lock.Unlock()
	capInstance.capabilities = nil
	capInstance.once = sync.Once{}
}

// Get returns a read-only copy of the system capabilities.
func Get() Capabilities {
	capInstance.lock.Lock()
	defer capInstance.lock.Unlock()
	// This check prevents clobbering of capabilities that might've been set via SetForTests
	if capInstance.capabilities == nil {
		Initialize(Capabilities{
			AllowPrivileged: false,
			PrivilegedSources: PrivilegedSources{
				HostNetworkSources: []string{},
				HostPIDSources:     []string{},
				HostIPCSources:     []string{},
			},
		})
	}
	return *capInstance.capabilities
}
