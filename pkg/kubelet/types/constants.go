/*
Copyright 2015 The Kubernetes Authors.

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

package types

const (
	// ResolvConfDefault is the system default DNS resolver configuration.
	ResolvConfDefault = "/etc/resolv.conf"
	// RFC3339NanoFixed is the fixed width version of time.RFC3339Nano.
	RFC3339NanoFixed = "2006-01-02T15:04:05.000000000Z07:00"
	// RFC3339NanoLenient is the variable width RFC3339 time format for lenient parsing of strings into timestamps.
	RFC3339NanoLenient = "2006-01-02T15:04:05.999999999Z07:00"
)

// Different container runtimes.
const (
	DockerContainerRuntime = "docker"
	RemoteContainerRuntime = "remote"
)

// User visible keys for managing node allocatable enforcement on the node.
const (
	NodeAllocatableEnforcementKey = "pods"
	SystemReservedEnforcementKey  = "system-reserved"
	KubeReservedEnforcementKey    = "kube-reserved"
	NodeAllocatableNoneKey        = "none"
)

// SwapBehavior types
const (
	LimitedSwap   = "LimitedSwap"
	UnlimitedSwap = "UnlimitedSwap"
)
