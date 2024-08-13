/*
Copyright 2022 The Kubernetes Authors.

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

package topologycache

// TopologyAwareHints events messages list.
const (
	NoZoneSpecified                       = "One or more endpoints do not have a zone specified"
	NoAllocatedHintsForZones              = "No hints allocated for zones"
	TopologyAwareHintsEnabled             = "Topology Aware Hints has been enabled"
	TopologyAwareHintsDisabled            = "Topology Aware Hints has been disabled"
	InsufficientNodeInfo                  = "Insufficient Node information: allocatable CPU or zone not specified on one or more nodes"
	NodesReadyInOneZoneOnly               = "Nodes only ready in one zone"
	InsufficientNumberOfEndpoints         = "Insufficient number of endpoints"
	MinAllocationExceedsOverloadThreshold = "Unable to allocate minimum required endpoints to each zone without exceeding overload threshold"
)

// EventBuilder let's us construct events in the code.
// We use it to build events and return them from a function instead of publishing them from within it.
// EventType, Reason, and Message fields are equivalent to the v1.Event fields - https://pkg.go.dev/k8s.io/api/core/v1#Event.
type EventBuilder struct {
	EventType string
	Reason    string
	Message   string
}
