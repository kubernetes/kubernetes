// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package api

import (
	"time"

	cadvisor "github.com/google/cadvisor/info/v1"
	kubeapi "k8s.io/kubernetes/pkg/api"
)

type PodMetadata struct {
	Name         string `json:"name,omitempty"`
	Namespace    string `json:"namespace,omitempty"`
	NamespaceUID string
	// TODO(vishh): Rename to UID.
	ID             string
	Hostname       string
	Status         string
	PodIP          string
	Labels         map[string]string
	HostPublicIP   string
	HostInternalIP string
	ExternalID     string
}

// PodState is the state of a pod, used as either input (desired state) or output (current state)
type Pod struct {
	PodMetadata `json:",inline"`
	Containers  []Container `json:"containers"`
}

type AggregateData struct {
	Pods       []Pod
	Containers []Container
	// TODO: Add node metadata.
	Machine []Container
	Events  []kubeapi.Event
}

func (a *AggregateData) Merge(b *AggregateData) {
	a.Pods = append(a.Pods, b.Pods...)
	a.Containers = append(a.Containers, b.Containers...)
	a.Machine = append(a.Machine, b.Machine...)
	a.Events = append(a.Events, b.Events...)
}

type ContainerSpec struct {
	// TODO(piosz): Consider defining an internal Spec API to guard against changes to cadvisor API.
	cadvisor.ContainerSpec
	CpuRequest    int64
	MemoryRequest int64
}

type ContainerStats struct {
	// TODO(piosz): Consider defining an internal Stats API (see above).
	cadvisor.ContainerStats
}

type Container struct {
	Hostname   string
	ExternalID string
	Name       string
	Image      string
	Spec       ContainerSpec
	Stats      []*ContainerStats
}

func NewContainer() *Container {
	return &Container{Stats: make([]*ContainerStats, 0)}
}

// An external node represents a host which is running cadvisor. Heapster will expect
// data in this format via its external files API.
type ExternalNode struct {
	Name string `json:"name,omitempty"`
	IP   string `json:"ip,omitempty"`
}

// ExternalNodeList represents a list of all hosts that are running cadvisor which heapster will monitor.
type ExternalNodeList struct {
	Items []ExternalNode `json:"items,omitempty"`
}

// Source represents a plugin that generates data.
// Each Source needs to implement a Register
type Source interface {
	// GetInfo Fetches information about pods or containers.
	// start, end: Represents the time range for stats
	// Returns:
	// AggregateData
	GetInfo(start, end time.Time) (AggregateData, error)
	// Returns debug information for the source.
	DebugInfo() string
	// Returns a user friendly string that describes the source.
	Name() string
}
