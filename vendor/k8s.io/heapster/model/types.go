// Copyright 2015 Google Inc. All Rights Reserved.
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

package model

import (
	"sync"
	"time"

	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

type Model interface {
	// The Update operation populates the Model from a cache.
	Update(cache.Cache) error
	GetCacheListener() cache.CacheListener

	// The simple Get operations extract structural information from the Model.
	GetAvailableMetrics() []string
	GetNodes() []EntityListEntry
	GetNamespaces() []EntityListEntry
	GetPods(string) []EntityListEntry
	GetPodContainers(string, string) []EntityListEntry
	GetNodePods(string) []EntityListEntry
	GetFreeContainers(string) []EntityListEntry

	// The GetXMetric operations extract timeseries from the Model.
	// The returned time.Time values signify the latest metric timestamp in the cluster.
	GetClusterMetric(ClusterMetricRequest) ([]statstore.TimePoint, time.Time, error)
	GetNodeMetric(NodeMetricRequest) ([]statstore.TimePoint, time.Time, error)
	GetNamespaceMetric(NamespaceMetricRequest) ([]statstore.TimePoint, time.Time, error)
	GetPodMetric(PodMetricRequest) ([]statstore.TimePoint, time.Time, error)
	GetBatchPodMetric(req BatchPodRequest) ([][]statstore.TimePoint, time.Time, error)
	GetPodContainerMetric(PodContainerMetricRequest) ([]statstore.TimePoint, time.Time, error)
	GetFreeContainerMetric(FreeContainerMetricRequest) ([]statstore.TimePoint, time.Time, error)

	// The GetXStats operations extract all derived stats for a single entity of the cluster.
	GetClusterStats() (*StatsResult, error)
	GetNodeStats(NodeRequest) (*StatsResult, error)
	GetNamespaceStats(NamespaceRequest) (*StatsResult, error)
	GetPodStats(PodRequest) (*StatsResult, error)
	GetPodContainerStats(PodContainerRequest) (*StatsResult, error)
	GetFreeContainerStats(FreeContainerRequest) (*StatsResult, error)
}

// realModel is an implementation of the Model interface.
// timestamp marks the latest timestamp of any metric present in the realModel.
// tsConstructor generates a new empty TimeStore, used for storing historical data.
type realModel struct {
	timestamp  time.Time
	lock       sync.RWMutex
	resolution time.Duration
	ClusterInfo
}

// Supported metric names, used as keys for all map[string]*daystore.DayStore
const cpuLimit = "cpu-limit"
const cpuUsage = "cpu-usage"
const memLimit = "memory-limit"
const memUsage = "memory-usage"
const memWorking = "memory-working"
const fsLimit = "fs-limit"
const fsUsage = "fs-usage"

// epsilon values for the underlying in-memory stores
// Epsilon values for CPU metrics are expressed in millicores
const cpuLimitEpsilon = 10 // 10 millicores
const cpuUsageEpsilon = 10 // 10 millicores

// Epsilon values for memory and filesystem metrics are expressed in bytes
const memLimitEpsilon = 4194304   // 4 MB
const memUsageEpsilon = 4194304   // 4 MB
const memWorkingEpsilon = 4194304 // 4 MB
const fsLimitEpsilon = 10485760   // 10 MB
const fsUsageEpsilon = 10485760   // 10 MB

// TODO(afein): move defaultEpsilon to impl_test after handling FS epsilon
const defaultEpsilon = 100 // used for testing

// Simple Request Types.
type MetricRequest struct {
	MetricName string
	Start      time.Time
	End        time.Time
}

type NodeRequest struct {
	NodeName string
}

type NamespaceRequest struct {
	NamespaceName string
}

type PodRequest struct {
	NamespaceName string
	PodName       string
}

type BatchPodRequest struct {
	NamespaceName string
	PodNames      []string
	MetricName    string
	Start         time.Time
	End           time.Time
}

type PodContainerRequest struct {
	NamespaceName string
	PodName       string
	ContainerName string
}

type FreeContainerRequest struct {
	NodeName      string
	ContainerName string
}

// Metric Request Types
type ClusterMetricRequest struct {
	MetricRequest
}

type NodeMetricRequest struct {
	NodeName string
	MetricRequest
}

type NamespaceMetricRequest struct {
	NamespaceName string
	MetricRequest
}

type PodMetricRequest struct {
	NamespaceName string
	PodName       string
	MetricRequest
}

type PodContainerMetricRequest struct {
	NamespaceName string
	PodName       string
	ContainerName string
	MetricRequest
}

type FreeContainerMetricRequest struct {
	NodeName      string
	ContainerName string
	MetricRequest
}

// Derived Stats Types

type Stats struct {
	Average     uint64
	NinetyFifth uint64
	Max         uint64
}

type StatBundle struct {
	Minute Stats
	Hour   Stats
	Day    Stats
}

type StatsResult struct {
	ByName    map[string]StatBundle
	Timestamp time.Time
	Uptime    time.Duration
}

// Listing Types
type EntityListEntry struct {
	Name     string
	CPUUsage uint64
	MemUsage uint64
}

// Internal Types
type InfoType struct {
	Creation time.Time
	Metrics  map[string]*daystore.DayStore // key: Metric Name
	Labels   map[string]string             // key: Label
	// Context retains instantaneous state for a specific InfoType.
	// Currently used for calculating instantaneous metrics from cumulative counterparts.
	Context map[string]*statstore.TimePoint // key: metric name
}

type ClusterInfo struct {
	InfoType
	Namespaces map[string]*NamespaceInfo // key: Namespace Name
	Nodes      map[string]*NodeInfo      // key: Hostname
}

type NamespaceInfo struct {
	InfoType
	Name string
	Pods map[string]*PodInfo // key: Pod Name
}

type NodeInfo struct {
	InfoType
	Name           string
	Pods           map[string]*PodInfo       // key: Pod Name
	FreeContainers map[string]*ContainerInfo // key: Container Name
}

type PodInfo struct {
	InfoType
	UID        string
	Name       string
	Namespace  string
	Hostname   string
	Containers map[string]*ContainerInfo // key: Container Name
}

type ContainerInfo struct {
	InfoType
}
