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

package types

import (
	"time"
)

type MetricPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     uint64    `json:"value"`
	// This will be populated only for float custom metrics. In that case
	// "value" will be zero. This is a temporary hack. Overall most likely
	// we will need a new api versioned in the similar way as K8S api.
	FloatValue *float64 `json:"floatValue,omitempty"`
}

type MetricResult struct {
	Metrics         []MetricPoint `json:"metrics"`
	LatestTimestamp time.Time     `json:"latestTimestamp"`
}

type MetricResultList struct {
	Items []MetricResult `json:"items"`
}

type Stats struct {
	Average     uint64 `json:"average"`
	NinetyFifth uint64 `json:"percentile"`
	Max         uint64 `json:"max"`
}

type ExternalStatBundle struct {
	Minute Stats `json:"minute"`
	Hour   Stats `json:"hour"`
	Day    Stats `json:"day"`
}

type StatsResponse struct {
	// Uptime is in seconds
	Uptime uint64                        `json:"uptime"`
	Stats  map[string]ExternalStatBundle `json:"stats"`
}

// An ExternalEntityListEntry represents the latest CPU and Memory usage of a model entity.
// A model entity can be a Pod, a Container, a Namespace or a Node.
type ExternalEntityListEntry struct {
	Name     string `json:"name"`
	CPUUsage uint64 `json:"cpuUsage"`
	MemUsage uint64 `json:"memUsage"`
}
