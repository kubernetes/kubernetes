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

package perftype

// ResourceSeries defines the time series of the resource usage.
type ResourceSeries struct {
	Timestamp            []int64           `json:"ts"`
	CPUUsageInMilliCores []int64           `json:"cpu"`
	MemoryRSSInMegaBytes []int64           `json:"memory"`
	Units                map[string]string `json:"unit"`
}

// NodeTimeSeries defines the time series of the operations and the resource
// usage.
type NodeTimeSeries struct {
	OperationData map[string][]int64         `json:"op_series,omitempty"`
	ResourceData  map[string]*ResourceSeries `json:"resource_series,omitempty"`
	Labels        map[string]string          `json:"labels"`
	Version       string                     `json:"version"`
}
