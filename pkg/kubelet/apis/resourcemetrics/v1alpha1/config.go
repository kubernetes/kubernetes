/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"time"

	summary "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
)

// Version is the string representation of the version of this configuration
const Version = "v1alpha1"

// Config is the v1alpha1 resource metrics definition
func Config() stats.ResourceMetricsConfig {
	return stats.ResourceMetricsConfig{
		NodeMetrics: []stats.NodeResourceMetric{
			{
				Name:        "node_cpu_usage_seconds_total",
				Description: "Cumulative cpu time consumed by the node in core-seconds",
				ValueFn: func(s summary.NodeStats) (*float64, time.Time) {
					if s.CPU == nil {
						return nil, time.Time{}
					}
					v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
					return &v, s.CPU.Time.Time
				},
			},
			{
				Name:        "node_memory_working_set_bytes",
				Description: "Current working set of the node in bytes",
				ValueFn: func(s summary.NodeStats) (*float64, time.Time) {
					if s.Memory == nil {
						return nil, time.Time{}
					}
					v := float64(*s.Memory.WorkingSetBytes)
					return &v, s.Memory.Time.Time
				},
			},
		},
		ContainerMetrics: []stats.ContainerResourceMetric{
			{
				Name:        "container_cpu_usage_seconds_total",
				Description: "Cumulative cpu time consumed by the container in core-seconds",
				ValueFn: func(s summary.ContainerStats) (*float64, time.Time) {
					if s.CPU == nil {
						return nil, time.Time{}
					}
					v := float64(*s.CPU.UsageCoreNanoSeconds) / float64(time.Second)
					return &v, s.CPU.Time.Time
				},
			},
			{
				Name:        "container_memory_working_set_bytes",
				Description: "Current working set of the container in bytes",
				ValueFn: func(s summary.ContainerStats) (*float64, time.Time) {
					if s.Memory == nil {
						return nil, time.Time{}
					}
					v := float64(*s.Memory.WorkingSetBytes)
					return &v, s.Memory.Time.Time
				},
			},
		},
	}
}
