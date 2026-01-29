/*
Copyright 2026 The Kubernetes Authors.

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

package config

// CAdvisorConfiguration contains settings for the embedded cAdvisor.
type CAdvisorConfiguration struct {
	// IncludedMetrics specifies which cAdvisor metric collectors are enabled.
	// All collectors are enabled by default for backward compatibility.
	// +optional
	IncludedMetrics CAdvisorIncludedMetrics
}

// CAdvisorIncludedMetrics specifies which cAdvisor metric collectors to enable.
// All fields default to true for backward compatibility.
type CAdvisorIncludedMetrics struct {
	// ProcessMetrics enables collection of process/thread metrics.
	// These metrics scan /proc for every thread in every container, which
	// causes significant CPU overhead on high-density nodes (100+ pods).
	// Disabling this can reduce kubelet CPU usage by up to 99% on such nodes.
	//
	// Affected metrics when disabled:
	//   - container_processes
	//   - container_threads
	//   - container_file_descriptors
	//   - container_sockets
	//   - container_ulimits_soft
	//   - container_ulimits_hard
	//
	// Default: true (enabled for backward compatibility)
	// +optional
	ProcessMetrics *bool
}

// ProcessMetricsEnabled returns whether ProcessMetrics collection is enabled.
// Returns true if not explicitly disabled (backward compatible default).
func (c *CAdvisorConfiguration) ProcessMetricsEnabled() bool {
	if c == nil {
		return true
	}
	if c.IncludedMetrics.ProcessMetrics == nil {
		return true
	}
	return *c.IncludedMetrics.ProcessMetrics
}
