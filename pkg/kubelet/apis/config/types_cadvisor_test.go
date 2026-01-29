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

import (
	"testing"

	"k8s.io/utils/ptr"
)

func TestCAdvisorConfiguration_ProcessMetricsEnabled(t *testing.T) {
	tests := []struct {
		name   string
		config *CAdvisorConfiguration
		want   bool
	}{
		{
			name:   "nil config returns true (backward compatible default)",
			config: nil,
			want:   true,
		},
		{
			name:   "empty config returns true (backward compatible default)",
			config: &CAdvisorConfiguration{},
			want:   true,
		},
		{
			name: "nil ProcessMetrics returns true (backward compatible default)",
			config: &CAdvisorConfiguration{
				IncludedMetrics: CAdvisorIncludedMetrics{
					ProcessMetrics: nil,
				},
			},
			want: true,
		},
		{
			name: "ProcessMetrics explicitly true returns true",
			config: &CAdvisorConfiguration{
				IncludedMetrics: CAdvisorIncludedMetrics{
					ProcessMetrics: ptr.To(true),
				},
			},
			want: true,
		},
		{
			name: "ProcessMetrics explicitly false returns false",
			config: &CAdvisorConfiguration{
				IncludedMetrics: CAdvisorIncludedMetrics{
					ProcessMetrics: ptr.To(false),
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.config.ProcessMetricsEnabled()
			if got != tt.want {
				t.Errorf("ProcessMetricsEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestCAdvisorConfiguration_BackwardCompatibility verifies that the default
// behavior matches pre-feature Kubernetes versions where ProcessMetrics
// was always collected.
func TestCAdvisorConfiguration_BackwardCompatibility(t *testing.T) {
	// Simulate what happens when CAdvisor field is not set in KubeletConfiguration
	// (which is the case for existing clusters upgrading to new Kubernetes version)
	kc := KubeletConfiguration{}

	// The default zero value should result in ProcessMetrics being enabled
	if !kc.CAdvisor.ProcessMetricsEnabled() {
		t.Error("Expected ProcessMetrics to be enabled by default for backward compatibility")
	}
}
