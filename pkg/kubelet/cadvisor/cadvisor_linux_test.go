//go:build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package cadvisor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	cadvisormetrics "github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/crio"
	cadvisorfs "github.com/google/cadvisor/fs"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func TestImageFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioImages,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no imagefs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ImageFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}

func TestContainerFsInfoLabel(t *testing.T) {
	testcases := []struct {
		description     string
		runtime         string
		runtimeEndpoint string
		expectedLabel   string
		expectedError   error
	}{{
		description:     "LabelCrioWriteableImages should be returned",
		runtimeEndpoint: crio.CrioSocket,
		expectedLabel:   cadvisorfs.LabelCrioContainers,
		expectedError:   nil,
	}, {
		description:     "Cannot find valid imagefs label",
		runtimeEndpoint: "",
		expectedLabel:   "",
		expectedError:   fmt.Errorf("no containerfs label for configured runtime"),
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			infoProvider := NewImageFsInfoProvider(tc.runtimeEndpoint)
			label, err := infoProvider.ContainerFsInfoLabel()
			assert.Equal(t, tc.expectedLabel, label)
			assert.Equal(t, tc.expectedError, err)
		})
	}
}

func TestProcessMetricsConfiguration(t *testing.T) {
	tests := []struct {
		name                 string
		cadvisorConfig       *kubeletconfig.CAdvisorConfiguration
		expectProcessMetrics bool
	}{
		{
			name:                 "nil config - ProcessMetrics enabled (backward compatible)",
			cadvisorConfig:       nil,
			expectProcessMetrics: true,
		},
		{
			name:                 "empty config - ProcessMetrics enabled (backward compatible)",
			cadvisorConfig:       &kubeletconfig.CAdvisorConfiguration{},
			expectProcessMetrics: true,
		},
		{
			name: "ProcessMetrics nil - enabled (backward compatible)",
			cadvisorConfig: &kubeletconfig.CAdvisorConfiguration{
				IncludedMetrics: kubeletconfig.CAdvisorIncludedMetrics{
					ProcessMetrics: nil,
				},
			},
			expectProcessMetrics: true,
		},
		{
			name: "ProcessMetrics explicitly true - enabled",
			cadvisorConfig: &kubeletconfig.CAdvisorConfiguration{
				IncludedMetrics: kubeletconfig.CAdvisorIncludedMetrics{
					ProcessMetrics: ptr.To(true),
				},
			},
			expectProcessMetrics: true,
		},
		{
			name: "ProcessMetrics explicitly false - disabled",
			cadvisorConfig: &kubeletconfig.CAdvisorConfiguration{
				IncludedMetrics: kubeletconfig.CAdvisorIncludedMetrics{
					ProcessMetrics: ptr.To(false),
				},
			},
			expectProcessMetrics: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test the ProcessMetricsEnabled helper function
			got := tt.cadvisorConfig.ProcessMetricsEnabled()
			if got != tt.expectProcessMetrics {
				t.Errorf("ProcessMetricsEnabled() = %v, want %v", got, tt.expectProcessMetrics)
			}
		})
	}
}

// TestBackwardCompatibility ensures that existing KubeletConfiguration
// without the CAdvisor field continues to work with ProcessMetrics enabled.
func TestBackwardCompatibility(t *testing.T) {
	// Simulate an upgrade scenario where CAdvisor config is not set
	// (zero value of struct should enable ProcessMetrics)
	config := &kubeletconfig.CAdvisorConfiguration{}

	if !config.ProcessMetricsEnabled() {
		t.Error("ProcessMetrics should be enabled by default for backward compatibility")
	}
}

func TestDetermineIncludedMetrics_ProcessMetricsFeatureGate(t *testing.T) {
	tests := []struct {
		name                 string
		gateEnabled          bool
		processMetricsConfig *bool
		expectIncluded       bool
	}{
		{
			name:                 "feature gate disabled always includes ProcessMetrics",
			gateEnabled:          false,
			processMetricsConfig: ptr.To(false),
			expectIncluded:       true,
		},
		{
			name:                 "feature gate enabled default includes ProcessMetrics",
			gateEnabled:          true,
			processMetricsConfig: nil,
			expectIncluded:       true,
		},
		{
			name:                 "feature gate enabled config disables ProcessMetrics",
			gateEnabled:          true,
			processMetricsConfig: ptr.To(false),
			expectIncluded:       false,
		},
		{
			name:                 "feature gate enabled config enables ProcessMetrics",
			gateEnabled:          true,
			processMetricsConfig: ptr.To(true),
			expectIncluded:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultMutableFeatureGate, features.ConfigurableCAdvisorMetrics, tt.gateEnabled)
			cfg := &kubeletconfig.CAdvisorConfiguration{}
			if tt.processMetricsConfig != nil {
				cfg.IncludedMetrics.ProcessMetrics = tt.processMetricsConfig
			}

			metrics := determineIncludedMetrics(cfg, false, false)
			_, present := metrics[cadvisormetrics.ProcessMetrics]
			if present != tt.expectIncluded {
				t.Fatalf("ProcessMetrics included = %v, want %v", present, tt.expectIncluded)
			}
		})
	}
}
