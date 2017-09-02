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

package validation

import (
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestValidateKubeControllerManagerConfiguration(t *testing.T) {
	successCase := &componentconfig.KubeControllerManagerConfiguration{
		ConcurrentDeploymentSyncs:    10,
		ConcurrentEndpointSyncs:      10,
		ConcurrentGCSyncs:            10,
		ConcurrentNamespaceSyncs:     10,
		ConcurrentRSSyncs:            10,
		ConcurrentResourceQuotaSyncs: 10,
		ConcurrentServiceSyncs:       10,
		ConcurrentSATokenSyncs:       10,
		ConcurrentRCSyncs:            10,
		KubeAPIBurst:                 10,
		KubeAPIQPS:                   10,
		LargeClusterSizeThreshold:    10,
		NodeCIDRMaskSize:             10,
		NodeEvictionRate:             10,
		Port:                         10,
		VolumeConfiguration: componentconfig.VolumeConfiguration{
			PersistentVolumeRecyclerConfiguration: componentconfig.PersistentVolumeRecyclerConfiguration{
				IncrementTimeoutNFS:      10,
				MinimumTimeoutHostPath:   10,
				MinimumTimeoutNFS:        10,
				IncrementTimeoutHostPath: 10,
			},
		},
		SecondaryNodeEvictionRate: 10,
	}
	if allErrors := ValidateKubeControllerManagerConfiguration(successCase); allErrors != nil {
		t.Errorf("expect no errors got %v", allErrors)
	}

	errorCase := &componentconfig.KubeControllerManagerConfiguration{
		ConcurrentDeploymentSyncs:    -10,
		ConcurrentEndpointSyncs:      -10,
		ConcurrentGCSyncs:            -10,
		ConcurrentNamespaceSyncs:     -10,
		ConcurrentRSSyncs:            -10,
		ConcurrentResourceQuotaSyncs: -10,
		ConcurrentServiceSyncs:       -10,
		ConcurrentSATokenSyncs:       -10,
		ConcurrentRCSyncs:            -10,
		KubeAPIBurst:                 -10,
		KubeAPIQPS:                   -10,
		LargeClusterSizeThreshold:    -10,
		NodeCIDRMaskSize:             -10,
		NodeEvictionRate:             -10,
		Port:                         0,
		VolumeConfiguration: componentconfig.VolumeConfiguration{
			PersistentVolumeRecyclerConfiguration: componentconfig.PersistentVolumeRecyclerConfiguration{
				IncrementTimeoutNFS:      -10,
				MinimumTimeoutHostPath:   -10,
				MinimumTimeoutNFS:        -10,
				IncrementTimeoutHostPath: -10,
			},
		},
		SecondaryNodeEvictionRate: -10,
	}
	if allErrors := ValidateKubeControllerManagerConfiguration(errorCase); len(allErrors.(utilerrors.Aggregate).Errors()) != 20 {
		t.Errorf("expect 20 errors got %v", len(allErrors.(utilerrors.Aggregate).Errors()))
	}
}
