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
	"fmt"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// ValidateKubeControllerManagerConfiguration validates KubeControllerManagerConfiguration
// and returns an aggregated error if it is invalid.
func ValidateKubeControllerManagerConfiguration(kc *componentconfig.KubeControllerManagerConfiguration) error {
	allErrors := []error{}

	if kc.ConcurrentDeploymentSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentDeploymentSyncs (--concurrent-deployment-syncs) %v must not be a negative number", kc.ConcurrentDeploymentSyncs))
	}
	if kc.ConcurrentEndpointSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentEndpointSyncs (--concurrent-endpoint-syncs) %v must not be a negative number", kc.ConcurrentEndpointSyncs))
	}
	if kc.ConcurrentGCSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentGCSyncs (--concurrent-gc-syncs) %v must not be a negative number", kc.ConcurrentGCSyncs))
	}
	if kc.ConcurrentNamespaceSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentNamespaceSyncs (--concurrent-namespace-syncs) %v must not be a negative number", kc.ConcurrentNamespaceSyncs))
	}
	if kc.ConcurrentRSSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentRSSyncs (--concurrent-replicaset-syncs) %v must not be a negative number", kc.ConcurrentRSSyncs))
	}
	if kc.ConcurrentResourceQuotaSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentResourceQuotaSyncs (--concurrent-resource-quota-syncs) %v must not be a negative number", kc.ConcurrentResourceQuotaSyncs))
	}
	if kc.ConcurrentServiceSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentServiceSyncs (--concurrent-service-syncs) %v must not be a negative number", kc.ConcurrentServiceSyncs))
	}
	if kc.ConcurrentSATokenSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentSATokenSyncs (--concurrent-serviceaccount-token-syncs) %v must not be a negative number", kc.ConcurrentSATokenSyncs))
	}
	if kc.ConcurrentRCSyncs < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: ConcurrentRCSyncs (--concurrent_rc_syncs) %v must not be a negative number", kc.ConcurrentRCSyncs))
	}
	if kc.KubeAPIBurst < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: KubeAPIBurst (--kube-api-burst) %v must not be a negative number", kc.KubeAPIBurst))
	}
	if kc.KubeAPIQPS < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: KubeAPIQPS (--kube-api-qps) %v must not be a negative number", kc.KubeAPIQPS))
	}
	if kc.LargeClusterSizeThreshold < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: LargeClusterSizeThreshold (--large-cluster-size-threshold) %v must not be a negative number", kc.LargeClusterSizeThreshold))
	}
	if kc.NodeCIDRMaskSize < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: NodeCIDRMaskSize (--node-cidr-mask-size) %v must not be a negative number", kc.NodeCIDRMaskSize))
	}
	if kc.NodeEvictionRate < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: NodeEvictionRate (--node-eviction-rate) %v must not be a negative number", kc.NodeEvictionRate))
	}
	if utilvalidation.IsValidPortNum(int(kc.Port)) != nil {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: Port (--port) %v must be between 1 and 65535, inclusive", kc.Port))
	}
	if kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: IncrementTimeoutNFS (--pv-recycler-increment-timeout-nfs) %v must not be a negative number", kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS))
	}
	if kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: MinimumTimeoutHostPath (--pv-recycler-minimum-timeout-nfs) %v must not be a negative number", kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath))
	}
	if kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: MinimumTimeoutNFS (--pv-recycler-increment-timeout-nfs) %v must not be a negative number", kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS))
	}
	if kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: IncrementTimeoutHostPath (--pv-recycler-timeout-increment-hostpath) %v must not be a negative number", kc.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath))
	}
	if kc.SecondaryNodeEvictionRate < 0 {
		allErrors = append(allErrors, fmt.Errorf("Invalid configuration: SecondaryNodeEvictionRate (--secondary-node-eviction-rate) %v must not be a negative number", kc.SecondaryNodeEvictionRate))
	}

	return utilerrors.NewAggregate(allErrors)
}
