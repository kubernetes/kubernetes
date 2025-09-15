/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"fmt"

	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
)

// NodeLifecycleControllerOptions holds the NodeLifecycleController options.
type NodeLifecycleControllerOptions struct {
	*nodelifecycleconfig.NodeLifecycleControllerConfiguration
}

// AddFlags adds flags related to NodeLifecycleController for controller manager to the specified FlagSet.
func (o *NodeLifecycleControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.DurationVar(&o.NodeStartupGracePeriod.Duration, "node-startup-grace-period", o.NodeStartupGracePeriod.Duration,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealthy.")
	fs.DurationVar(&o.NodeMonitorGracePeriod.Duration, "node-monitor-grace-period", o.NodeMonitorGracePeriod.Duration,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealthy. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status. "+
			"This value should also be greater than the sum of HTTP2_PING_TIMEOUT_SECONDS and HTTP2_READ_IDLE_TIMEOUT_SECONDS")
	fs.Float32Var(&o.NodeEvictionRate, "node-eviction-rate", 0.1, "Number of nodes per second on which pods are deleted in case of node failure when a zone is healthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters.")
	fs.Float32Var(&o.SecondaryNodeEvictionRate, "secondary-node-eviction-rate", 0.01, "Number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters. This value is implicitly overridden to 0 if the cluster size is smaller than --large-cluster-size-threshold.")
	fs.Int32Var(&o.LargeClusterSizeThreshold, "large-cluster-size-threshold", 50, fmt.Sprintf("Number of nodes from which %s treats the cluster as large for the eviction logic purposes. --secondary-node-eviction-rate is implicitly overridden to 0 for clusters this size or smaller. Notice: If nodes reside in multiple zones, this threshold will be considered as zone node size threshold for each zone to determine node eviction rate independently.", names.NodeLifecycleController))
	fs.Float32Var(&o.UnhealthyZoneThreshold, "unhealthy-zone-threshold", 0.55, "Fraction of Nodes in a zone which needs to be not Ready (minimum 3) for zone to be treated as unhealthy. ")
}

// ApplyTo fills up NodeLifecycleController config with options.
func (o *NodeLifecycleControllerOptions) ApplyTo(cfg *nodelifecycleconfig.NodeLifecycleControllerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.NodeStartupGracePeriod = o.NodeStartupGracePeriod
	cfg.NodeMonitorGracePeriod = o.NodeMonitorGracePeriod
	cfg.NodeEvictionRate = o.NodeEvictionRate
	cfg.SecondaryNodeEvictionRate = o.SecondaryNodeEvictionRate
	cfg.LargeClusterSizeThreshold = o.LargeClusterSizeThreshold
	cfg.UnhealthyZoneThreshold = o.UnhealthyZoneThreshold

	return nil
}

// Validate checks validation of NodeLifecycleControllerOptions.
func (o *NodeLifecycleControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
