/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"net"
	"time"

	"github.com/spf13/pflag"
)

type NodeControllerOptions struct {
	AllocateNodeCIDRs      bool
	ClusterCIDR            net.IPNet
	DeletingPodsBurst      int
	DeletingPodsQps        float32
	NodeMonitorGracePeriod time.Duration
	NodeMonitorPeriod      time.Duration
	NodeStartupGracePeriod time.Duration
	PodEvictionTimeout     time.Duration
}

func NewNodeControllerOptions() NodeControllerOptions {
	return NodeControllerOptions{
		AllocateNodeCIDRs:      false,
		DeletingPodsBurst:      10,
		DeletingPodsQps:        0.1,
		NodeMonitorGracePeriod: 40 * time.Second,
		NodeMonitorPeriod:      5 * time.Second,
		NodeStartupGracePeriod: 60 * time.Second,
		PodEvictionTimeout:     5 * time.Minute,
	}
}

func (o *NodeControllerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.BoolVar(&o.AllocateNodeCIDRs, "allocate-node-cidrs", o.AllocateNodeCIDRs, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.IPNetVar(&o.ClusterCIDR, "cluster-cidr", o.ClusterCIDR, "CIDR Range for Pods in cluster.")
	fs.IntVar(&o.DeletingPodsBurst, "deleting-pods-burst", o.DeletingPodsBurst,
		"Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.Float32Var(&o.DeletingPodsQps, "deleting-pods-qps", o.DeletingPodsQps,
		"Number of nodes per second on which pods are deleted in case of node failure.")
	fs.DurationVar(&o.NodeMonitorGracePeriod, "node-monitor-grace-period", o.NodeMonitorGracePeriod,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealty. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status.")
	fs.DurationVar(&o.NodeMonitorPeriod, "node-monitor-period", o.NodeMonitorPeriod,
		"The period for syncing NodeStatus in NodeController.")
	fs.DurationVar(&o.NodeStartupGracePeriod, "node-startup-grace-period", o.NodeStartupGracePeriod,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealty.")
	fs.DurationVar(&o.PodEvictionTimeout, "pod-eviction-timeout", o.PodEvictionTimeout,
		"The grace period for deleting pods on failed nodes.")

}
