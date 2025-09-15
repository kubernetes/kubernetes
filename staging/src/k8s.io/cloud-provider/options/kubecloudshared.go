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

	cpconfig "k8s.io/cloud-provider/config"
	"k8s.io/cloud-provider/names"
)

// KubeCloudSharedOptions holds the options shared between kube-controller-manager
// and cloud-controller-manager.
type KubeCloudSharedOptions struct {
	*cpconfig.KubeCloudSharedConfiguration
	CloudProvider *CloudProviderOptions
}

// NewKubeCloudSharedOptions returns common/default configuration values for both
// the kube-controller-manager and the cloud-contoller-manager. Any common changes should
// be made here. Any individual changes should be made in that controller.
func NewKubeCloudSharedOptions(cfg *cpconfig.KubeCloudSharedConfiguration) *KubeCloudSharedOptions {
	o := &KubeCloudSharedOptions{
		KubeCloudSharedConfiguration: cfg,
		CloudProvider: &CloudProviderOptions{
			CloudProviderConfiguration: &cpconfig.CloudProviderConfiguration{},
		},
	}

	return o
}

// AddFlags adds flags related to shared variable for controller manager to the specified FlagSet.
func (o *KubeCloudSharedOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	o.CloudProvider.AddFlags(fs)
	fs.StringVar(&o.ExternalCloudVolumePlugin, "external-cloud-volume-plugin", o.ExternalCloudVolumePlugin, "The plugin to use when cloud provider is set to external. Can be empty, should only be set when cloud-provider is external. Currently used to allow node-ipam-controller, persistentvolume-binder-controller, persistentvolume-expander-controller and attach-detach-controller to work for in tree cloud providers.")
	fs.BoolVar(&o.UseServiceAccountCredentials, "use-service-account-credentials", o.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.BoolVar(&o.AllowUntaggedCloud, "allow-untagged-cloud", false, "Allow the cluster to run without the cluster-id on cloud instances. This is a legacy mode of operation and a cluster-id will be required in the future.")
	fs.MarkDeprecated("allow-untagged-cloud", "This flag is deprecated and will be removed in a future release. A cluster-id will be required on cloud instances.")
	fs.DurationVar(&o.RouteReconciliationPeriod.Duration, "route-reconciliation-period", o.RouteReconciliationPeriod.Duration, "The period for reconciling routes created for Nodes by cloud provider.")
	fs.DurationVar(&o.NodeMonitorPeriod.Duration, "node-monitor-period", o.NodeMonitorPeriod.Duration,
		fmt.Sprintf("The period for syncing NodeStatus in %s.", names.CloudNodeLifecycleController))
	fs.StringVar(&o.ClusterName, "cluster-name", o.ClusterName, "The instance prefix for the cluster.")
	fs.StringVar(&o.ClusterCIDR, "cluster-cidr", o.ClusterCIDR, "CIDR Range for Pods in cluster. Only used when --allocate-node-cidrs=true; if false, this option will be ignored.")
	fs.BoolVar(&o.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider. Requires --cluster-cidr.")
	fs.StringVar(&o.CIDRAllocatorType, "cidr-allocator-type", "RangeAllocator", "Type of CIDR allocator to use")
	fs.BoolVar(&o.ConfigureCloudRoutes, "configure-cloud-routes", true, "Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.")

	fs.DurationVar(&o.NodeSyncPeriod.Duration, "node-sync-period", 0, ""+
		"This flag is deprecated and will be removed in future releases. See node-monitor-period for Node health checking or "+
		"route-reconciliation-period for cloud provider's route configuration settings.")
	fs.MarkDeprecated("node-sync-period", "This flag is currently no-op and will be deleted.")
}

// ApplyTo fills up KubeCloudShared config with options.
func (o *KubeCloudSharedOptions) ApplyTo(cfg *cpconfig.KubeCloudSharedConfiguration) error {
	if o == nil {
		return nil
	}

	if err := o.CloudProvider.ApplyTo(&cfg.CloudProvider); err != nil {
		return err
	}

	cfg.ExternalCloudVolumePlugin = o.ExternalCloudVolumePlugin
	cfg.UseServiceAccountCredentials = o.UseServiceAccountCredentials
	cfg.AllowUntaggedCloud = o.AllowUntaggedCloud
	cfg.RouteReconciliationPeriod = o.RouteReconciliationPeriod
	cfg.NodeMonitorPeriod = o.NodeMonitorPeriod
	cfg.ClusterName = o.ClusterName
	cfg.ClusterCIDR = o.ClusterCIDR
	cfg.AllocateNodeCIDRs = o.AllocateNodeCIDRs
	cfg.CIDRAllocatorType = o.CIDRAllocatorType
	cfg.ConfigureCloudRoutes = o.ConfigureCloudRoutes
	cfg.NodeSyncPeriod = o.NodeSyncPeriod

	return nil
}

// Validate checks validation of KubeCloudSharedOptions.
func (o *KubeCloudSharedOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	errs = append(errs, o.CloudProvider.Validate()...)

	return errs
}
