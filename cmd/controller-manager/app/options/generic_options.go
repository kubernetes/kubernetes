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
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cmconfig "k8s.io/kubernetes/cmd/controller-manager/app"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
)

// GenericComponentConfigOptions is part of context object for the controller manager.
type GenericComponentConfigOptions struct {
	Port            int32
	Address         string
	CloudConfigFile string
	CloudProvider   string

	UseServiceAccountCredentials bool
	MinResyncPeriod              metav1.Duration
	ControllerStartInterval      metav1.Duration
	LeaderElection               componentconfig.LeaderElectionConfiguration

	ConcurrentServiceSyncs    int32
	ServiceAccountKeyFile     string
	AllowUntaggedCloud        bool
	RouteReconciliationPeriod metav1.Duration
	NodeMonitorPeriod         metav1.Duration
	ClusterName               string
	ClusterCIDR               string
	AllocateNodeCIDRs         bool
	CIDRAllocatorType         string
	ConfigureCloudRoutes      bool
	ContentType               string
	KubeAPIQPS                float32
	KubeAPIBurst              int32
}

// AddFlags adds flags related to debugging for controller manager to the specified FlagSet
func (o *GenericComponentConfigOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.BoolVar(&o.UseServiceAccountCredentials, "use-service-account-credentials", o.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.DurationVar(&o.MinResyncPeriod.Duration, "min-resync-period", o.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod.")
	fs.DurationVar(&o.ControllerStartInterval.Duration, "controller-start-interval", o.ControllerStartInterval.Duration, "Interval between starting controller managers.")
	fs.Int32Var(&o.ConcurrentServiceSyncs, "concurrent-service-syncs", o.ConcurrentServiceSyncs, "The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load")
	// TODO: remove --service-account-private-key-file 6 months after 1.8 is released (~1.10)
	fs.StringVar(&o.ServiceAccountKeyFile, "service-account-private-key-file", o.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.MarkDeprecated("service-account-private-key-file", "This flag is currently no-op and will be deleted.")
	fs.BoolVar(&o.AllowUntaggedCloud, "allow-untagged-cloud", false, "Allow the cluster to run without the cluster-id on cloud instances. This is a legacy mode of operation and a cluster-id will be required in the future.")
	fs.MarkDeprecated("allow-untagged-cloud", "This flag is deprecated and will be removed in a future release. A cluster-id will be required on cloud instances.")
	fs.DurationVar(&o.RouteReconciliationPeriod.Duration, "route-reconciliation-period", o.RouteReconciliationPeriod.Duration, "The period for reconciling routes created for Nodes by cloud provider.")
	fs.DurationVar(&o.NodeMonitorPeriod.Duration, "node-monitor-period", o.NodeMonitorPeriod.Duration,
		"The period for syncing NodeStatus in NodeController.")
	fs.StringVar(&o.ClusterName, "cluster-name", o.ClusterName, "The instance prefix for the cluster.")
	fs.StringVar(&o.ClusterCIDR, "cluster-cidr", o.ClusterCIDR, "CIDR Range for Pods in cluster. Requires --allocate-node-cidrs to be true")
	fs.BoolVar(&o.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&o.CIDRAllocatorType, "cidr-allocator-type", "RangeAllocator", "Type of CIDR allocator to use")
	fs.BoolVar(&o.ConfigureCloudRoutes, "configure-cloud-routes", true, "Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.")
	fs.StringVar(&o.ContentType, "kube-api-content-type", o.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.KubeAPIQPS, "kube-api-qps", o.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver.")
	fs.Int32Var(&o.KubeAPIBurst, "kube-api-burst", o.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver.")
	leaderelectionconfig.BindFlags(&o.LeaderElection, fs)
}

// ApplyTo fills up parts of controller manager config with options.
func (o *GenericComponentConfigOptions) ApplyTo(c *cmconfig.GenericConfigInfo) error {
	if o == nil {
		return nil
	}
	c = &cmconfig.GenericConfigInfo{
		UseServiceAccountCredentials: o.UseServiceAccountCredentials,
		MinResyncPeriod:              o.MinResyncPeriod,
		ControllerStartInterval:      o.ControllerStartInterval,
		ConcurrentServiceSyncs:       o.ConcurrentServiceSyncs,
		RouteReconciliationPeriod:    o.RouteReconciliationPeriod,
		NodeMonitorPeriod:            o.NodeMonitorPeriod,
		ClusterName:                  o.ClusterName,
		ConfigureCloudRoutes:         o.ConfigureCloudRoutes,
		ContentType:                  o.ContentType,
		KubeAPIQPS:                   o.KubeAPIQPS,
		KubeAPIBurst:                 o.KubeAPIBurst,
	}

	return nil
}

// Validate checks validation of GenericOptions.
func (o *GenericComponentConfigOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}
	return errs
}
