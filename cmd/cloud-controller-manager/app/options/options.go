/*
Copyright 2016 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// CloudControllerMangerServer is the main context object for the controller manager.
type CloudControllerManagerServer struct {
	componentconfig.KubeControllerManagerConfiguration

	Master     string
	Kubeconfig string

	// NodeStatusUpdateFrequency is the freuency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

// NewCloudControllerManagerServer creates a new ExternalCMServer with a default config.
func NewCloudControllerManagerServer() *CloudControllerManagerServer {
	s := CloudControllerManagerServer{
		// Part of these default values also present in 'cmd/kube-controller-manager/app/options/options.go'.
		// Please keep them in sync when doing update.
		KubeControllerManagerConfiguration: componentconfig.KubeControllerManagerConfiguration{
			Port:                      ports.CloudControllerManagerPort,
			Address:                   "0.0.0.0",
			ConcurrentServiceSyncs:    1,
			MinResyncPeriod:           metav1.Duration{Duration: 12 * time.Hour},
			NodeMonitorPeriod:         metav1.Duration{Duration: 5 * time.Second},
			ClusterName:               "kubernetes",
			ConfigureCloudRoutes:      true,
			ContentType:               "application/vnd.kubernetes.protobuf",
			KubeAPIQPS:                20.0,
			KubeAPIBurst:              30,
			LeaderElection:            leaderelectionconfig.DefaultLeaderElectionConfiguration(),
			ControllerStartInterval:   metav1.Duration{Duration: 0 * time.Second},
			RouteReconciliationPeriod: metav1.Duration{Duration: 10 * time.Second},
		},
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 5 * time.Minute},
	}
	s.LeaderElection.LeaderElect = true
	return &s
}

// AddFlags adds flags for a specific ExternalCMServer to the specified FlagSet
func (s *CloudControllerManagerServer) AddFlags(fs *pflag.FlagSet) {
	fs.Int32Var(&s.Port, "port", s.Port, "The port that the cloud-controller-manager's http service runs on.")
	fs.Var(componentconfig.IPVar{Val: &s.Address}, "address", "The IP address to serve on (set to 0.0.0.0 for all interfaces).")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider of cloud services. Cannot be empty.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file. Empty string for no configuration file.")
	fs.BoolVar(&s.AllowUntaggedCloud, "allow-untagged-cloud", false, "Allow the cluster to run without the cluster-id on cloud instances. This is a legacy mode of operation and a cluster-id will be required in the future.")
	fs.MarkDeprecated("allow-untagged-cloud", "This flag is deprecated and will be removed in a future release. A cluster-id will be required on cloud instances.")
	fs.DurationVar(&s.MinResyncPeriod.Duration, "min-resync-period", s.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod.")
	fs.DurationVar(&s.NodeMonitorPeriod.Duration, "node-monitor-period", s.NodeMonitorPeriod.Duration,
		"The period for syncing NodeStatus in NodeController.")
	fs.DurationVar(&s.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", s.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")
	// TODO: remove --service-account-private-key-file 6 months after 1.8 is released (~1.10)
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.MarkDeprecated("service-account-private-key-file", "This flag is currently no-op and will be deleted.")
	fs.BoolVar(&s.UseServiceAccountCredentials, "use-service-account-credentials", s.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.DurationVar(&s.RouteReconciliationPeriod.Duration, "route-reconciliation-period", s.RouteReconciliationPeriod.Duration, "The period for reconciling routes created for Nodes by cloud provider.")
	fs.BoolVar(&s.ConfigureCloudRoutes, "configure-cloud-routes", true, "Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/.")
	fs.BoolVar(&s.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled.")
	fs.StringVar(&s.ClusterCIDR, "cluster-cidr", s.ClusterCIDR, "CIDR Range for Pods in cluster.")
	fs.StringVar(&s.ClusterName, "cluster-name", s.ClusterName, "The instance prefix for the cluster.")
	fs.BoolVar(&s.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver.")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver.")
	fs.DurationVar(&s.ControllerStartInterval.Duration, "controller-start-interval", s.ControllerStartInterval.Duration, "Interval between starting controller managers.")

	leaderelectionconfig.BindFlags(&s.LeaderElection, fs)

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}
