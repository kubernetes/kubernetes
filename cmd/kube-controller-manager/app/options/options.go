/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Package options provides the flags used for the controller manager.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/pkg/controllermanager/controllermanager.go
package options

import (
	"net"
	"time"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/master/ports"

	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	Port                              int
	Address                           net.IP
	CloudProvider                     string
	CloudConfigFile                   string
	ConcurrentEndpointSyncs           int
	ConcurrentRCSyncs                 int
	ConcurrentRSSyncs                 int
	ConcurrentDSCSyncs                int
	ConcurrentJobSyncs                int
	ConcurrentResourceQuotaSyncs      int
	ConcurrentDeploymentSyncs         int
	ServiceSyncPeriod                 time.Duration
	NodeSyncPeriod                    time.Duration
	ResourceQuotaSyncPeriod           time.Duration
	NamespaceSyncPeriod               time.Duration
	PVClaimBinderSyncPeriod           time.Duration
	VolumeConfigFlags                 VolumeConfigFlags
	TerminatedPodGCThreshold          int
	HorizontalPodAutoscalerSyncPeriod time.Duration
	DeploymentControllerSyncPeriod    time.Duration
	MinResyncPeriod                   time.Duration
	RegisterRetryCount                int
	NodeMonitorGracePeriod            time.Duration
	NodeStartupGracePeriod            time.Duration
	NodeMonitorPeriod                 time.Duration
	NodeStatusUpdateRetry             int
	PodEvictionTimeout                time.Duration
	DeletingPodsQps                   float32
	DeletingPodsBurst                 int
	ServiceAccountKeyFile             string
	RootCAFile                        string

	ClusterName       string
	ClusterCIDR       net.IPNet
	AllocateNodeCIDRs bool
	EnableProfiling   bool

	Master       string
	Kubeconfig   string
	KubeAPIQPS   float32
	KubeAPIBurst int

	LeaderElection componentconfig.LeaderElectionConfiguration
}

// VolumeConfigFlags is used to bind CLI flags to variables.  This top-level struct contains *all* enumerated
// CLI flags meant to configure all volume plugins.  From this config, the binary will create many instances
// of volume.VolumeConfig which are then passed to the appropriate plugin. The ControllerManager binary is the only
// part of the code which knows what plugins are supported and which CLI flags correspond to each plugin.
type VolumeConfigFlags struct {
	PersistentVolumeRecyclerMaximumRetry                int
	PersistentVolumeRecyclerMinimumTimeoutNFS           int
	PersistentVolumeRecyclerPodTemplateFilePathNFS      string
	PersistentVolumeRecyclerIncrementTimeoutNFS         int
	PersistentVolumeRecyclerPodTemplateFilePathHostPath string
	PersistentVolumeRecyclerMinimumTimeoutHostPath      int
	PersistentVolumeRecyclerIncrementTimeoutHostPath    int
	EnableHostPathProvisioning                          bool
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	s := CMServer{
		Port:                              ports.ControllerManagerPort,
		Address:                           net.ParseIP("0.0.0.0"),
		ConcurrentEndpointSyncs:           5,
		ConcurrentRCSyncs:                 5,
		ConcurrentRSSyncs:                 5,
		ConcurrentDSCSyncs:                2,
		ConcurrentJobSyncs:                5,
		ConcurrentResourceQuotaSyncs:      5,
		ConcurrentDeploymentSyncs:         5,
		ServiceSyncPeriod:                 5 * time.Minute,
		NodeSyncPeriod:                    10 * time.Second,
		ResourceQuotaSyncPeriod:           5 * time.Minute,
		NamespaceSyncPeriod:               5 * time.Minute,
		PVClaimBinderSyncPeriod:           10 * time.Minute,
		HorizontalPodAutoscalerSyncPeriod: 30 * time.Second,
		DeploymentControllerSyncPeriod:    30 * time.Second,
		MinResyncPeriod:                   12 * time.Hour,
		RegisterRetryCount:                10,
		PodEvictionTimeout:                5 * time.Minute,
		NodeMonitorGracePeriod:            40 * time.Second,
		NodeStartupGracePeriod:            60 * time.Second,
		NodeMonitorPeriod:                 5 * time.Second,
		ClusterName:                       "kubernetes",
		TerminatedPodGCThreshold:          12500,
		VolumeConfigFlags: VolumeConfigFlags{
			// default values here
			PersistentVolumeRecyclerMaximumRetry:             3,
			PersistentVolumeRecyclerMinimumTimeoutNFS:        300,
			PersistentVolumeRecyclerIncrementTimeoutNFS:      30,
			PersistentVolumeRecyclerMinimumTimeoutHostPath:   60,
			PersistentVolumeRecyclerIncrementTimeoutHostPath: 30,
			EnableHostPathProvisioning:                       false,
		},
		KubeAPIQPS:     20.0,
		KubeAPIBurst:   30,
		LeaderElection: leaderelection.DefaultLeaderElectionConfiguration(),
	}
	return &s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.IPVar(&s.Address, "address", s.Address, "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.IntVar(&s.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", s.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentRCSyncs, "concurrent_rc_syncs", s.ConcurrentRCSyncs, "The number of replication controllers that are allowed to sync concurrently. Larger number = more reponsive replica management, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentRSSyncs, "concurrent-replicaset-syncs", s.ConcurrentRSSyncs, "The number of replica sets that are allowed to sync concurrently. Larger number = more reponsive replica management, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", s.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.IntVar(&s.ConcurrentDeploymentSyncs, "concurrent-deployment-syncs", s.ConcurrentDeploymentSyncs, "The number of deployment objects that are allowed to sync concurrently. Larger number = more reponsive deployments, but more CPU (and network) load")
	fs.DurationVar(&s.ServiceSyncPeriod, "service-sync-period", s.ServiceSyncPeriod, "The period for syncing services with their external load balancers")
	fs.DurationVar(&s.NodeSyncPeriod, "node-sync-period", s.NodeSyncPeriod, ""+
		"The period for syncing nodes from cloudprovider. Longer periods will result in "+
		"fewer calls to cloud provider, but may delay addition of new nodes to cluster.")
	fs.DurationVar(&s.ResourceQuotaSyncPeriod, "resource-quota-sync-period", s.ResourceQuotaSyncPeriod, "The period for syncing quota usage status in the system")
	fs.DurationVar(&s.NamespaceSyncPeriod, "namespace-sync-period", s.NamespaceSyncPeriod, "The period for syncing namespace life-cycle updates")
	fs.DurationVar(&s.PVClaimBinderSyncPeriod, "pvclaimbinder-sync-period", s.PVClaimBinderSyncPeriod, "The period for syncing persistent volumes and persistent volume claims")
	fs.DurationVar(&s.MinResyncPeriod, "min-resync-period", s.MinResyncPeriod, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod")
	fs.StringVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS, "pv-recycler-pod-template-filepath-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathNFS, "The file path to a pod definition used as a template for NFS persistent volume recycling")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "pv-recycler-minimum-timeout-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutNFS, "The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "pv-recycler-increment-timeout-nfs", s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutNFS, "the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod")
	fs.StringVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, "pv-recycler-pod-template-filepath-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, "The file path to a pod definition used as a template for HostPath persistent volume recycling. This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath, "pv-recycler-minimum-timeout-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerMinimumTimeoutHostPath, "The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.IntVar(&s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath, "pv-recycler-timeout-increment-hostpath", s.VolumeConfigFlags.PersistentVolumeRecyclerIncrementTimeoutHostPath, "the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.BoolVar(&s.VolumeConfigFlags.EnableHostPathProvisioning, "enable-hostpath-provisioner", s.VolumeConfigFlags.EnableHostPathProvisioning, "Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features.  HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.")
	fs.IntVar(&s.TerminatedPodGCThreshold, "terminated-pod-gc-threshold", s.TerminatedPodGCThreshold, "Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. If <= 0, the terminated pod garbage collector is disabled.")
	fs.DurationVar(&s.HorizontalPodAutoscalerSyncPeriod, "horizontal-pod-autoscaler-sync-period", s.HorizontalPodAutoscalerSyncPeriod, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&s.DeploymentControllerSyncPeriod, "deployment-controller-sync-period", s.DeploymentControllerSyncPeriod, "Period for syncing the deployments.")
	fs.DurationVar(&s.PodEvictionTimeout, "pod-eviction-timeout", s.PodEvictionTimeout, "The grace period for deleting pods on failed nodes.")
	fs.Float32Var(&s.DeletingPodsQps, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.IntVar(&s.DeletingPodsBurst, "deleting-pods-burst", 10, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.IntVar(&s.RegisterRetryCount, "register-retry-count", s.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.NodeMonitorGracePeriod, "node-monitor-grace-period", s.NodeMonitorGracePeriod,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealty. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status.")
	fs.DurationVar(&s.NodeStartupGracePeriod, "node-startup-grace-period", s.NodeStartupGracePeriod,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealty.")
	fs.DurationVar(&s.NodeMonitorPeriod, "node-monitor-period", s.NodeMonitorPeriod,
		"The period for syncing NodeStatus in NodeController.")
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA key used to sign service account tokens.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.StringVar(&s.ClusterName, "cluster-name", s.ClusterName, "The instance prefix for the cluster")
	fs.IPNetVar(&s.ClusterCIDR, "cluster-cidr", s.ClusterCIDR, "CIDR Range for Pods in cluster.")
	fs.BoolVar(&s.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.IntVar(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	leaderelection.BindFlags(&s.LeaderElection, fs)
}
