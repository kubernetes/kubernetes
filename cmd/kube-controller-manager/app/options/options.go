/*
Copyright 2014 The Kubernetes Authors.

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
package options

import (
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelection"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/spf13/pflag"
)

// CMServer is the main context object for the controller manager.
type CMServer struct {
	componentconfig.KubeControllerManagerConfiguration

	Master     string
	Kubeconfig string
}

// NewCMServer creates a new CMServer with a default config.
func NewCMServer() *CMServer {
	gcIgnoredResources := make([]componentconfig.GroupResource, 0, len(garbagecollector.DefaultIgnoredResources()))
	for r := range garbagecollector.DefaultIgnoredResources() {
		gcIgnoredResources = append(gcIgnoredResources, componentconfig.GroupResource{Group: r.Group, Resource: r.Resource})
	}

	s := CMServer{
		KubeControllerManagerConfiguration: componentconfig.KubeControllerManagerConfiguration{
			Controllers:                                     []string{"*"},
			Port:                                            ports.ControllerManagerPort,
			Address:                                         "0.0.0.0",
			ConcurrentEndpointSyncs:                         5,
			ConcurrentServiceSyncs:                          1,
			ConcurrentRCSyncs:                               5,
			ConcurrentRSSyncs:                               5,
			ConcurrentDaemonSetSyncs:                        2,
			ConcurrentJobSyncs:                              5,
			ConcurrentResourceQuotaSyncs:                    5,
			ConcurrentDeploymentSyncs:                       5,
			ConcurrentNamespaceSyncs:                        10,
			ConcurrentSATokenSyncs:                          5,
			LookupCacheSizeForRC:                            4096,
			LookupCacheSizeForRS:                            4096,
			LookupCacheSizeForDaemonSet:                     1024,
			ServiceSyncPeriod:                               metav1.Duration{Duration: 5 * time.Minute},
			RouteReconciliationPeriod:                       metav1.Duration{Duration: 10 * time.Second},
			ResourceQuotaSyncPeriod:                         metav1.Duration{Duration: 5 * time.Minute},
			NamespaceSyncPeriod:                             metav1.Duration{Duration: 5 * time.Minute},
			PVClaimBinderSyncPeriod:                         metav1.Duration{Duration: 15 * time.Second},
			HorizontalPodAutoscalerSyncPeriod:               metav1.Duration{Duration: 30 * time.Second},
			HorizontalPodAutoscalerUpscaleForbiddenWindow:   metav1.Duration{Duration: 3 * time.Minute},
			HorizontalPodAutoscalerDownscaleForbiddenWindow: metav1.Duration{Duration: 5 * time.Minute},
			DeploymentControllerSyncPeriod:                  metav1.Duration{Duration: 30 * time.Second},
			MinResyncPeriod:                                 metav1.Duration{Duration: 12 * time.Hour},
			RegisterRetryCount:                              10,
			PodEvictionTimeout:                              metav1.Duration{Duration: 5 * time.Minute},
			NodeMonitorGracePeriod:                          metav1.Duration{Duration: 40 * time.Second},
			NodeStartupGracePeriod:                          metav1.Duration{Duration: 60 * time.Second},
			NodeMonitorPeriod:                               metav1.Duration{Duration: 5 * time.Second},
			ClusterName:                                     "kubernetes",
			NodeCIDRMaskSize:                                24,
			ConfigureCloudRoutes:                            true,
			TerminatedPodGCThreshold:                        12500,
			VolumeConfiguration: componentconfig.VolumeConfiguration{
				EnableHostPathProvisioning: false,
				EnableDynamicProvisioning:  true,
				PersistentVolumeRecyclerConfiguration: componentconfig.PersistentVolumeRecyclerConfiguration{
					MaximumRetry:             3,
					MinimumTimeoutNFS:        300,
					IncrementTimeoutNFS:      30,
					MinimumTimeoutHostPath:   60,
					IncrementTimeoutHostPath: 30,
				},
				FlexVolumePluginDir: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/",
			},
			ContentType:                           "application/vnd.kubernetes.protobuf",
			KubeAPIQPS:                            20.0,
			KubeAPIBurst:                          30,
			LeaderElection:                        leaderelection.DefaultLeaderElectionConfiguration(),
			ControllerStartInterval:               metav1.Duration{Duration: 0 * time.Second},
			EnableGarbageCollector:                true,
			ConcurrentGCSyncs:                     20,
			GCIgnoredResources:                    gcIgnoredResources,
			ClusterSigningCertFile:                "/etc/kubernetes/ca/ca.pem",
			ClusterSigningKeyFile:                 "/etc/kubernetes/ca/ca.key",
			ClusterSigningDuration:                metav1.Duration{Duration: helpers.OneYear},
			ReconcilerSyncLoopPeriod:              metav1.Duration{Duration: 60 * time.Second},
			EnableTaintManager:                    true,
			HorizontalPodAutoscalerUseRESTClients: false,
		},
	}
	s.LeaderElection.LeaderElect = true
	return &s
}

// AddFlags adds flags for a specific CMServer to the specified FlagSet
func (s *CMServer) AddFlags(fs *pflag.FlagSet, allControllers []string, disabledByDefaultControllers []string) {
	fs.StringSliceVar(&s.Controllers, "controllers", s.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))
	fs.Int32Var(&s.Port, "port", s.Port, "The port that the controller-manager's http service runs on")
	fs.Var(componentconfig.IPVar{Val: &s.Address}, "address", "The IP address to serve on (set to 0.0.0.0 for all interfaces)")
	fs.BoolVar(&s.UseServiceAccountCredentials, "use-service-account-credentials", s.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.StringVar(&s.CloudProvider, "cloud-provider", s.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.CloudConfigFile, "cloud-config", s.CloudConfigFile, "The path to the cloud provider configuration file.  Empty string for no configuration file.")
	fs.Int32Var(&s.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", s.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentServiceSyncs, "concurrent-service-syncs", s.ConcurrentServiceSyncs, "The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentRCSyncs, "concurrent_rc_syncs", s.ConcurrentRCSyncs, "The number of replication controllers that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentRSSyncs, "concurrent-replicaset-syncs", s.ConcurrentRSSyncs, "The number of replica sets that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", s.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentDeploymentSyncs, "concurrent-deployment-syncs", s.ConcurrentDeploymentSyncs, "The number of deployment objects that are allowed to sync concurrently. Larger number = more responsive deployments, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentNamespaceSyncs, "concurrent-namespace-syncs", s.ConcurrentNamespaceSyncs, "The number of namespace objects that are allowed to sync concurrently. Larger number = more responsive namespace termination, but more CPU (and network) load")
	fs.Int32Var(&s.ConcurrentSATokenSyncs, "concurrent-serviceaccount-token-syncs", s.ConcurrentSATokenSyncs, "The number of service account token objects that are allowed to sync concurrently. Larger number = more responsive token generation, but more CPU (and network) load")
	// TODO(#43388): Remove the following flag 6 months after v1.6.0 is released.
	fs.Int32Var(&s.LookupCacheSizeForRC, "replication-controller-lookup-cache-size", s.LookupCacheSizeForRC, "This flag is deprecated and will be removed in future releases. ReplicationController no longer requires a lookup cache.")
	fs.MarkDeprecated("replication-controller-lookup-cache-size", "This flag is deprecated and will be removed in future releases. ReplicationController no longer requires a lookup cache.")
	// TODO(#43388): Remove the following flag 6 months after v1.6.0 is released.
	fs.Int32Var(&s.LookupCacheSizeForRS, "replicaset-lookup-cache-size", s.LookupCacheSizeForRS, "This flag is deprecated and will be removed in future releases. ReplicaSet no longer requires a lookup cache.")
	fs.MarkDeprecated("replicaset-lookup-cache-size", "This flag is deprecated and will be removed in future releases. ReplicaSet no longer requires a lookup cache.")
	// TODO(#43388): Remove the following flag 6 months after v1.6.0 is released.
	fs.Int32Var(&s.LookupCacheSizeForDaemonSet, "daemonset-lookup-cache-size", s.LookupCacheSizeForDaemonSet, "This flag is deprecated and will be removed in future releases. DaemonSet no longer requires a lookup cache.")
	fs.MarkDeprecated("daemonset-lookup-cache-size", "This flag is deprecated and will be removed in future releases. DaemonSet no longer requires a lookup cache.")
	fs.DurationVar(&s.ServiceSyncPeriod.Duration, "service-sync-period", s.ServiceSyncPeriod.Duration, "The period for syncing services with their external load balancers")
	fs.DurationVar(&s.NodeSyncPeriod.Duration, "node-sync-period", 0, ""+
		"This flag is deprecated and will be removed in future releases. See node-monitor-period for Node health checking or "+
		"route-reconciliation-period for cloud provider's route configuration settings.")
	fs.MarkDeprecated("node-sync-period", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.RouteReconciliationPeriod.Duration, "route-reconciliation-period", s.RouteReconciliationPeriod.Duration, "The period for reconciling routes created for Nodes by cloud provider.")
	fs.DurationVar(&s.ResourceQuotaSyncPeriod.Duration, "resource-quota-sync-period", s.ResourceQuotaSyncPeriod.Duration, "The period for syncing quota usage status in the system")
	fs.DurationVar(&s.NamespaceSyncPeriod.Duration, "namespace-sync-period", s.NamespaceSyncPeriod.Duration, "The period for syncing namespace life-cycle updates")
	fs.DurationVar(&s.PVClaimBinderSyncPeriod.Duration, "pvclaimbinder-sync-period", s.PVClaimBinderSyncPeriod.Duration, "The period for syncing persistent volumes and persistent volume claims")
	fs.DurationVar(&s.MinResyncPeriod.Duration, "min-resync-period", s.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod")
	fs.StringVar(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, "pv-recycler-pod-template-filepath-nfs", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, "The file path to a pod definition used as a template for NFS persistent volume recycling")
	fs.Int32Var(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS, "pv-recycler-minimum-timeout-nfs", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS, "The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod")
	fs.Int32Var(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS, "pv-recycler-increment-timeout-nfs", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS, "the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod")
	fs.StringVar(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, "pv-recycler-pod-template-filepath-hostpath", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, "The file path to a pod definition used as a template for HostPath persistent volume recycling. This is for development and testing only and will not work in a multi-node cluster.")
	fs.Int32Var(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath, "pv-recycler-minimum-timeout-hostpath", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath, "The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.Int32Var(&s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath, "pv-recycler-timeout-increment-hostpath", s.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath, "the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.BoolVar(&s.VolumeConfiguration.EnableHostPathProvisioning, "enable-hostpath-provisioner", s.VolumeConfiguration.EnableHostPathProvisioning, "Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features.  HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.")
	fs.BoolVar(&s.VolumeConfiguration.EnableDynamicProvisioning, "enable-dynamic-provisioning", s.VolumeConfiguration.EnableDynamicProvisioning, "Enable dynamic provisioning for environments that support it.")
	fs.StringVar(&s.VolumeConfiguration.FlexVolumePluginDir, "flex-volume-plugin-dir", s.VolumeConfiguration.FlexVolumePluginDir, "Full path of the directory in which the flex volume plugin should search for additional third party volume plugins.")
	fs.Int32Var(&s.TerminatedPodGCThreshold, "terminated-pod-gc-threshold", s.TerminatedPodGCThreshold, "Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. If <= 0, the terminated pod garbage collector is disabled.")
	fs.DurationVar(&s.HorizontalPodAutoscalerSyncPeriod.Duration, "horizontal-pod-autoscaler-sync-period", s.HorizontalPodAutoscalerSyncPeriod.Duration, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&s.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-upscale-delay", s.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "The period since last upscale, before another upscale can be performed in horizontal pod autoscaler.")
	fs.DurationVar(&s.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-downscale-delay", s.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "The period since last downscale, before another downscale can be performed in horizontal pod autoscaler.")
	fs.DurationVar(&s.DeploymentControllerSyncPeriod.Duration, "deployment-controller-sync-period", s.DeploymentControllerSyncPeriod.Duration, "Period for syncing the deployments.")
	fs.DurationVar(&s.PodEvictionTimeout.Duration, "pod-eviction-timeout", s.PodEvictionTimeout.Duration, "The grace period for deleting pods on failed nodes.")
	fs.Float32Var(&s.DeletingPodsQps, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.MarkDeprecated("deleting-pods-qps", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.DeletingPodsBurst, "deleting-pods-burst", 0, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.MarkDeprecated("deleting-pods-burst", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.RegisterRetryCount, "register-retry-count", s.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.NodeMonitorGracePeriod.Duration, "node-monitor-grace-period", s.NodeMonitorGracePeriod.Duration,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealthy. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status.")
	fs.DurationVar(&s.NodeStartupGracePeriod.Duration, "node-startup-grace-period", s.NodeStartupGracePeriod.Duration,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealthy.")
	fs.DurationVar(&s.NodeMonitorPeriod.Duration, "node-monitor-period", s.NodeMonitorPeriod.Duration,
		"The period for syncing NodeStatus in NodeController.")
	fs.StringVar(&s.ServiceAccountKeyFile, "service-account-private-key-file", s.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.StringVar(&s.ClusterSigningCertFile, "cluster-signing-cert-file", s.ClusterSigningCertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue cluster-scoped certificates")
	fs.StringVar(&s.ClusterSigningKeyFile, "cluster-signing-key-file", s.ClusterSigningKeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign cluster-scoped certificates")
	fs.DurationVar(&s.ClusterSigningDuration.Duration, "experimental-cluster-signing-duration", s.ClusterSigningDuration.Duration, "The length of duration signed certificates will be given.")
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	fs.BoolVar(&s.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&s.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled")
	fs.StringVar(&s.ClusterName, "cluster-name", s.ClusterName, "The instance prefix for the cluster")
	fs.StringVar(&s.ClusterCIDR, "cluster-cidr", s.ClusterCIDR, "CIDR Range for Pods in cluster.")
	fs.StringVar(&s.ServiceCIDR, "service-cluster-ip-range", s.ServiceCIDR, "CIDR Range for Services in cluster.")
	fs.Int32Var(&s.NodeCIDRMaskSize, "node-cidr-mask-size", s.NodeCIDRMaskSize, "Mask size for node cidr in cluster.")
	fs.BoolVar(&s.AllocateNodeCIDRs, "allocate-node-cidrs", false,
		"Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&s.CIDRAllocatorType, "cidr-allocator-type", "RangeAllocator",
		"Type of CIDR allocator to use")
	fs.BoolVar(&s.ConfigureCloudRoutes, "configure-cloud-routes", true, "Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.")
	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
	fs.StringVar(&s.ContentType, "kube-api-content-type", s.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&s.KubeAPIQPS, "kube-api-qps", s.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver")
	fs.Int32Var(&s.KubeAPIBurst, "kube-api-burst", s.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver")
	fs.DurationVar(&s.ControllerStartInterval.Duration, "controller-start-interval", s.ControllerStartInterval.Duration, "Interval between starting controller managers.")
	fs.BoolVar(&s.EnableGarbageCollector, "enable-garbage-collector", s.EnableGarbageCollector, "Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-apiserver.")
	fs.Int32Var(&s.ConcurrentGCSyncs, "concurrent-gc-syncs", s.ConcurrentGCSyncs, "The number of garbage collector workers that are allowed to sync concurrently.")
	fs.Float32Var(&s.NodeEvictionRate, "node-eviction-rate", 0.1, "Number of nodes per second on which pods are deleted in case of node failure when a zone is healthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters.")
	fs.Float32Var(&s.SecondaryNodeEvictionRate, "secondary-node-eviction-rate", 0.01, "Number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters. This value is implicitly overridden to 0 if the cluster size is smaller than --large-cluster-size-threshold.")
	fs.Int32Var(&s.LargeClusterSizeThreshold, "large-cluster-size-threshold", 50, "Number of nodes from which NodeController treats the cluster as large for the eviction logic purposes. --secondary-node-eviction-rate is implicitly overridden to 0 for clusters this size or smaller.")
	fs.Float32Var(&s.UnhealthyZoneThreshold, "unhealthy-zone-threshold", 0.55, "Fraction of Nodes in a zone which needs to be not Ready (minimum 3) for zone to be treated as unhealthy. ")
	fs.BoolVar(&s.DisableAttachDetachReconcilerSync, "disable-attach-detach-reconcile-sync", false, "Disable volume attach detach reconciler sync. Disabling this may cause volumes to be mismatched with pods. Use wisely.")
	fs.DurationVar(&s.ReconcilerSyncLoopPeriod.Duration, "attach-detach-reconcile-sync-period", s.ReconcilerSyncLoopPeriod.Duration, "The reconciler sync wait time between volume attach detach. This duration must be larger than one second, and increasing this value from the default may allow for volumes to be mismatched with pods.")
	fs.BoolVar(&s.EnableTaintManager, "enable-taint-manager", s.EnableTaintManager, "WARNING: Beta feature. If set to true enables NoExecute Taints and will evict all not-tolerating Pod running on Nodes tainted with this kind of Taints.")
	fs.BoolVar(&s.HorizontalPodAutoscalerUseRESTClients, "horizontal-pod-autoscaler-use-rest-clients", s.HorizontalPodAutoscalerUseRESTClients, "WARNING: alpha feature.  If set to true, causes the horizontal pod autoscaler controller to use REST clients through the kube-aggregator, instead of using the legacy metrics client through the API server proxy.  This is required for custom metrics support in the horizonal pod autoscaler.")

	leaderelection.BindFlags(&s.LeaderElection, fs)

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// Validate is used to validate the options and config before launching the controller manager
func (s *CMServer) Validate(allControllers []string, disabledByDefaultControllers []string) error {
	var errs []error

	allControllersSet := sets.NewString(allControllers...)
	for _, controller := range s.Controllers {
		if controller == "*" {
			continue
		}
		if strings.HasPrefix(controller, "-") {
			controller = controller[1:]
		}

		if !allControllersSet.Has(controller) {
			errs = append(errs, fmt.Errorf("%q is not in the list of known controllers", controller))
		}
	}

	return utilerrors.NewAggregate(errs)
}
