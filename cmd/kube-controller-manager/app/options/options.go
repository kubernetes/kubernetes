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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// KubeControllerManagerOptions is the main context object for the controller manager.
type KubeControllerManagerOptions struct {
	Generic cmoptions.GenericControllerManagerOptions
}

// NewKubeControllerManagerOptions creates a new KubeControllerManagerOptions with a default config.
func NewKubeControllerManagerOptions() *KubeControllerManagerOptions {
	componentConfig := cmoptions.NewDefaultControllerManagerComponentConfig(ports.InsecureKubeControllerManagerPort)
	s := KubeControllerManagerOptions{
		// The common/default are kept in 'cmd/kube-controller-manager/app/options/util.go'.
		// Please make common changes there but put anything kube-controller specific here.
		Generic: cmoptions.NewGenericControllerManagerOptions(componentConfig),
	}

	s.Generic.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.Generic.SecureServing.ServerCert.PairName = "kube-controller-manager"

	gcIgnoredResources := make([]componentconfig.GroupResource, 0, len(garbagecollector.DefaultIgnoredResources()))
	for r := range garbagecollector.DefaultIgnoredResources() {
		gcIgnoredResources = append(gcIgnoredResources, componentconfig.GroupResource{Group: r.Group, Resource: r.Resource})
	}
	s.Generic.ComponentConfig.GCIgnoredResources = gcIgnoredResources
	s.Generic.ComponentConfig.LeaderElection.LeaderElect = true

	return &s
}

// AddFlags adds flags for a specific KubeControllerManagerOptions to the specified FlagSet
func (s *KubeControllerManagerOptions) AddFlags(fs *pflag.FlagSet, allControllers []string, disabledByDefaultControllers []string) {
	s.Generic.AddFlags(fs)

	fs.StringSliceVar(&s.Generic.ComponentConfig.Controllers, "controllers", s.Generic.ComponentConfig.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))
	fs.StringVar(&s.Generic.ComponentConfig.CloudProvider, "cloud-provider", s.Generic.ComponentConfig.CloudProvider, "The provider for cloud services.  Empty string for no provider.")
	fs.StringVar(&s.Generic.ComponentConfig.ExternalCloudVolumePlugin, "external-cloud-volume-plugin", s.Generic.ComponentConfig.ExternalCloudVolumePlugin, "The plugin to use when cloud provider is set to external. Can be empty, should only be set when cloud-provider is external. Currently used to allow node and volume controllers to work for in tree cloud providers.")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentEndpointSyncs, "concurrent-endpoint-syncs", s.Generic.ComponentConfig.ConcurrentEndpointSyncs, "The number of endpoint syncing operations that will be done concurrently. Larger number = faster endpoint updating, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentServiceSyncs, "concurrent-service-syncs", s.Generic.ComponentConfig.ConcurrentServiceSyncs, "The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentRCSyncs, "concurrent_rc_syncs", s.Generic.ComponentConfig.ConcurrentRCSyncs, "The number of replication controllers that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentRSSyncs, "concurrent-replicaset-syncs", s.Generic.ComponentConfig.ConcurrentRSSyncs, "The number of replica sets that are allowed to sync concurrently. Larger number = more responsive replica management, but more CPU (and network) load")

	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", s.Generic.ComponentConfig.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentDeploymentSyncs, "concurrent-deployment-syncs", s.Generic.ComponentConfig.ConcurrentDeploymentSyncs, "The number of deployment objects that are allowed to sync concurrently. Larger number = more responsive deployments, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentNamespaceSyncs, "concurrent-namespace-syncs", s.Generic.ComponentConfig.ConcurrentNamespaceSyncs, "The number of namespace objects that are allowed to sync concurrently. Larger number = more responsive namespace termination, but more CPU (and network) load")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentSATokenSyncs, "concurrent-serviceaccount-token-syncs", s.Generic.ComponentConfig.ConcurrentSATokenSyncs, "The number of service account token objects that are allowed to sync concurrently. Larger number = more responsive token generation, but more CPU (and network) load")
	fs.DurationVar(&s.Generic.ComponentConfig.NodeSyncPeriod.Duration, "node-sync-period", 0, ""+
		"This flag is deprecated and will be removed in future releases. See node-monitor-period for Node health checking or "+
		"route-reconciliation-period for cloud provider's route configuration settings.")
	fs.MarkDeprecated("node-sync-period", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.Generic.ComponentConfig.ResourceQuotaSyncPeriod.Duration, "resource-quota-sync-period", s.Generic.ComponentConfig.ResourceQuotaSyncPeriod.Duration, "The period for syncing quota usage status in the system")
	fs.DurationVar(&s.Generic.ComponentConfig.NamespaceSyncPeriod.Duration, "namespace-sync-period", s.Generic.ComponentConfig.NamespaceSyncPeriod.Duration, "The period for syncing namespace life-cycle updates")
	fs.DurationVar(&s.Generic.ComponentConfig.PVClaimBinderSyncPeriod.Duration, "pvclaimbinder-sync-period", s.Generic.ComponentConfig.PVClaimBinderSyncPeriod.Duration, "The period for syncing persistent volumes and persistent volume claims")
	fs.StringVar(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, "pv-recycler-pod-template-filepath-nfs", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, "The file path to a pod definition used as a template for NFS persistent volume recycling")
	fs.Int32Var(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS, "pv-recycler-minimum-timeout-nfs", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS, "The minimum ActiveDeadlineSeconds to use for an NFS Recycler pod")
	fs.Int32Var(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS, "pv-recycler-increment-timeout-nfs", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS, "the increment of time added per Gi to ActiveDeadlineSeconds for an NFS scrubber pod")
	fs.StringVar(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, "pv-recycler-pod-template-filepath-hostpath", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, "The file path to a pod definition used as a template for HostPath persistent volume recycling. This is for development and testing only and will not work in a multi-node cluster.")
	fs.Int32Var(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath, "pv-recycler-minimum-timeout-hostpath", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath, "The minimum ActiveDeadlineSeconds to use for a HostPath Recycler pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.Int32Var(&s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath, "pv-recycler-timeout-increment-hostpath", s.Generic.ComponentConfig.VolumeConfiguration.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath, "the increment of time added per Gi to ActiveDeadlineSeconds for a HostPath scrubber pod.  This is for development and testing only and will not work in a multi-node cluster.")
	fs.BoolVar(&s.Generic.ComponentConfig.VolumeConfiguration.EnableHostPathProvisioning, "enable-hostpath-provisioner", s.Generic.ComponentConfig.VolumeConfiguration.EnableHostPathProvisioning, "Enable HostPath PV provisioning when running without a cloud provider. This allows testing and development of provisioning features.  HostPath provisioning is not supported in any way, won't work in a multi-node cluster, and should not be used for anything other than testing or development.")
	fs.BoolVar(&s.Generic.ComponentConfig.VolumeConfiguration.EnableDynamicProvisioning, "enable-dynamic-provisioning", s.Generic.ComponentConfig.VolumeConfiguration.EnableDynamicProvisioning, "Enable dynamic provisioning for environments that support it.")
	fs.StringVar(&s.Generic.ComponentConfig.VolumeConfiguration.FlexVolumePluginDir, "flex-volume-plugin-dir", s.Generic.ComponentConfig.VolumeConfiguration.FlexVolumePluginDir, "Full path of the directory in which the flex volume plugin should search for additional third party volume plugins.")
	fs.Int32Var(&s.Generic.ComponentConfig.TerminatedPodGCThreshold, "terminated-pod-gc-threshold", s.Generic.ComponentConfig.TerminatedPodGCThreshold, "Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. If <= 0, the terminated pod garbage collector is disabled.")
	fs.DurationVar(&s.Generic.ComponentConfig.HorizontalPodAutoscalerSyncPeriod.Duration, "horizontal-pod-autoscaler-sync-period", s.Generic.ComponentConfig.HorizontalPodAutoscalerSyncPeriod.Duration, "The period for syncing the number of pods in horizontal pod autoscaler.")
	fs.DurationVar(&s.Generic.ComponentConfig.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-upscale-delay", s.Generic.ComponentConfig.HorizontalPodAutoscalerUpscaleForbiddenWindow.Duration, "The period since last upscale, before another upscale can be performed in horizontal pod autoscaler.")
	fs.DurationVar(&s.Generic.ComponentConfig.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "horizontal-pod-autoscaler-downscale-delay", s.Generic.ComponentConfig.HorizontalPodAutoscalerDownscaleForbiddenWindow.Duration, "The period since last downscale, before another downscale can be performed in horizontal pod autoscaler.")
	fs.Float64Var(&s.Generic.ComponentConfig.HorizontalPodAutoscalerTolerance, "horizontal-pod-autoscaler-tolerance", s.Generic.ComponentConfig.HorizontalPodAutoscalerTolerance, "The minimum change (from 1.0) in the desired-to-actual metrics ratio for the horizontal pod autoscaler to consider scaling.")
	fs.DurationVar(&s.Generic.ComponentConfig.DeploymentControllerSyncPeriod.Duration, "deployment-controller-sync-period", s.Generic.ComponentConfig.DeploymentControllerSyncPeriod.Duration, "Period for syncing the deployments.")
	fs.DurationVar(&s.Generic.ComponentConfig.PodEvictionTimeout.Duration, "pod-eviction-timeout", s.Generic.ComponentConfig.PodEvictionTimeout.Duration, "The grace period for deleting pods on failed nodes.")
	fs.Float32Var(&s.Generic.ComponentConfig.DeletingPodsQps, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.MarkDeprecated("deleting-pods-qps", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.Generic.ComponentConfig.DeletingPodsBurst, "deleting-pods-burst", 0, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.MarkDeprecated("deleting-pods-burst", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.Generic.ComponentConfig.RegisterRetryCount, "register-retry-count", s.Generic.ComponentConfig.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.Generic.ComponentConfig.NodeMonitorGracePeriod.Duration, "node-monitor-grace-period", s.Generic.ComponentConfig.NodeMonitorGracePeriod.Duration,
		"Amount of time which we allow running Node to be unresponsive before marking it unhealthy. "+
			"Must be N times more than kubelet's nodeStatusUpdateFrequency, "+
			"where N means number of retries allowed for kubelet to post node status.")
	fs.DurationVar(&s.Generic.ComponentConfig.NodeStartupGracePeriod.Duration, "node-startup-grace-period", s.Generic.ComponentConfig.NodeStartupGracePeriod.Duration,
		"Amount of time which we allow starting Node to be unresponsive before marking it unhealthy.")
	fs.StringVar(&s.Generic.ComponentConfig.ServiceAccountKeyFile, "service-account-private-key-file", s.Generic.ComponentConfig.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.StringVar(&s.Generic.ComponentConfig.ClusterSigningCertFile, "cluster-signing-cert-file", s.Generic.ComponentConfig.ClusterSigningCertFile, "Filename containing a PEM-encoded X509 CA certificate used to issue cluster-scoped certificates")
	fs.StringVar(&s.Generic.ComponentConfig.ClusterSigningKeyFile, "cluster-signing-key-file", s.Generic.ComponentConfig.ClusterSigningKeyFile, "Filename containing a PEM-encoded RSA or ECDSA private key used to sign cluster-scoped certificates")
	fs.DurationVar(&s.Generic.ComponentConfig.ClusterSigningDuration.Duration, "experimental-cluster-signing-duration", s.Generic.ComponentConfig.ClusterSigningDuration.Duration, "The length of duration signed certificates will be given.")
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	fs.StringVar(&s.Generic.ComponentConfig.ServiceCIDR, "service-cluster-ip-range", s.Generic.ComponentConfig.ServiceCIDR, "CIDR Range for Services in cluster. Requires --allocate-node-cidrs to be true")
	fs.Int32Var(&s.Generic.ComponentConfig.NodeCIDRMaskSize, "node-cidr-mask-size", s.Generic.ComponentConfig.NodeCIDRMaskSize, "Mask size for node cidr in cluster.")
	fs.StringVar(&s.Generic.ComponentConfig.RootCAFile, "root-ca-file", s.Generic.ComponentConfig.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")
	fs.BoolVar(&s.Generic.ComponentConfig.EnableGarbageCollector, "enable-garbage-collector", s.Generic.ComponentConfig.EnableGarbageCollector, "Enables the generic garbage collector. MUST be synced with the corresponding flag of the kube-apiserver.")
	fs.Int32Var(&s.Generic.ComponentConfig.ConcurrentGCSyncs, "concurrent-gc-syncs", s.Generic.ComponentConfig.ConcurrentGCSyncs, "The number of garbage collector workers that are allowed to sync concurrently.")
	fs.Int32Var(&s.Generic.ComponentConfig.LargeClusterSizeThreshold, "large-cluster-size-threshold", 50, "Number of nodes from which NodeController treats the cluster as large for the eviction logic purposes. --secondary-node-eviction-rate is implicitly overridden to 0 for clusters this size or smaller.")
	fs.Float32Var(&s.Generic.ComponentConfig.UnhealthyZoneThreshold, "unhealthy-zone-threshold", 0.55, "Fraction of Nodes in a zone which needs to be not Ready (minimum 3) for zone to be treated as unhealthy. ")
	fs.BoolVar(&s.Generic.ComponentConfig.DisableAttachDetachReconcilerSync, "disable-attach-detach-reconcile-sync", false, "Disable volume attach detach reconciler sync. Disabling this may cause volumes to be mismatched with pods. Use wisely.")
	fs.DurationVar(&s.Generic.ComponentConfig.ReconcilerSyncLoopPeriod.Duration, "attach-detach-reconcile-sync-period", s.Generic.ComponentConfig.ReconcilerSyncLoopPeriod.Duration, "The reconciler sync wait time between volume attach detach. This duration must be larger than one second, and increasing this value from the default may allow for volumes to be mismatched with pods.")
	fs.BoolVar(&s.Generic.ComponentConfig.EnableTaintManager, "enable-taint-manager", s.Generic.ComponentConfig.EnableTaintManager, "WARNING: Beta feature. If set to true enables NoExecute Taints and will evict all not-tolerating Pod running on Nodes tainted with this kind of Taints.")
	fs.BoolVar(&s.Generic.ComponentConfig.HorizontalPodAutoscalerUseRESTClients, "horizontal-pod-autoscaler-use-rest-clients", s.Generic.ComponentConfig.HorizontalPodAutoscalerUseRESTClients, "WARNING: alpha feature.  If set to true, causes the horizontal pod autoscaler controller to use REST clients through the kube-aggregator, instead of using the legacy metrics client through the API server proxy.  This is required for custom metrics support in the horizontal pod autoscaler.")
	fs.Float32Var(&s.Generic.ComponentConfig.NodeEvictionRate, "node-eviction-rate", 0.1, "Number of nodes per second on which pods are deleted in case of node failure when a zone is healthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters.")
	fs.Float32Var(&s.Generic.ComponentConfig.SecondaryNodeEvictionRate, "secondary-node-eviction-rate", 0.01, "Number of nodes per second on which pods are deleted in case of node failure when a zone is unhealthy (see --unhealthy-zone-threshold for definition of healthy/unhealthy). Zone refers to entire cluster in non-multizone clusters. This value is implicitly overridden to 0 if the cluster size is smaller than --large-cluster-size-threshold.")

	leaderelectionconfig.BindFlags(&s.Generic.ComponentConfig.LeaderElection, fs)

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config) error {
	err := s.Generic.ApplyTo(&c.Generic, "controller-manager")

	return err
}

// Validate is used to validate the options and config before launching the controller manager
func (s *KubeControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string) error {
	var errs []error

	allControllersSet := sets.NewString(allControllers...)
	for _, controller := range s.Generic.ComponentConfig.Controllers {
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

// Config return a controller manager config objective
func (s KubeControllerManagerOptions) Config(allControllers []string, disabledByDefaultControllers []string) (*kubecontrollerconfig.Config, error) {
	if err := s.Validate(allControllers, disabledByDefaultControllers); err != nil {
		return nil, err
	}

	c := &kubecontrollerconfig.Config{}
	if err := s.ApplyTo(c); err != nil {
		return nil, err
	}

	return c, nil
}
