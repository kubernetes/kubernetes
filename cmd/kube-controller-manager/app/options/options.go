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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// KubeControllerManagerOptions is the main context object for the controller manager.
type KubeControllerManagerOptions struct {
	GenericControllerManagerOptions  *cmoptions.GenericControllerManagerOptions
	PersistentVolumeBinderController *PersistentVolumeBinderControllerOptions
	HPAController                    *HPAControllerOptions
	NamespaceController              *NamespaceControllerOptions
	NodeLifecycleController          *NodeLifecycleControllerOptions
	CSRSigningController             *CSRSigningControllerOptions
	AttachDetachController           *AttachDetachControllerOptions
	PodGCController                  *PodGCControllerOptions
	ResourceQuotaController          *ResourceQuotaControllerOptions
	GarbageCollectorController       *GarbageCollectorControllerOptions
	ConcurrentResourcesSyncs         *ConcurrentResourcesSyncsOptions

	Controllers                    []string
	ExternalCloudVolumePlugin      string
	NodeSyncPeriod                 metav1.Duration
	DeploymentControllerSyncPeriod metav1.Duration
	DeletingPodsQPS                float32
	DeletingPodsBurst              int32
	RegisterRetryCount             int32
	ServiceCIDR                    string
	NodeCIDRMaskSize               int32
	RootCAFile                     string
}

// NewKubeControllerManagerOptions creates a new KubeControllerManagerOptions with a default config.
func NewKubeControllerManagerOptions() *KubeControllerManagerOptions {
	componentConfig := cmoptions.NewDefaultControllerManagerComponentConfig(ports.InsecureKubeControllerManagerPort)
	s := KubeControllerManagerOptions{
		// The common/default are kept in 'cmd/kube-controller-manager/app/options/util.go'.
		// Please make common changes there but put anything kube-controller specific here.
		GenericControllerManagerOptions: cmoptions.NewGenericControllerManagerOptions(componentConfig),
		PersistentVolumeBinderController: &PersistentVolumeBinderControllerOptions{
			PVClaimBinderSyncPeriod: componentConfig.PVClaimBinderSyncPeriod,
			VolumeConfiguration:     componentConfig.VolumeConfiguration,
		},
		HPAController: &HPAControllerOptions{
			HorizontalPodAutoscalerSyncPeriod:               componentConfig.HorizontalPodAutoscalerSyncPeriod,
			HorizontalPodAutoscalerUpscaleForbiddenWindow:   componentConfig.HorizontalPodAutoscalerUpscaleForbiddenWindow,
			HorizontalPodAutoscalerDownscaleForbiddenWindow: componentConfig.HorizontalPodAutoscalerDownscaleForbiddenWindow,
			HorizontalPodAutoscalerTolerance:                componentConfig.HorizontalPodAutoscalerTolerance,
			HorizontalPodAutoscalerUseRESTClients:           componentConfig.HorizontalPodAutoscalerUseRESTClients,
		},
		NamespaceController: &NamespaceControllerOptions{
			NamespaceSyncPeriod: componentConfig.NamespaceSyncPeriod,
		},
		NodeLifecycleController: &NodeLifecycleControllerOptions{
			EnableTaintManager:        componentConfig.EnableTaintManager,
			NodeEvictionRate:          componentConfig.NodeEvictionRate,
			SecondaryNodeEvictionRate: componentConfig.SecondaryNodeEvictionRate,
			NodeMonitorGracePeriod:    componentConfig.NodeMonitorGracePeriod,
			NodeStartupGracePeriod:    componentConfig.NodeStartupGracePeriod,
			PodEvictionTimeout:        componentConfig.PodEvictionTimeout,
			LargeClusterSizeThreshold: componentConfig.LargeClusterSizeThreshold,
			UnhealthyZoneThreshold:    componentConfig.UnhealthyZoneThreshold,
		},
		CSRSigningController: &CSRSigningControllerOptions{
			ClusterSigningCertFile: componentConfig.ClusterSigningCertFile,
			ClusterSigningKeyFile:  componentConfig.ClusterSigningKeyFile,
			ClusterSigningDuration: componentConfig.ClusterSigningDuration,
		},
		AttachDetachController: &AttachDetachControllerOptions{
			ReconcilerSyncLoopPeriod:          componentConfig.ReconcilerSyncLoopPeriod,
			DisableAttachDetachReconcilerSync: componentConfig.DisableAttachDetachReconcilerSync,
		},
		PodGCController: &PodGCControllerOptions{
			TerminatedPodGCThreshold: componentConfig.TerminatedPodGCThreshold,
		},
		ResourceQuotaController: &ResourceQuotaControllerOptions{
			ResourceQuotaSyncPeriod: componentConfig.ResourceQuotaSyncPeriod,
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			ConcurrentGCSyncs:      componentConfig.ConcurrentGCSyncs,
			GCIgnoredResources:     componentConfig.GCIgnoredResources,
			EnableGarbageCollector: componentConfig.EnableGarbageCollector,
		},
		ConcurrentResourcesSyncs: &ConcurrentResourcesSyncsOptions{
			ConcurrentDeploymentSyncs:    componentConfig.ConcurrentDeploymentSyncs,
			ConcurrentEndpointSyncs:      componentConfig.ConcurrentEndpointSyncs,
			ConcurrentNamespaceSyncs:     componentConfig.ConcurrentNamespaceSyncs,
			ConcurrentRSSyncs:            componentConfig.ConcurrentRSSyncs,
			ConcurrentResourceQuotaSyncs: componentConfig.ConcurrentResourceQuotaSyncs,
			ConcurrentSATokenSyncs:       componentConfig.ConcurrentSATokenSyncs,
			ConcurrentRCSyncs:            componentConfig.ConcurrentRCSyncs,
			ConcurrentDaemonSetSyncs:     componentConfig.ConcurrentDaemonSetSyncs,
			ConcurrentJobSyncs:           componentConfig.ConcurrentJobSyncs,
		},
		Controllers:                    componentConfig.Controllers,
		DeploymentControllerSyncPeriod: componentConfig.DeploymentControllerSyncPeriod,
		DeletingPodsQPS:                componentConfig.DeletingPodsQps,
		RegisterRetryCount:             componentConfig.RegisterRetryCount,
		NodeCIDRMaskSize:               componentConfig.NodeCIDRMaskSize,
	}

	s.GenericControllerManagerOptions.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.GenericControllerManagerOptions.SecureServing.ServerCert.PairName = "kube-controller-manager"

	gcIgnoredResources := make([]componentconfig.GroupResource, 0, len(garbagecollector.DefaultIgnoredResources()))
	for r := range garbagecollector.DefaultIgnoredResources() {
		gcIgnoredResources = append(gcIgnoredResources, componentconfig.GroupResource{Group: r.Group, Resource: r.Resource})
	}
	s.GarbageCollectorController.GCIgnoredResources = gcIgnoredResources
	s.GenericControllerManagerOptions.GenericComponentConfig.LeaderElection.LeaderElect = true

	return &s
}

// AddFlags adds flags for a specific KubeControllerManagerOptions to the specified FlagSet
func (s *KubeControllerManagerOptions) AddFlags(fs *pflag.FlagSet, allControllers []string, disabledByDefaultControllers []string) {
	s.GenericControllerManagerOptions.AddFlags(fs)

	s.PersistentVolumeBinderController.AddFlags(fs)
	s.HPAController.AddFlags(fs)
	s.NamespaceController.AddFlags(fs)
	s.NodeLifecycleController.AddFlags(fs)
	s.CSRSigningController.AddFlags(fs)
	s.AttachDetachController.AddFlags(fs)
	s.PodGCController.AddFlags(fs)
	s.ResourceQuotaController.AddFlags(fs)
	s.GarbageCollectorController.AddFlags(fs)
	s.ConcurrentResourcesSyncs.AddFlags(fs)

	fs.StringSliceVar(&s.Controllers, "controllers", s.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))
	fs.StringVar(&s.ExternalCloudVolumePlugin, "external-cloud-volume-plugin", s.ExternalCloudVolumePlugin, "The plugin to use when cloud provider is set to external. Can be empty, should only be set when cloud-provider is external. Currently used to allow node and volume controllers to work for in tree cloud providers.")
	fs.DurationVar(&s.NodeSyncPeriod.Duration, "node-sync-period", 0, ""+
		"This flag is deprecated and will be removed in future releases. See node-monitor-period for Node health checking or "+
		"route-reconciliation-period for cloud provider's route configuration settings.")
	fs.MarkDeprecated("node-sync-period", "This flag is currently no-op and will be deleted.")
	fs.DurationVar(&s.DeploymentControllerSyncPeriod.Duration, "deployment-controller-sync-period", s.DeploymentControllerSyncPeriod.Duration, "Period for syncing the deployments.")
	fs.Float32Var(&s.DeletingPodsQPS, "deleting-pods-qps", 0.1, "Number of nodes per second on which pods are deleted in case of node failure.")
	fs.MarkDeprecated("deleting-pods-qps", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.DeletingPodsBurst, "deleting-pods-burst", 0, "Number of nodes on which pods are bursty deleted in case of node failure. For more details look into RateLimiter.")
	fs.MarkDeprecated("deleting-pods-burst", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&s.RegisterRetryCount, "register-retry-count", s.RegisterRetryCount, ""+
		"The number of retries for initial node registration.  Retry interval equals node-sync-period.")
	fs.MarkDeprecated("register-retry-count", "This flag is currently no-op and will be deleted.")
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	fs.StringVar(&s.ServiceCIDR, "service-cluster-ip-range", s.ServiceCIDR, "CIDR Range for Services in cluster. Requires --allocate-node-cidrs to be true")
	fs.Int32Var(&s.NodeCIDRMaskSize, "node-cidr-mask-size", s.NodeCIDRMaskSize, "Mask size for node cidr in cluster.")
	fs.StringVar(&s.RootCAFile, "root-ca-file", s.RootCAFile, "If set, this root certificate authority will be included in service account's token secret. This must be a valid PEM-encoded CA bundle.")

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config) error {
	if err := s.GenericControllerManagerOptions.ApplyTo(&c.Generic, "controller-manager"); err != nil {
		return err
	}
	if err := s.PersistentVolumeBinderController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.HPAController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.NamespaceController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.NodeLifecycleController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.CSRSigningController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.AttachDetachController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.PodGCController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.ResourceQuotaController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.GarbageCollectorController.ApplyTo(&c.Generic); err != nil {
		return err
	}
	if err := s.ConcurrentResourcesSyncs.ApplyTo(&c.Generic); err != nil {
		return err
	}

	c.Generic.ComponentConfig.Controllers = s.Controllers
	c.Generic.ComponentConfig.ExternalCloudVolumePlugin = s.ExternalCloudVolumePlugin
	c.Generic.ComponentConfig.NodeSyncPeriod = s.NodeSyncPeriod
	c.Generic.ComponentConfig.DeploymentControllerSyncPeriod = s.DeploymentControllerSyncPeriod
	c.Generic.ComponentConfig.DeletingPodsQPS = s.DeletingPodsQPS
	c.Generic.ComponentConfig.DeletingPodsBurst = s.DeletingPodsBurst
	c.Generic.ComponentConfig.RegisterRetryCount = s.RegisterRetryCount
	c.Generic.ComponentConfig.ServiceCIDR = s.ServiceCIDR
	c.Generic.ComponentConfig.NodeCIDRMaskSize = s.NodeCIDRMaskSize
	c.Generic.ComponentConfig.RootCAFile = s.RootCAFile

	return nil
}

// Validate is used to validate the options and config before launching the controller manager
func (s *KubeControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string) error {
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

	errs = append(errs, s.GenericControllerManagerOptions.Validate()...)
	errs = append(errs, s.PersistentVolumeBinderController.Validate()...)
	errs = append(errs, s.HPAController.Validate()...)
	errs = append(errs, s.NamespaceController.Validate()...)
	errs = append(errs, s.NodeLifecycleController.Validate()...)
	errs = append(errs, s.CSRSigningController.Validate()...)
	errs = append(errs, s.AttachDetachController.Validate()...)
	errs = append(errs, s.PodGCController.Validate()...)
	errs = append(errs, s.ResourceQuotaController.Validate()...)
	errs = append(errs, s.GarbageCollectorController.Validate()...)
	errs = append(errs, s.ConcurrentResourcesSyncs.Validate()...)

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
