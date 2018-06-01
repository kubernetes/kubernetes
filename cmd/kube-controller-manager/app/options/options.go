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
	"net"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	kubecontrollerconfig "k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master/ports"
	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

// KubeControllerManagerOptions is the main context object for the kube-controller manager.
type KubeControllerManagerOptions struct {
	CloudProvider    *cmoptions.CloudProviderOptions
	Debugging        *cmoptions.DebuggingOptions
	GenericComponent *cmoptions.GenericComponentConfigOptions
	KubeCloudShared  *cmoptions.KubeCloudSharedOptions

	AttachDetachController           *cmoptions.AttachDetachControllerOptions
	CSRSigningController             *cmoptions.CSRSigningControllerOptions
	DaemonSetController              *cmoptions.DaemonSetControllerOptions
	DeploymentController             *cmoptions.DeploymentControllerOptions
	DeprecatedFlags                  *cmoptions.DeprecatedControllerOptions
	EndPointController               *cmoptions.EndPointControllerOptions
	GarbageCollectorController       *cmoptions.GarbageCollectorControllerOptions
	HPAController                    *cmoptions.HPAControllerOptions
	JobController                    *cmoptions.JobControllerOptions
	NamespaceController              *cmoptions.NamespaceControllerOptions
	NodeIpamController               *cmoptions.NodeIpamControllerOptions
	NodeLifecycleController          *cmoptions.NodeLifecycleControllerOptions
	PersistentVolumeBinderController *cmoptions.PersistentVolumeBinderControllerOptions
	PodGCController                  *cmoptions.PodGCControllerOptions
	ReplicaSetController             *cmoptions.ReplicaSetControllerOptions
	ReplicationController            *cmoptions.ReplicationControllerOptions
	ResourceQuotaController          *cmoptions.ResourceQuotaControllerOptions
	SAController                     *cmoptions.SAControllerOptions
	ServiceController                *cmoptions.ServiceControllerOptions

	Controllers               []string
	ExternalCloudVolumePlugin string

	SecureServing *apiserveroptions.SecureServingOptions
	// TODO: remove insecure serving mode
	InsecureServing *cmoptions.InsecureServingOptions
	Authentication  *apiserveroptions.DelegatingAuthenticationOptions
	Authorization   *apiserveroptions.DelegatingAuthorizationOptions

	Master     string
	Kubeconfig string
}

// NewKubeControllerManagerOptions creates a new KubeControllerManagerOptions with a default config.
func NewKubeControllerManagerOptions() (*KubeControllerManagerOptions, error) {
	componentConfig, err := NewDefaultComponentConfig(ports.InsecureKubeControllerManagerPort)
	if err != nil {
		return nil, err
	}

	s := KubeControllerManagerOptions{
		CloudProvider:    &cmoptions.CloudProviderOptions{},
		Debugging:        &cmoptions.DebuggingOptions{},
		GenericComponent: cmoptions.NewGenericComponentConfigOptions(componentConfig.GenericComponent),
		KubeCloudShared:  cmoptions.NewKubeCloudSharedOptions(componentConfig.KubeCloudShared),
		AttachDetachController: &cmoptions.AttachDetachControllerOptions{
			ReconcilerSyncLoopPeriod: componentConfig.AttachDetachController.ReconcilerSyncLoopPeriod,
		},
		CSRSigningController: &cmoptions.CSRSigningControllerOptions{
			ClusterSigningCertFile: componentConfig.CSRSigningController.ClusterSigningCertFile,
			ClusterSigningKeyFile:  componentConfig.CSRSigningController.ClusterSigningKeyFile,
			ClusterSigningDuration: componentConfig.CSRSigningController.ClusterSigningDuration,
		},
		DaemonSetController: &cmoptions.DaemonSetControllerOptions{
			ConcurrentDaemonSetSyncs: componentConfig.DaemonSetController.ConcurrentDaemonSetSyncs,
		},
		DeploymentController: &cmoptions.DeploymentControllerOptions{
			ConcurrentDeploymentSyncs:      componentConfig.DeploymentController.ConcurrentDeploymentSyncs,
			DeploymentControllerSyncPeriod: componentConfig.DeploymentController.DeploymentControllerSyncPeriod,
		},
		DeprecatedFlags: &cmoptions.DeprecatedControllerOptions{
			RegisterRetryCount: componentConfig.DeprecatedController.RegisterRetryCount,
		},
		EndPointController: &cmoptions.EndPointControllerOptions{
			ConcurrentEndpointSyncs: componentConfig.EndPointController.ConcurrentEndpointSyncs,
		},
		GarbageCollectorController: &cmoptions.GarbageCollectorControllerOptions{
			ConcurrentGCSyncs:      componentConfig.GarbageCollectorController.ConcurrentGCSyncs,
			EnableGarbageCollector: componentConfig.GarbageCollectorController.EnableGarbageCollector,
		},
		HPAController: &cmoptions.HPAControllerOptions{
			HorizontalPodAutoscalerSyncPeriod:               componentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod,
			HorizontalPodAutoscalerUpscaleForbiddenWindow:   componentConfig.HPAController.HorizontalPodAutoscalerUpscaleForbiddenWindow,
			HorizontalPodAutoscalerDownscaleForbiddenWindow: componentConfig.HPAController.HorizontalPodAutoscalerDownscaleForbiddenWindow,
			HorizontalPodAutoscalerTolerance:                componentConfig.HPAController.HorizontalPodAutoscalerTolerance,
			HorizontalPodAutoscalerUseRESTClients:           componentConfig.HPAController.HorizontalPodAutoscalerUseRESTClients,
		},
		JobController: &cmoptions.JobControllerOptions{
			ConcurrentJobSyncs: componentConfig.JobController.ConcurrentJobSyncs,
		},
		NamespaceController: &cmoptions.NamespaceControllerOptions{
			NamespaceSyncPeriod:      componentConfig.NamespaceController.NamespaceSyncPeriod,
			ConcurrentNamespaceSyncs: componentConfig.NamespaceController.ConcurrentNamespaceSyncs,
		},
		NodeIpamController: &cmoptions.NodeIpamControllerOptions{
			NodeCIDRMaskSize: componentConfig.NodeIpamController.NodeCIDRMaskSize,
		},
		NodeLifecycleController: &cmoptions.NodeLifecycleControllerOptions{
			EnableTaintManager:     componentConfig.NodeLifecycleController.EnableTaintManager,
			NodeMonitorGracePeriod: componentConfig.NodeLifecycleController.NodeMonitorGracePeriod,
			NodeStartupGracePeriod: componentConfig.NodeLifecycleController.NodeStartupGracePeriod,
			PodEvictionTimeout:     componentConfig.NodeLifecycleController.PodEvictionTimeout,
		},
		PersistentVolumeBinderController: &cmoptions.PersistentVolumeBinderControllerOptions{
			PVClaimBinderSyncPeriod: componentConfig.PersistentVolumeBinderController.PVClaimBinderSyncPeriod,
			VolumeConfiguration:     componentConfig.PersistentVolumeBinderController.VolumeConfiguration,
		},
		PodGCController: &cmoptions.PodGCControllerOptions{
			TerminatedPodGCThreshold: componentConfig.PodGCController.TerminatedPodGCThreshold,
		},
		ReplicaSetController: &cmoptions.ReplicaSetControllerOptions{
			ConcurrentRSSyncs: componentConfig.ReplicaSetController.ConcurrentRSSyncs,
		},
		ReplicationController: &cmoptions.ReplicationControllerOptions{
			ConcurrentRCSyncs: componentConfig.ReplicationController.ConcurrentRCSyncs,
		},
		ResourceQuotaController: &cmoptions.ResourceQuotaControllerOptions{
			ResourceQuotaSyncPeriod:      componentConfig.ResourceQuotaController.ResourceQuotaSyncPeriod,
			ConcurrentResourceQuotaSyncs: componentConfig.ResourceQuotaController.ConcurrentResourceQuotaSyncs,
		},
		SAController: &cmoptions.SAControllerOptions{
			ConcurrentSATokenSyncs: componentConfig.SAController.ConcurrentSATokenSyncs,
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: componentConfig.ServiceController.ConcurrentServiceSyncs,
		},
		Controllers:   componentConfig.Controllers,
		SecureServing: apiserveroptions.NewSecureServingOptions(),
		InsecureServing: &cmoptions.InsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.KubeCloudShared.Address),
			BindPort:    int(componentConfig.KubeCloudShared.Port),
			BindNetwork: "tcp",
		},
		Authentication: nil, // TODO: enable with apiserveroptions.NewDelegatingAuthenticationOptions()
		Authorization:  nil, // TODO: enable with apiserveroptions.NewDelegatingAuthorizationOptions()
	}

	s.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.SecureServing.ServerCert.PairName = "kube-controller-manager"

	// disable secure serving for now
	// TODO: enable HTTPS by default
	s.SecureServing.BindPort = 0

	gcIgnoredResources := make([]componentconfig.GroupResource, 0, len(garbagecollector.DefaultIgnoredResources()))
	for r := range garbagecollector.DefaultIgnoredResources() {
		gcIgnoredResources = append(gcIgnoredResources, componentconfig.GroupResource{Group: r.Group, Resource: r.Resource})
	}

	s.GarbageCollectorController.GCIgnoredResources = gcIgnoredResources

	return &s, nil
}

// NewDefaultComponentConfig returns kube-controller manager configuration object.
func NewDefaultComponentConfig(insecurePort int32) (componentconfig.KubeControllerManagerConfiguration, error) {
	scheme := runtime.NewScheme()
	componentconfigv1alpha1.AddToScheme(scheme)
	componentconfig.AddToScheme(scheme)

	versioned := componentconfigv1alpha1.KubeControllerManagerConfiguration{}
	scheme.Default(&versioned)

	internal := componentconfig.KubeControllerManagerConfiguration{}
	if err := scheme.Convert(&versioned, &internal, nil); err != nil {
		return internal, err
	}
	internal.KubeCloudShared.Port = insecurePort
	return internal, nil
}

// AddFlags adds flags for a specific KubeControllerManagerOptions to the specified FlagSet
func (s *KubeControllerManagerOptions) AddFlags(fs *pflag.FlagSet, allControllers []string, disabledByDefaultControllers []string) {
	s.CloudProvider.AddFlags(fs)
	s.Debugging.AddFlags(fs)
	s.GenericComponent.AddFlags(fs)
	s.KubeCloudShared.AddFlags(fs)
	s.ServiceController.AddFlags(fs)

	s.SecureServing.AddFlags(fs)
	s.InsecureServing.AddFlags(fs)
	s.Authentication.AddFlags(fs)
	s.Authorization.AddFlags(fs)

	s.AttachDetachController.AddFlags(fs)
	s.CSRSigningController.AddFlags(fs)
	s.DeploymentController.AddFlags(fs)
	s.DaemonSetController.AddFlags(fs)
	s.DeprecatedFlags.AddFlags(fs)
	s.EndPointController.AddFlags(fs)
	s.GarbageCollectorController.AddFlags(fs)
	s.HPAController.AddFlags(fs)
	s.JobController.AddFlags(fs)
	s.NamespaceController.AddFlags(fs)
	s.NodeIpamController.AddFlags(fs)
	s.NodeLifecycleController.AddFlags(fs)
	s.PersistentVolumeBinderController.AddFlags(fs)
	s.PodGCController.AddFlags(fs)
	s.ReplicaSetController.AddFlags(fs)
	s.ReplicationController.AddFlags(fs)
	s.ResourceQuotaController.AddFlags(fs)
	s.SAController.AddFlags(fs)

	fs.StringVar(&s.Master, "master", s.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&s.Kubeconfig, "kubeconfig", s.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringSliceVar(&s.Controllers, "controllers", s.Controllers, fmt.Sprintf(""+
		"A list of controllers to enable.  '*' enables all on-by-default controllers, 'foo' enables the controller "+
		"named 'foo', '-foo' disables the controller named 'foo'.\nAll controllers: %s\nDisabled-by-default controllers: %s",
		strings.Join(allControllers, ", "), strings.Join(disabledByDefaultControllers, ", ")))
	fs.StringVar(&s.ExternalCloudVolumePlugin, "external-cloud-volume-plugin", s.ExternalCloudVolumePlugin, "The plugin to use when cloud provider is set to external. Can be empty, should only be set when cloud-provider is external. Currently used to allow node and volume controllers to work for in tree cloud providers.")
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config, userAgent string) error {
	if err := s.CloudProvider.ApplyTo(&c.ComponentConfig.CloudProvider); err != nil {
		return err
	}
	if err := s.Debugging.ApplyTo(&c.ComponentConfig.Debugging); err != nil {
		return err
	}
	if err := s.GenericComponent.ApplyTo(&c.ComponentConfig.GenericComponent); err != nil {
		return err
	}
	if err := s.KubeCloudShared.ApplyTo(&c.ComponentConfig.KubeCloudShared); err != nil {
		return err
	}
	if err := s.AttachDetachController.ApplyTo(&c.ComponentConfig.AttachDetachController); err != nil {
		return err
	}
	if err := s.CSRSigningController.ApplyTo(&c.ComponentConfig.CSRSigningController); err != nil {
		return err
	}
	if err := s.DaemonSetController.ApplyTo(&c.ComponentConfig.DaemonSetController); err != nil {
		return err
	}
	if err := s.DeploymentController.ApplyTo(&c.ComponentConfig.DeploymentController); err != nil {
		return err
	}
	if err := s.DeprecatedFlags.ApplyTo(&c.ComponentConfig.DeprecatedController); err != nil {
		return err
	}
	if err := s.EndPointController.ApplyTo(&c.ComponentConfig.EndPointController); err != nil {
		return err
	}
	if err := s.GarbageCollectorController.ApplyTo(&c.ComponentConfig.GarbageCollectorController); err != nil {
		return err
	}
	if err := s.HPAController.ApplyTo(&c.ComponentConfig.HPAController); err != nil {
		return err
	}
	if err := s.JobController.ApplyTo(&c.ComponentConfig.JobController); err != nil {
		return err
	}
	if err := s.NamespaceController.ApplyTo(&c.ComponentConfig.NamespaceController); err != nil {
		return err
	}
	if err := s.NodeIpamController.ApplyTo(&c.ComponentConfig.NodeIpamController); err != nil {
		return err
	}
	if err := s.NodeLifecycleController.ApplyTo(&c.ComponentConfig.NodeLifecycleController); err != nil {
		return err
	}
	if err := s.PersistentVolumeBinderController.ApplyTo(&c.ComponentConfig.PersistentVolumeBinderController); err != nil {
		return err
	}
	if err := s.PodGCController.ApplyTo(&c.ComponentConfig.PodGCController); err != nil {
		return err
	}
	if err := s.ReplicaSetController.ApplyTo(&c.ComponentConfig.ReplicaSetController); err != nil {
		return err
	}
	if err := s.ReplicationController.ApplyTo(&c.ComponentConfig.ReplicationController); err != nil {
		return err
	}
	if err := s.ResourceQuotaController.ApplyTo(&c.ComponentConfig.ResourceQuotaController); err != nil {
		return err
	}
	if err := s.SAController.ApplyTo(&c.ComponentConfig.SAController); err != nil {
		return err
	}
	if err := s.ServiceController.ApplyTo(&c.ComponentConfig.ServiceController); err != nil {
		return err
	}
	if err := s.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := s.InsecureServing.ApplyTo(&c.InsecureServing); err != nil {
		return err
	}
	if err := s.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
		return err
	}
	if err := s.Authorization.ApplyTo(&c.Authorization); err != nil {
		return err
	}

	// sync back to component config
	// TODO: find more elegant way than synching back the values.
	c.ComponentConfig.KubeCloudShared.Port = int32(s.InsecureServing.BindPort)
	c.ComponentConfig.KubeCloudShared.Address = s.InsecureServing.BindAddress.String()

	var err error
	c.Kubeconfig, err = clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return err
	}
	c.Kubeconfig.ContentConfig.ContentType = s.GenericComponent.ContentType
	c.Kubeconfig.QPS = s.GenericComponent.KubeAPIQPS
	c.Kubeconfig.Burst = int(s.GenericComponent.KubeAPIBurst)

	c.Client, err = clientset.NewForConfig(restclient.AddUserAgent(c.Kubeconfig, userAgent))
	if err != nil {
		return err
	}

	c.LeaderElectionClient = clientset.NewForConfigOrDie(restclient.AddUserAgent(c.Kubeconfig, "leader-election"))

	c.EventRecorder = createRecorder(c.Client, userAgent)

	c.ComponentConfig.Controllers = s.Controllers
	c.ComponentConfig.ExternalCloudVolumePlugin = s.ExternalCloudVolumePlugin

	return err
}

// Validate is used to validate the options and config before launching the controller manager
func (s *KubeControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string) error {
	var errs []error

	errs = append(errs, s.CloudProvider.Validate()...)
	errs = append(errs, s.Debugging.Validate()...)
	errs = append(errs, s.GenericComponent.Validate()...)
	errs = append(errs, s.KubeCloudShared.Validate()...)
	errs = append(errs, s.AttachDetachController.Validate()...)
	errs = append(errs, s.CSRSigningController.Validate()...)
	errs = append(errs, s.DaemonSetController.Validate()...)
	errs = append(errs, s.DeploymentController.Validate()...)
	errs = append(errs, s.DeprecatedFlags.Validate()...)
	errs = append(errs, s.EndPointController.Validate()...)
	errs = append(errs, s.GarbageCollectorController.Validate()...)
	errs = append(errs, s.HPAController.Validate()...)
	errs = append(errs, s.JobController.Validate()...)
	errs = append(errs, s.NamespaceController.Validate()...)
	errs = append(errs, s.NodeIpamController.Validate()...)
	errs = append(errs, s.NodeLifecycleController.Validate()...)
	errs = append(errs, s.PersistentVolumeBinderController.Validate()...)
	errs = append(errs, s.PodGCController.Validate()...)
	errs = append(errs, s.ReplicaSetController.Validate()...)
	errs = append(errs, s.ReplicationController.Validate()...)
	errs = append(errs, s.ResourceQuotaController.Validate()...)
	errs = append(errs, s.SAController.Validate()...)
	errs = append(errs, s.ServiceController.Validate()...)
	errs = append(errs, s.SecureServing.Validate()...)
	errs = append(errs, s.InsecureServing.Validate()...)
	errs = append(errs, s.Authentication.Validate()...)
	errs = append(errs, s.Authorization.Validate()...)

	// TODO: validate component config, master and kubeconfig

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

// Config return a controller manager config objective
func (s KubeControllerManagerOptions) Config(allControllers []string, disabledByDefaultControllers []string) (*kubecontrollerconfig.Config, error) {
	if err := s.Validate(allControllers, disabledByDefaultControllers); err != nil {
		return nil, err
	}

	c := &kubecontrollerconfig.Config{}
	if err := s.ApplyTo(c, "kube-controller-manager"); err != nil {
		return nil, err
	}

	return c, nil
}

func createRecorder(kubeClient kubernetes.Interface, userAgent string) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	return eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: userAgent})
}
