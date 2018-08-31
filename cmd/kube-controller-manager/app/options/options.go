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
	apiserverflag "k8s.io/apiserver/pkg/util/flag"
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
)

const (
	// KubeControllerManagerUserAgent is the userAgent name when starting kube-controller managers.
	KubeControllerManagerUserAgent = "kube-controller-manager"
)

// KubeControllerManagerOptions is the main context object for the kube-controller manager.
type KubeControllerManagerOptions struct {
	CloudProvider     *cmoptions.CloudProviderOptions
	Debugging         *cmoptions.DebuggingOptions
	GenericComponent  *cmoptions.GenericComponentConfigOptions
	KubeCloudShared   *cmoptions.KubeCloudSharedOptions
	ServiceController *cmoptions.ServiceControllerOptions

	AttachDetachController           *AttachDetachControllerOptions
	CSRSigningController             *CSRSigningControllerOptions
	DaemonSetController              *DaemonSetControllerOptions
	DeploymentController             *DeploymentControllerOptions
	DeprecatedFlags                  *DeprecatedControllerOptions
	EndPointController               *EndPointControllerOptions
	GarbageCollectorController       *GarbageCollectorControllerOptions
	HPAController                    *HPAControllerOptions
	JobController                    *JobControllerOptions
	NamespaceController              *NamespaceControllerOptions
	NodeIpamController               *NodeIpamControllerOptions
	NodeLifecycleController          *NodeLifecycleControllerOptions
	PersistentVolumeBinderController *PersistentVolumeBinderControllerOptions
	PodGCController                  *PodGCControllerOptions
	ReplicaSetController             *ReplicaSetControllerOptions
	ReplicationController            *ReplicationControllerOptions
	ResourceQuotaController          *ResourceQuotaControllerOptions
	SAController                     *SAControllerOptions

	Controllers               []string
	ExternalCloudVolumePlugin string

	SecureServing *apiserveroptions.SecureServingOptionsWithLoopback
	// TODO: remove insecure serving mode
	InsecureServing *apiserveroptions.DeprecatedInsecureServingOptionsWithLoopback
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
		AttachDetachController: &AttachDetachControllerOptions{
			ReconcilerSyncLoopPeriod: componentConfig.AttachDetachController.ReconcilerSyncLoopPeriod,
		},
		CSRSigningController: &CSRSigningControllerOptions{
			ClusterSigningCertFile: componentConfig.CSRSigningController.ClusterSigningCertFile,
			ClusterSigningKeyFile:  componentConfig.CSRSigningController.ClusterSigningKeyFile,
			ClusterSigningDuration: componentConfig.CSRSigningController.ClusterSigningDuration,
		},
		DaemonSetController: &DaemonSetControllerOptions{
			ConcurrentDaemonSetSyncs: componentConfig.DaemonSetController.ConcurrentDaemonSetSyncs,
		},
		DeploymentController: &DeploymentControllerOptions{
			ConcurrentDeploymentSyncs:      componentConfig.DeploymentController.ConcurrentDeploymentSyncs,
			DeploymentControllerSyncPeriod: componentConfig.DeploymentController.DeploymentControllerSyncPeriod,
		},
		DeprecatedFlags: &DeprecatedControllerOptions{
			RegisterRetryCount: componentConfig.DeprecatedController.RegisterRetryCount,
		},
		EndPointController: &EndPointControllerOptions{
			ConcurrentEndpointSyncs: componentConfig.EndPointController.ConcurrentEndpointSyncs,
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			ConcurrentGCSyncs:      componentConfig.GarbageCollectorController.ConcurrentGCSyncs,
			EnableGarbageCollector: componentConfig.GarbageCollectorController.EnableGarbageCollector,
		},
		HPAController: &HPAControllerOptions{
			HorizontalPodAutoscalerSyncPeriod:                   componentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod,
			HorizontalPodAutoscalerUpscaleForbiddenWindow:       componentConfig.HPAController.HorizontalPodAutoscalerUpscaleForbiddenWindow,
			HorizontalPodAutoscalerDownscaleForbiddenWindow:     componentConfig.HPAController.HorizontalPodAutoscalerDownscaleForbiddenWindow,
			HorizontalPodAutoscalerDownscaleStabilizationWindow: componentConfig.HPAController.HorizontalPodAutoscalerDownscaleStabilizationWindow,
			HorizontalPodAutoscalerCPUInitializationPeriod:      componentConfig.HPAController.HorizontalPodAutoscalerCPUInitializationPeriod,
			HorizontalPodAutoscalerInitialReadinessDelay:        componentConfig.HPAController.HorizontalPodAutoscalerInitialReadinessDelay,
			HorizontalPodAutoscalerTolerance:                    componentConfig.HPAController.HorizontalPodAutoscalerTolerance,
			HorizontalPodAutoscalerUseRESTClients:               componentConfig.HPAController.HorizontalPodAutoscalerUseRESTClients,
		},
		JobController: &JobControllerOptions{
			ConcurrentJobSyncs: componentConfig.JobController.ConcurrentJobSyncs,
		},
		NamespaceController: &NamespaceControllerOptions{
			NamespaceSyncPeriod:      componentConfig.NamespaceController.NamespaceSyncPeriod,
			ConcurrentNamespaceSyncs: componentConfig.NamespaceController.ConcurrentNamespaceSyncs,
		},
		NodeIpamController: &NodeIpamControllerOptions{
			NodeCIDRMaskSize: componentConfig.NodeIpamController.NodeCIDRMaskSize,
		},
		NodeLifecycleController: &NodeLifecycleControllerOptions{
			EnableTaintManager:     componentConfig.NodeLifecycleController.EnableTaintManager,
			NodeMonitorGracePeriod: componentConfig.NodeLifecycleController.NodeMonitorGracePeriod,
			NodeStartupGracePeriod: componentConfig.NodeLifecycleController.NodeStartupGracePeriod,
			PodEvictionTimeout:     componentConfig.NodeLifecycleController.PodEvictionTimeout,
		},
		PersistentVolumeBinderController: &PersistentVolumeBinderControllerOptions{
			PVClaimBinderSyncPeriod: componentConfig.PersistentVolumeBinderController.PVClaimBinderSyncPeriod,
			VolumeConfiguration:     componentConfig.PersistentVolumeBinderController.VolumeConfiguration,
		},
		PodGCController: &PodGCControllerOptions{
			TerminatedPodGCThreshold: componentConfig.PodGCController.TerminatedPodGCThreshold,
		},
		ReplicaSetController: &ReplicaSetControllerOptions{
			ConcurrentRSSyncs: componentConfig.ReplicaSetController.ConcurrentRSSyncs,
		},
		ReplicationController: &ReplicationControllerOptions{
			ConcurrentRCSyncs: componentConfig.ReplicationController.ConcurrentRCSyncs,
		},
		ResourceQuotaController: &ResourceQuotaControllerOptions{
			ResourceQuotaSyncPeriod:      componentConfig.ResourceQuotaController.ResourceQuotaSyncPeriod,
			ConcurrentResourceQuotaSyncs: componentConfig.ResourceQuotaController.ConcurrentResourceQuotaSyncs,
		},
		SAController: &SAControllerOptions{
			ConcurrentSATokenSyncs: componentConfig.SAController.ConcurrentSATokenSyncs,
		},
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: componentConfig.ServiceController.ConcurrentServiceSyncs,
		},
		Controllers:   componentConfig.Controllers,
		SecureServing: apiserveroptions.NewSecureServingOptions().WithLoopback(),
		InsecureServing: (&apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.KubeCloudShared.Address),
			BindPort:    int(componentConfig.KubeCloudShared.Port),
			BindNetwork: "tcp",
		}).WithLoopback(),
		Authentication: apiserveroptions.NewDelegatingAuthenticationOptions(),
		Authorization:  apiserveroptions.NewDelegatingAuthorizationOptions(),
	}

	s.Authentication.RemoteKubeConfigFileOptional = true
	s.Authorization.RemoteKubeConfigFileOptional = true
	s.Authorization.AlwaysAllowPaths = []string{"/healthz"}

	s.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.SecureServing.ServerCert.PairName = "kube-controller-manager"
	s.SecureServing.BindPort = ports.KubeControllerManagerPort

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
	if err := componentconfigv1alpha1.AddToScheme(scheme); err != nil {
		return componentconfig.KubeControllerManagerConfiguration{}, err
	}
	if err := componentconfig.AddToScheme(scheme); err != nil {
		return componentconfig.KubeControllerManagerConfiguration{}, err
	}

	versioned := componentconfigv1alpha1.KubeControllerManagerConfiguration{}
	scheme.Default(&versioned)

	internal := componentconfig.KubeControllerManagerConfiguration{}
	if err := scheme.Convert(&versioned, &internal, nil); err != nil {
		return internal, err
	}
	internal.KubeCloudShared.Port = insecurePort
	return internal, nil
}

// Flags returns flags for a specific APIServer by section name
func (s *KubeControllerManagerOptions) Flags(allControllers []string, disabledByDefaultControllers []string) (fss apiserverflag.NamedFlagSets) {
	s.CloudProvider.AddFlags(fss.FlagSet("cloud provider"))
	s.Debugging.AddFlags(fss.FlagSet("debugging"))
	s.GenericComponent.AddFlags(fss.FlagSet("generic"))
	s.KubeCloudShared.AddFlags(fss.FlagSet("generic"))
	s.ServiceController.AddFlags(fss.FlagSet("service controller"))

	s.SecureServing.AddFlags(fss.FlagSet("secure serving"))
	s.InsecureServing.AddUnqualifiedFlags(fss.FlagSet("insecure serving"))
	s.Authentication.AddFlags(fss.FlagSet("authentication"))
	s.Authorization.AddFlags(fss.FlagSet("authorization"))

	s.AttachDetachController.AddFlags(fss.FlagSet("attachdetach controller"))
	s.CSRSigningController.AddFlags(fss.FlagSet("csrsigning controller"))
	s.DeploymentController.AddFlags(fss.FlagSet("deployment controller"))
	s.DaemonSetController.AddFlags(fss.FlagSet("daemonset controller"))
	s.DeprecatedFlags.AddFlags(fss.FlagSet("deprecated"))
	s.EndPointController.AddFlags(fss.FlagSet("endpoint controller"))
	s.GarbageCollectorController.AddFlags(fss.FlagSet("garbagecollector controller"))
	s.HPAController.AddFlags(fss.FlagSet("horizontalpodautoscaling controller"))
	s.JobController.AddFlags(fss.FlagSet("job controller"))
	s.NamespaceController.AddFlags(fss.FlagSet("namespace controller"))
	s.NodeIpamController.AddFlags(fss.FlagSet("nodeipam controller"))
	s.NodeLifecycleController.AddFlags(fss.FlagSet("nodelifecycle controller"))
	s.PersistentVolumeBinderController.AddFlags(fss.FlagSet("persistentvolume-binder controller"))
	s.PodGCController.AddFlags(fss.FlagSet("podgc controller"))
	s.ReplicaSetController.AddFlags(fss.FlagSet("replicaset controller"))
	s.ReplicationController.AddFlags(fss.FlagSet("replicationcontroller"))
	s.ResourceQuotaController.AddFlags(fss.FlagSet("resourcequota controller"))
	s.SAController.AddFlags(fss.FlagSet("serviceaccount controller"))

	fs := fss.FlagSet("misc")
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
	utilfeature.DefaultFeatureGate.AddFlag(fss.FlagSet("generic"))

	return fss
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config) error {
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
	if err := s.InsecureServing.ApplyTo(&c.InsecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}
	if err := s.SecureServing.ApplyTo(&c.SecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}
	if s.SecureServing.BindPort != 0 || s.SecureServing.Listener != nil {
		if err := s.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
			return err
		}
		if err := s.Authorization.ApplyTo(&c.Authorization); err != nil {
			return err
		}
	}

	// sync back to component config
	// TODO: find more elegant way than syncing back the values.
	c.ComponentConfig.KubeCloudShared.Port = int32(s.InsecureServing.BindPort)
	c.ComponentConfig.KubeCloudShared.Address = s.InsecureServing.BindAddress.String()

	c.ComponentConfig.Controllers = s.Controllers
	c.ComponentConfig.ExternalCloudVolumePlugin = s.ExternalCloudVolumePlugin

	return nil
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

	if err := s.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{net.ParseIP("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	kubeconfig, err := clientcmd.BuildConfigFromFlags(s.Master, s.Kubeconfig)
	if err != nil {
		return nil, err
	}
	kubeconfig.ContentConfig.ContentType = s.GenericComponent.ContentType
	kubeconfig.QPS = s.GenericComponent.KubeAPIQPS
	kubeconfig.Burst = int(s.GenericComponent.KubeAPIBurst)

	client, err := clientset.NewForConfig(restclient.AddUserAgent(kubeconfig, KubeControllerManagerUserAgent))
	if err != nil {
		return nil, err
	}

	// shallow copy, do not modify the kubeconfig.Timeout.
	config := *kubeconfig
	config.Timeout = s.GenericComponent.LeaderElection.RenewDeadline.Duration
	leaderElectionClient := clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "leader-election"))

	eventRecorder := createRecorder(client, KubeControllerManagerUserAgent)

	c := &kubecontrollerconfig.Config{
		Client:               client,
		Kubeconfig:           kubeconfig,
		EventRecorder:        eventRecorder,
		LeaderElectionClient: leaderElectionClient,
	}
	if err := s.ApplyTo(c); err != nil {
		return nil, err
	}

	return c, nil
}

func createRecorder(kubeClient clientset.Interface, userAgent string) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	// TODO: remove dependency on the legacyscheme
	return eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: userAgent})
}
