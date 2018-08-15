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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
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

const (
	// KubeControllerManagerUserAgent is the userAgent name when starting kube-controller managers.
	KubeControllerManagerUserAgent = "kube-controller-manager"
)

// KubeControllerManagerOptions is the main context object for the kube-controller manager.
type KubeControllerManagerOptions struct {
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

	SecureServing *apiserveroptions.SecureServingOptions
	// TODO: remove insecure serving mode
	InsecureServing *apiserveroptions.DeprecatedInsecureServingOptions
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
		GenericComponent: cmoptions.NewGenericComponentConfigOptions(componentConfig.Generic),
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
			ConcurrentEndpointSyncs: componentConfig.EndpointController.ConcurrentEndpointSyncs,
		},
		GarbageCollectorController: &GarbageCollectorControllerOptions{
			ConcurrentGCSyncs:      componentConfig.GarbageCollectorController.ConcurrentGCSyncs,
			EnableGarbageCollector: componentConfig.GarbageCollectorController.EnableGarbageCollector,
		},
		HPAController: &HPAControllerOptions{
			HorizontalPodAutoscalerSyncPeriod:               componentConfig.HPAController.HorizontalPodAutoscalerSyncPeriod,
			HorizontalPodAutoscalerUpscaleForbiddenWindow:   componentConfig.HPAController.HorizontalPodAutoscalerUpscaleForbiddenWindow,
			HorizontalPodAutoscalerDownscaleForbiddenWindow: componentConfig.HPAController.HorizontalPodAutoscalerDownscaleForbiddenWindow,
			HorizontalPodAutoscalerTolerance:                componentConfig.HPAController.HorizontalPodAutoscalerTolerance,
			HorizontalPodAutoscalerUseRESTClients:           componentConfig.HPAController.HorizontalPodAutoscalerUseRESTClients,
		},
		JobController: &JobControllerOptions{
			ConcurrentJobSyncs: componentConfig.JobController.ConcurrentJobSyncs,
		},
		NamespaceController: &NamespaceControllerOptions{
			NamespaceSyncPeriod:      componentConfig.NamespaceController.NamespaceSyncPeriod,
			ConcurrentNamespaceSyncs: componentConfig.NamespaceController.ConcurrentNamespaceSyncs,
		},
		NodeIpamController: &NodeIpamControllerOptions{
			NodeCIDRMaskSize: componentConfig.NodeIPAMController.NodeCIDRMaskSize,
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
		SecureServing: apiserveroptions.NewSecureServingOptions(),
		InsecureServing: &apiserveroptions.DeprecatedInsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.Generic.Address),
			BindPort:    int(componentConfig.Generic.Port),
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
	internal.Generic.Port = insecurePort
	return internal, nil
}

// AddFlags adds flags for a specific KubeControllerManagerOptions to the specified FlagSet
func (s *KubeControllerManagerOptions) AddFlags(fs *pflag.FlagSet, allControllers, disabledByDefaultControllers []string) {
	s.GenericComponent.AddFlags(fs, allControllers, disabledByDefaultControllers)
	s.KubeCloudShared.AddFlags(fs)
	s.ServiceController.AddFlags(fs)

	s.SecureServing.AddFlags(fs)
	s.InsecureServing.AddUnqualifiedFlags(fs)
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
	var dummy string
	fs.MarkDeprecated("insecure-experimental-approve-all-kubelet-csrs-for-group", "This flag does nothing.")
	fs.StringVar(&dummy, "insecure-experimental-approve-all-kubelet-csrs-for-group", "", "This flag does nothing.")
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up controller manager config with options.
func (s *KubeControllerManagerOptions) ApplyTo(c *kubecontrollerconfig.Config) error {
	if err := s.GenericComponent.ApplyTo(&c.ComponentConfig.Generic); err != nil {
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
	if err := s.EndPointController.ApplyTo(&c.ComponentConfig.EndpointController); err != nil {
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
	if err := s.NodeIpamController.ApplyTo(&c.ComponentConfig.NodeIPAMController); err != nil {
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
	if err := s.InsecureServing.ApplyTo(&c.InsecureServing); err != nil {
		return err
	}
	if err := s.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := s.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
		return err
	}
	if err := s.Authorization.ApplyTo(&c.Authorization); err != nil {
		return err
	}

	// sync back to component config
	// TODO: find more elegant way than syncing back the values.
	c.ComponentConfig.Generic.Port = int32(s.InsecureServing.BindPort)
	c.ComponentConfig.Generic.Address = s.InsecureServing.BindAddress.String()

	return nil
}

// Validate is used to validate the options and config before launching the controller manager
func (s *KubeControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string) error {
	var errs []error

	errs = append(errs, s.GenericComponent.Validate(allControllers, disabledByDefaultControllers)...)
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
	kubeconfig.ContentConfig.ContentType = s.GenericComponent.ClientConnection.ContentType
	kubeconfig.QPS = s.GenericComponent.ClientConnection.QPS
	kubeconfig.Burst = int(s.GenericComponent.ClientConnection.Burst)

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
