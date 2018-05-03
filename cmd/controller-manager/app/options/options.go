/*
Copyright 2017 The Kubernetes Authors.

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
	"net"

	"github.com/golang/glog"

	"github.com/spf13/pflag"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	genericcontrollermanager "k8s.io/kubernetes/cmd/controller-manager/app"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
)

// GenericControllerManagerOptions is the common structure for a controller manager. It works with NewGenericControllerManagerOptions
// and AddDefaultControllerFlags to create the common components of kube-controller-manager and cloud-controller-manager.
type GenericControllerManagerOptions struct {
	CloudProvider    *CloudProviderOptions
	Debugging        *DebuggingOptions
	GenericComponent *GenericComponentConfigOptions
	KubeCloudShared  *KubeCloudSharedOptions

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
	ServiceController                *ServiceControllerOptions

	Controllers               []string
	ExternalCloudVolumePlugin string

	SecureServing *apiserveroptions.SecureServingOptions
	// TODO: remove insecure serving mode
	InsecureServing *InsecureServingOptions
	Authentication  *apiserveroptions.DelegatingAuthenticationOptions
	Authorization   *apiserveroptions.DelegatingAuthorizationOptions

	Master     string
	Kubeconfig string
}

const (
	// These defaults are deprecated and exported so that we can warn if
	// they are being used.

	// DefaultClusterSigningCertFile is deprecated. Do not use.
	DefaultClusterSigningCertFile = "/etc/kubernetes/ca/ca.pem"
	// DefaultClusterSigningKeyFile is deprecated. Do not use.
	DefaultClusterSigningKeyFile = "/etc/kubernetes/ca/ca.key"
)

// NewGenericControllerManagerOptions returns common/default configuration values for both
// the kube-controller-manager and the cloud-contoller-manager. Any common changes should
// be made here. Any individual changes should be made in that controller.
func NewGenericControllerManagerOptions(componentConfig componentconfig.KubeControllerManagerConfiguration) *GenericControllerManagerOptions {
	o := &GenericControllerManagerOptions{
		CloudProvider: &CloudProviderOptions{},
		Debugging:     &DebuggingOptions{},
		GenericComponent: &GenericComponentConfigOptions{
			MinResyncPeriod:         componentConfig.GenericComponent.MinResyncPeriod,
			ContentType:             componentConfig.GenericComponent.ContentType,
			KubeAPIQPS:              componentConfig.GenericComponent.KubeAPIQPS,
			KubeAPIBurst:            componentConfig.GenericComponent.KubeAPIBurst,
			ControllerStartInterval: componentConfig.GenericComponent.ControllerStartInterval,
			LeaderElection:          componentConfig.GenericComponent.LeaderElection,
		},
		KubeCloudShared: &KubeCloudSharedOptions{
			Port:                      componentConfig.KubeCloudShared.Port,
			Address:                   componentConfig.KubeCloudShared.Address,
			RouteReconciliationPeriod: componentConfig.KubeCloudShared.RouteReconciliationPeriod,
			NodeMonitorPeriod:         componentConfig.KubeCloudShared.NodeMonitorPeriod,
			ClusterName:               componentConfig.KubeCloudShared.ClusterName,
			ConfigureCloudRoutes:      componentConfig.KubeCloudShared.ConfigureCloudRoutes,
		},
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
		ServiceController: &ServiceControllerOptions{
			ConcurrentServiceSyncs: componentConfig.ServiceController.ConcurrentServiceSyncs,
		},
		Controllers:   componentConfig.Controllers,
		SecureServing: apiserveroptions.NewSecureServingOptions(),
		InsecureServing: &InsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.KubeCloudShared.Address),
			BindPort:    int(componentConfig.KubeCloudShared.Port),
			BindNetwork: "tcp",
		},
		Authentication: nil, // TODO: enable with apiserveroptions.NewDelegatingAuthenticationOptions()
		Authorization:  nil, // TODO: enable with apiserveroptions.NewDelegatingAuthorizationOptions()
	}

	// disable secure serving for now
	// TODO: enable HTTPS by default
	o.SecureServing.BindPort = 0

	return o
}

// NewDefaultControllerManagerComponentConfig returns default kube-controller manager configuration object.
func NewDefaultControllerManagerComponentConfig(insecurePort int32) componentconfig.KubeControllerManagerConfiguration {
	scheme := runtime.NewScheme()
	componentconfigv1alpha1.AddToScheme(scheme)
	versioned := componentconfigv1alpha1.KubeControllerManagerConfiguration{}
	scheme.Default(&versioned)
	internal := componentconfig.KubeControllerManagerConfiguration{}
	scheme.Convert(&versioned, &internal, nil)
	internal.KubeCloudShared.Port = insecurePort
	return internal
}

// AddFlags adds common/default flags for both the kube and cloud Controller Manager Server to the
// specified FlagSet. Any common changes should be made here. Any individual changes should be made in that controller.
func (o *GenericControllerManagerOptions) AddFlags(fs *pflag.FlagSet) {

	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&o.Kubeconfig, "kubeconfig", o.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	o.CloudProvider.AddFlags(fs)
	o.Debugging.AddFlags(fs)
	o.GenericComponent.AddFlags(fs)
	o.KubeCloudShared.AddFlags(fs)
	o.ServiceController.AddFlags(fs)
	o.SecureServing.AddFlags(fs)
	o.InsecureServing.AddFlags(fs)
	o.InsecureServing.AddDeprecatedFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)
}

// ApplyTo fills up controller manager config with options and userAgent
func (o *GenericControllerManagerOptions) ApplyTo(c *genericcontrollermanager.Config, userAgent string) error {
	if err := o.CloudProvider.ApplyTo(&c.ComponentConfig.CloudProvider); err != nil {
		return err
	}
	if err := o.Debugging.ApplyTo(&c.ComponentConfig.Debugging); err != nil {
		return err
	}
	if err := o.GenericComponent.ApplyTo(&c.ComponentConfig.GenericComponent); err != nil {
		return err
	}
	if err := o.KubeCloudShared.ApplyTo(&c.ComponentConfig.KubeCloudShared); err != nil {
		return err
	}
	if err := o.AttachDetachController.ApplyTo(&c.ComponentConfig.AttachDetachController); err != nil {
		return err
	}
	if err := o.CSRSigningController.ApplyTo(&c.ComponentConfig.CSRSigningController); err != nil {
		return err
	}
	if err := o.DaemonSetController.ApplyTo(&c.ComponentConfig.DaemonSetController); err != nil {
		return err
	}
	if err := o.DeploymentController.ApplyTo(&c.ComponentConfig.DeploymentController); err != nil {
		return err
	}
	if err := o.DeprecatedFlags.ApplyTo(&c.ComponentConfig.DeprecatedController); err != nil {
		return err
	}
	if err := o.EndPointController.ApplyTo(&c.ComponentConfig.EndPointController); err != nil {
		return err
	}
	if err := o.GarbageCollectorController.ApplyTo(&c.ComponentConfig.GarbageCollectorController); err != nil {
		return err
	}
	if err := o.HPAController.ApplyTo(&c.ComponentConfig.HPAController); err != nil {
		return err
	}
	if err := o.JobController.ApplyTo(&c.ComponentConfig.JobController); err != nil {
		return err
	}
	if err := o.NamespaceController.ApplyTo(&c.ComponentConfig.NamespaceController); err != nil {
		return err
	}
	if err := o.NodeIpamController.ApplyTo(&c.ComponentConfig.NodeIpamController); err != nil {
		return err
	}
	if err := o.NodeLifecycleController.ApplyTo(&c.ComponentConfig.NodeLifecycleController); err != nil {
		return err
	}
	if err := o.PersistentVolumeBinderController.ApplyTo(&c.ComponentConfig.PersistentVolumeBinderController); err != nil {
		return err
	}
	if err := o.PodGCController.ApplyTo(&c.ComponentConfig.PodGCController); err != nil {
		return err
	}
	if err := o.ReplicaSetController.ApplyTo(&c.ComponentConfig.ReplicaSetController); err != nil {
		return err
	}
	if err := o.ReplicationController.ApplyTo(&c.ComponentConfig.ReplicationController); err != nil {
		return err
	}
	if err := o.ResourceQuotaController.ApplyTo(&c.ComponentConfig.ResourceQuotaController); err != nil {
		return err
	}
	if err := o.SAController.ApplyTo(&c.ComponentConfig.SAController); err != nil {
		return err
	}
	if err := o.ServiceController.ApplyTo(&c.ComponentConfig.ServiceController); err != nil {
		return err
	}
	if err := o.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := o.InsecureServing.ApplyTo(&c.InsecureServing, &c.ComponentConfig.KubeCloudShared); err != nil {
		return err
	}
	if err := o.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
		return err
	}
	if err := o.Authorization.ApplyTo(&c.Authorization); err != nil {
		return err
	}

	var err error
	c.Kubeconfig, err = clientcmd.BuildConfigFromFlags(o.Master, o.Kubeconfig)
	if err != nil {
		return err
	}
	c.Kubeconfig.ContentConfig.ContentType = o.GenericComponent.ContentType
	c.Kubeconfig.QPS = o.GenericComponent.KubeAPIQPS
	c.Kubeconfig.Burst = int(o.GenericComponent.KubeAPIBurst)

	c.Client, err = clientset.NewForConfig(restclient.AddUserAgent(c.Kubeconfig, userAgent))
	if err != nil {
		return err
	}

	c.LeaderElectionClient = clientset.NewForConfigOrDie(restclient.AddUserAgent(c.Kubeconfig, "leader-election"))

	c.EventRecorder = createRecorder(c.Client, userAgent)

	return nil
}

// Validate checks GenericControllerManagerOptions and return a slice of found errors.
func (o *GenericControllerManagerOptions) Validate() []error {
	errors := []error{}
	errors = append(errors, o.CloudProvider.Validate()...)
	errors = append(errors, o.Debugging.Validate()...)
	errors = append(errors, o.GenericComponent.Validate()...)
	errors = append(errors, o.KubeCloudShared.Validate()...)
	errors = append(errors, o.AttachDetachController.Validate()...)
	errors = append(errors, o.CSRSigningController.Validate()...)
	errors = append(errors, o.DaemonSetController.Validate()...)
	errors = append(errors, o.DeploymentController.Validate()...)
	errors = append(errors, o.DeprecatedFlags.Validate()...)
	errors = append(errors, o.EndPointController.Validate()...)
	errors = append(errors, o.GarbageCollectorController.Validate()...)
	errors = append(errors, o.HPAController.Validate()...)
	errors = append(errors, o.JobController.Validate()...)
	errors = append(errors, o.NamespaceController.Validate()...)
	errors = append(errors, o.NodeIpamController.Validate()...)
	errors = append(errors, o.NodeLifecycleController.Validate()...)
	errors = append(errors, o.PersistentVolumeBinderController.Validate()...)
	errors = append(errors, o.PodGCController.Validate()...)
	errors = append(errors, o.ReplicaSetController.Validate()...)
	errors = append(errors, o.ReplicationController.Validate()...)
	errors = append(errors, o.ResourceQuotaController.Validate()...)
	errors = append(errors, o.SAController.Validate()...)
	errors = append(errors, o.ServiceController.Validate()...)
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.InsecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)

	// TODO: validate component config, master and kubeconfig

	return errors
}

func createRecorder(kubeClient kubernetes.Interface, userAgent string) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	return eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: userAgent})
}
