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
	"time"

	"github.com/cloudflare/cfssl/helpers"
	"github.com/golang/glog"

	"github.com/spf13/pflag"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
)

// GenericControllerManagerOptions is the common structure for a controller manager. It works with NewGenericControllerManagerOptions
// and AddDefaultControllerFlags to create the common components of kube-controller-manager and cloud-controller-manager.
type GenericControllerManagerOptions struct {
	// TODO: turn ComponentConfig into modular option structs. This is not generic.
	ComponentConfig componentconfig.KubeControllerManagerConfiguration

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
func NewGenericControllerManagerOptions(componentConfig componentconfig.KubeControllerManagerConfiguration) GenericControllerManagerOptions {
	o := GenericControllerManagerOptions{
		ComponentConfig: componentConfig,
		SecureServing:   apiserveroptions.NewSecureServingOptions(),
		InsecureServing: &InsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.Address),
			BindPort:    int(componentConfig.Port),
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
	return componentconfig.KubeControllerManagerConfiguration{
		Controllers:                                     []string{"*"},
		Port:                                            insecurePort,
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
		RouteReconciliationPeriod:                       metav1.Duration{Duration: 10 * time.Second},
		ResourceQuotaSyncPeriod:                         metav1.Duration{Duration: 5 * time.Minute},
		NamespaceSyncPeriod:                             metav1.Duration{Duration: 5 * time.Minute},
		PVClaimBinderSyncPeriod:                         metav1.Duration{Duration: 15 * time.Second},
		HorizontalPodAutoscalerSyncPeriod:               metav1.Duration{Duration: 30 * time.Second},
		HorizontalPodAutoscalerUpscaleForbiddenWindow:   metav1.Duration{Duration: 3 * time.Minute},
		HorizontalPodAutoscalerDownscaleForbiddenWindow: metav1.Duration{Duration: 5 * time.Minute},
		HorizontalPodAutoscalerTolerance:                0.1,
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
		LeaderElection:                        leaderelectionconfig.DefaultLeaderElectionConfiguration(),
		ControllerStartInterval:               metav1.Duration{Duration: 0 * time.Second},
		EnableGarbageCollector:                true,
		ConcurrentGCSyncs:                     20,
		ClusterSigningCertFile:                DefaultClusterSigningCertFile,
		ClusterSigningKeyFile:                 DefaultClusterSigningKeyFile,
		ClusterSigningDuration:                metav1.Duration{Duration: helpers.OneYear},
		ReconcilerSyncLoopPeriod:              metav1.Duration{Duration: 60 * time.Second},
		EnableTaintManager:                    true,
		HorizontalPodAutoscalerUseRESTClients: true,
	}
}

// AddFlags adds common/default flags for both the kube and cloud Controller Manager Server to the
// specified FlagSet. Any common changes should be made here. Any individual changes should be made in that controller.
func (o *GenericControllerManagerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.BoolVar(&o.ComponentConfig.UseServiceAccountCredentials, "use-service-account-credentials", o.ComponentConfig.UseServiceAccountCredentials, "If true, use individual service account credentials for each controller.")
	fs.StringVar(&o.ComponentConfig.CloudConfigFile, "cloud-config", o.ComponentConfig.CloudConfigFile, "The path to the cloud provider configuration file. Empty string for no configuration file.")
	fs.BoolVar(&o.ComponentConfig.AllowUntaggedCloud, "allow-untagged-cloud", false, "Allow the cluster to run without the cluster-id on cloud instances. This is a legacy mode of operation and a cluster-id will be required in the future.")
	fs.MarkDeprecated("allow-untagged-cloud", "This flag is deprecated and will be removed in a future release. A cluster-id will be required on cloud instances.")
	fs.DurationVar(&o.ComponentConfig.RouteReconciliationPeriod.Duration, "route-reconciliation-period", o.ComponentConfig.RouteReconciliationPeriod.Duration, "The period for reconciling routes created for Nodes by cloud provider.")
	fs.DurationVar(&o.ComponentConfig.MinResyncPeriod.Duration, "min-resync-period", o.ComponentConfig.MinResyncPeriod.Duration, "The resync period in reflectors will be random between MinResyncPeriod and 2*MinResyncPeriod.")
	fs.DurationVar(&o.ComponentConfig.NodeMonitorPeriod.Duration, "node-monitor-period", o.ComponentConfig.NodeMonitorPeriod.Duration,
		"The period for syncing NodeStatus in NodeController.")
	fs.BoolVar(&o.ComponentConfig.EnableProfiling, "profiling", true, "Enable profiling via web interface host:port/debug/pprof/")
	fs.BoolVar(&o.ComponentConfig.EnableContentionProfiling, "contention-profiling", false, "Enable lock contention profiling, if profiling is enabled.")
	fs.StringVar(&o.ComponentConfig.ClusterName, "cluster-name", o.ComponentConfig.ClusterName, "The instance prefix for the cluster.")
	fs.StringVar(&o.ComponentConfig.ClusterCIDR, "cluster-cidr", o.ComponentConfig.ClusterCIDR, "CIDR Range for Pods in cluster. Requires --allocate-node-cidrs to be true")
	fs.BoolVar(&o.ComponentConfig.AllocateNodeCIDRs, "allocate-node-cidrs", false, "Should CIDRs for Pods be allocated and set on the cloud provider.")
	fs.StringVar(&o.ComponentConfig.CIDRAllocatorType, "cidr-allocator-type", "RangeAllocator", "Type of CIDR allocator to use")
	fs.BoolVar(&o.ComponentConfig.ConfigureCloudRoutes, "configure-cloud-routes", true, "Should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&o.Kubeconfig, "kubeconfig", o.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.StringVar(&o.ComponentConfig.ContentType, "kube-api-content-type", o.ComponentConfig.ContentType, "Content type of requests sent to apiserver.")
	fs.Float32Var(&o.ComponentConfig.KubeAPIQPS, "kube-api-qps", o.ComponentConfig.KubeAPIQPS, "QPS to use while talking with kubernetes apiserver.")
	fs.Int32Var(&o.ComponentConfig.KubeAPIBurst, "kube-api-burst", o.ComponentConfig.KubeAPIBurst, "Burst to use while talking with kubernetes apiserver.")
	fs.DurationVar(&o.ComponentConfig.ControllerStartInterval.Duration, "controller-start-interval", o.ComponentConfig.ControllerStartInterval.Duration, "Interval between starting controller managers.")

	o.SecureServing.AddFlags(fs)
	o.InsecureServing.AddFlags(fs)
	o.InsecureServing.AddDeprecatedFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)
}

// ApplyTo fills up controller manager config with options and userAgent
func (o *GenericControllerManagerOptions) ApplyTo(c *genericcontrollermanager.Config, userAgent string) error {
	c.ComponentConfig = o.ComponentConfig

	if err := o.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := o.InsecureServing.ApplyTo(&c.InsecureServing, &c.ComponentConfig); err != nil {
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
	c.Kubeconfig.ContentConfig.ContentType = o.ComponentConfig.ContentType
	c.Kubeconfig.QPS = o.ComponentConfig.KubeAPIQPS
	c.Kubeconfig.Burst = int(o.ComponentConfig.KubeAPIBurst)

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
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.InsecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)

	// TODO: validate component config, master and kubeconfig

	return errors
}

func createRecorder(kubeClient *kubernetes.Clientset, userAgent string) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})
	return eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: userAgent})
}
