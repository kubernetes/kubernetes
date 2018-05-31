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
	"fmt"
	"net"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	cloudcontrollerconfig "k8s.io/kubernetes/cmd/cloud-controller-manager/app/config"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/master/ports"
	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

const (
	// CloudControllerManagerUserAgent is the userAgent name when starting cloud-controller managers.
	CloudControllerManagerUserAgent = "cloud-controller-manager"
)

// CloudControllerManagerOptions is the main context object for the controller manager.
type CloudControllerManagerOptions struct {
	CloudProvider     *cmoptions.CloudProviderOptions
	Debugging         *cmoptions.DebuggingOptions
	GenericComponent  *cmoptions.GenericComponentConfigOptions
	KubeCloudShared   *cmoptions.KubeCloudSharedOptions
	ServiceController *cmoptions.ServiceControllerOptions

	SecureServing *apiserveroptions.SecureServingOptions
	// TODO: remove insecure serving mode
	InsecureServing *cmoptions.InsecureServingOptions
	Authentication  *apiserveroptions.DelegatingAuthenticationOptions
	Authorization   *apiserveroptions.DelegatingAuthorizationOptions

	Master     string
	Kubeconfig string

	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

// NewCloudControllerManagerOptions creates a new ExternalCMServer with a default config.
func NewCloudControllerManagerOptions() (*CloudControllerManagerOptions, error) {
	componentConfig, err := NewDefaultComponentConfig(ports.InsecureCloudControllerManagerPort)
	if err != nil {
		return nil, err
	}

	s := CloudControllerManagerOptions{
		CloudProvider:    &cmoptions.CloudProviderOptions{},
		Debugging:        &cmoptions.DebuggingOptions{},
		GenericComponent: cmoptions.NewGenericComponentConfigOptions(componentConfig.GenericComponent),
		KubeCloudShared:  cmoptions.NewKubeCloudSharedOptions(componentConfig.KubeCloudShared),
		ServiceController: &cmoptions.ServiceControllerOptions{
			ConcurrentServiceSyncs: componentConfig.ServiceController.ConcurrentServiceSyncs,
		},
		SecureServing: apiserveroptions.NewSecureServingOptions(),
		InsecureServing: &cmoptions.InsecureServingOptions{
			BindAddress: net.ParseIP(componentConfig.KubeCloudShared.Address),
			BindPort:    int(componentConfig.KubeCloudShared.Port),
			BindNetwork: "tcp",
		},
		Authentication:            nil, // TODO: enable with apiserveroptions.NewDelegatingAuthenticationOptions()
		Authorization:             nil, // TODO: enable with apiserveroptions.NewDelegatingAuthorizationOptions()
		NodeStatusUpdateFrequency: componentConfig.NodeStatusUpdateFrequency,
	}

	s.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.SecureServing.ServerCert.PairName = "cloud-controller-manager"

	// disable secure serving for now
	// TODO: enable HTTPS by default
	s.SecureServing.BindPort = 0

	return &s, nil
}

// NewDefaultComponentConfig returns cloud-controller manager configuration object.
func NewDefaultComponentConfig(insecurePort int32) (componentconfig.CloudControllerManagerConfiguration, error) {
	scheme := runtime.NewScheme()
	componentconfigv1alpha1.AddToScheme(scheme)
	componentconfig.AddToScheme(scheme)

	versioned := componentconfigv1alpha1.CloudControllerManagerConfiguration{}
	scheme.Default(&versioned)

	internal := componentconfig.CloudControllerManagerConfiguration{}
	if err := scheme.Convert(&versioned, &internal, nil); err != nil {
		return internal, err
	}
	internal.KubeCloudShared.Port = insecurePort
	return internal, nil
}

// AddFlags adds flags for a specific ExternalCMServer to the specified FlagSet
func (o *CloudControllerManagerOptions) AddFlags(fs *pflag.FlagSet) {
	o.CloudProvider.AddFlags(fs)
	o.Debugging.AddFlags(fs)
	o.GenericComponent.AddFlags(fs)
	o.KubeCloudShared.AddFlags(fs)
	o.ServiceController.AddFlags(fs)

	o.SecureServing.AddFlags(fs)
	o.InsecureServing.AddFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)

	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&o.Kubeconfig, "kubeconfig", o.Kubeconfig, "Path to kubeconfig file with authorization and master location information.")
	fs.DurationVar(&o.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", o.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up cloud controller manager config with options.
func (o *CloudControllerManagerOptions) ApplyTo(c *cloudcontrollerconfig.Config, userAgent string) error {
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
	if err := o.ServiceController.ApplyTo(&c.ComponentConfig.ServiceController); err != nil {
		return err
	}
	if err := o.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := o.InsecureServing.ApplyTo(&c.InsecureServing); err != nil {
		return err
	}
	if err := o.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
		return err
	}
	if err := o.Authorization.ApplyTo(&c.Authorization); err != nil {
		return err
	}

	// sync back to component config
	// TODO: find more elegant way than syncing back the values.
	c.ComponentConfig.KubeCloudShared.Port = int32(o.InsecureServing.BindPort)
	c.ComponentConfig.KubeCloudShared.Address = o.InsecureServing.BindAddress.String()

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
	c.ComponentConfig.NodeStatusUpdateFrequency = o.NodeStatusUpdateFrequency

	return nil
}

// Validate is used to validate config before launching the cloud controller manager
func (o *CloudControllerManagerOptions) Validate() error {
	errors := []error{}

	errors = append(errors, o.CloudProvider.Validate()...)
	errors = append(errors, o.Debugging.Validate()...)
	errors = append(errors, o.GenericComponent.Validate()...)
	errors = append(errors, o.KubeCloudShared.Validate()...)
	errors = append(errors, o.ServiceController.Validate()...)
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.InsecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)

	if len(o.CloudProvider.Name) == 0 {
		errors = append(errors, fmt.Errorf("--cloud-provider cannot be empty"))
	}

	return utilerrors.NewAggregate(errors)
}

// Config return a cloud controller manager config objective
func (o *CloudControllerManagerOptions) Config() (*cloudcontrollerconfig.Config, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}

	c := &cloudcontrollerconfig.Config{}
	if err := o.ApplyTo(c, CloudControllerManagerUserAgent); err != nil {
		return nil, err
	}

	return c, nil
}

func createRecorder(kubeClient kubernetes.Interface, userAgent string) record.EventRecorder {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})
	// TODO: remove dependence on the legacyscheme
	return eventBroadcaster.NewRecorder(legacyscheme.Scheme, v1.EventSource{Component: userAgent})
}
