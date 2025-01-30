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
	"context"
	"fmt"
	"math/rand"
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app/config"
	ccmconfig "k8s.io/cloud-provider/config"
	ccmconfigscheme "k8s.io/cloud-provider/config/install"
	ccmconfigv1alpha1 "k8s.io/cloud-provider/config/v1alpha1"
	"k8s.io/cloud-provider/names"
	cliflag "k8s.io/component-base/cli/flag"
	cmoptions "k8s.io/controller-manager/options"
	"k8s.io/controller-manager/pkg/clientbuilder"
	netutils "k8s.io/utils/net"

	// add the related feature gates
	_ "k8s.io/controller-manager/pkg/features/register"
)

const (
	// CloudControllerManagerUserAgent is the userAgent name when starting cloud-controller managers.
	CloudControllerManagerUserAgent = "cloud-controller-manager"
)

// CloudControllerManagerOptions is the main context object for the controller manager.
type CloudControllerManagerOptions struct {
	Generic           *cmoptions.GenericControllerManagerConfigurationOptions
	KubeCloudShared   *KubeCloudSharedOptions
	ServiceController *ServiceControllerOptions
	NodeController    *NodeControllerOptions

	SecureServing  *apiserveroptions.SecureServingOptionsWithLoopback
	Authentication *apiserveroptions.DelegatingAuthenticationOptions
	Authorization  *apiserveroptions.DelegatingAuthorizationOptions

	Master string

	WebhookServing *WebhookServingOptions
	Webhook        *WebhookOptions

	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

// ProviderDefaults are provided by the consumer when calling
// NewCloudControllerManagerOptions(), so that they can customize certain flag
// default values.
type ProviderDefaults struct {
	// WebhookBindAddress is the default address.  It can be overridden by "--webhook-bind-address".
	WebhookBindAddress *net.IP
	// WebhookBindPort is the default port.  It can be overridden by "--webhook-bind-port".
	WebhookBindPort *int
}

// NewCloudControllerManagerOptions creates a new ExternalCMServer with a default config.
func NewCloudControllerManagerOptions() (*CloudControllerManagerOptions, error) {
	return NewCloudControllerManagerOptionsWithProviderDefaults(ProviderDefaults{})
}

// NewCloudControllerManagerOptionsWithProviderDefaults creates a new
// ExternalCMServer with a default config, but allows the cloud provider to
// override a select number of default option values.
func NewCloudControllerManagerOptionsWithProviderDefaults(defaults ProviderDefaults) (*CloudControllerManagerOptions, error) {
	componentConfig, err := NewDefaultComponentConfig()
	if err != nil {
		return nil, err
	}

	s := CloudControllerManagerOptions{
		Generic:         cmoptions.NewGenericControllerManagerConfigurationOptions(&componentConfig.Generic),
		KubeCloudShared: NewKubeCloudSharedOptions(&componentConfig.KubeCloudShared),
		NodeController: &NodeControllerOptions{
			NodeControllerConfiguration: &componentConfig.NodeController,
		},
		ServiceController: &ServiceControllerOptions{
			ServiceControllerConfiguration: &componentConfig.ServiceController,
		},
		SecureServing:             apiserveroptions.NewSecureServingOptions().WithLoopback(),
		Webhook:                   NewWebhookOptions(),
		WebhookServing:            NewWebhookServingOptions(defaults),
		Authentication:            apiserveroptions.NewDelegatingAuthenticationOptions(),
		Authorization:             apiserveroptions.NewDelegatingAuthorizationOptions(),
		NodeStatusUpdateFrequency: componentConfig.NodeStatusUpdateFrequency,
	}

	s.Authentication.RemoteKubeConfigFileOptional = true
	s.Authorization.RemoteKubeConfigFileOptional = true

	// Set the PairName but leave certificate directory blank to generate in-memory by default
	s.SecureServing.ServerCert.CertDirectory = ""
	s.SecureServing.ServerCert.PairName = "cloud-controller-manager"
	s.SecureServing.BindPort = cloudprovider.CloudControllerManagerPort

	s.Generic.LeaderElection.ResourceName = "cloud-controller-manager"
	s.Generic.LeaderElection.ResourceNamespace = "kube-system"

	return &s, nil
}

// NewDefaultComponentConfig returns cloud-controller manager configuration object.
func NewDefaultComponentConfig() (*ccmconfig.CloudControllerManagerConfiguration, error) {
	versioned := &ccmconfigv1alpha1.CloudControllerManagerConfiguration{}
	ccmconfigscheme.Scheme.Default(versioned)

	internal := &ccmconfig.CloudControllerManagerConfiguration{}
	if err := ccmconfigscheme.Scheme.Convert(versioned, internal, nil); err != nil {
		return nil, err
	}

	return internal, nil
}

// Flags returns flags for a specific CloudController by section name
func (o *CloudControllerManagerOptions) Flags(allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string, allWebhooks, disabledByDefaultWebhooks []string) cliflag.NamedFlagSets {
	fss := cliflag.NamedFlagSets{}
	o.Generic.AddFlags(&fss, allControllers, disabledByDefaultControllers, controllerAliases)
	o.KubeCloudShared.AddFlags(fss.FlagSet("generic"))
	o.NodeController.AddFlags(fss.FlagSet(names.CloudNodeController))
	o.ServiceController.AddFlags(fss.FlagSet(names.ServiceLBController))
	if o.Webhook != nil {
		o.Webhook.AddFlags(fss.FlagSet("webhook"), allWebhooks, disabledByDefaultWebhooks)
	}
	if o.WebhookServing != nil {
		o.WebhookServing.AddFlags(fss.FlagSet("webhook serving"))
	}

	o.SecureServing.AddFlags(fss.FlagSet("secure serving"))
	o.Authentication.AddFlags(fss.FlagSet("authentication"))
	o.Authorization.AddFlags(fss.FlagSet("authorization"))

	fs := fss.FlagSet("misc")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig).")
	fs.StringVar(&o.Generic.ClientConnection.Kubeconfig, "kubeconfig", o.Generic.ClientConnection.Kubeconfig, "Path to kubeconfig file with authorization and master location information (the master location can be overridden by the master flag).")
	fs.DurationVar(&o.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", o.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")
	utilfeature.DefaultMutableFeatureGate.AddFlag(fss.FlagSet("generic"))

	return fss
}

// ApplyTo fills up cloud controller manager config with options.
func (o *CloudControllerManagerOptions) ApplyTo(c *config.Config, allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string, userAgent string) error {
	var err error

	// Build kubeconfig first to so that if it fails, it doesn't cause leaking
	// goroutines (started from initializing secure serving - which underneath
	// creates a queue which in its constructor starts a goroutine).
	c.Kubeconfig, err = clientcmd.BuildConfigFromFlags(o.Master, o.Generic.ClientConnection.Kubeconfig)
	if err != nil {
		return err
	}
	c.Kubeconfig.DisableCompression = true
	c.Kubeconfig.ContentConfig.AcceptContentTypes = o.Generic.ClientConnection.AcceptContentTypes
	c.Kubeconfig.ContentConfig.ContentType = o.Generic.ClientConnection.ContentType
	c.Kubeconfig.QPS = o.Generic.ClientConnection.QPS
	c.Kubeconfig.Burst = int(o.Generic.ClientConnection.Burst)

	if err = o.Generic.ApplyTo(&c.ComponentConfig.Generic, allControllers, disabledByDefaultControllers, controllerAliases); err != nil {
		return err
	}
	if err = o.KubeCloudShared.ApplyTo(&c.ComponentConfig.KubeCloudShared); err != nil {
		return err
	}
	if err = o.ServiceController.ApplyTo(&c.ComponentConfig.ServiceController); err != nil {
		return err
	}
	if o.Webhook != nil {
		if err = o.Webhook.ApplyTo(&c.ComponentConfig.Webhook); err != nil {
			return err
		}
	}
	if o.WebhookServing != nil {
		if err = o.WebhookServing.ApplyTo(&c.WebhookSecureServing, c.ComponentConfig.Webhook); err != nil {
			return err
		}
	}
	if err = o.SecureServing.ApplyTo(&c.SecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}
	if o.SecureServing.BindPort != 0 || o.SecureServing.Listener != nil {
		if err = o.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
			return err
		}
		if err = o.Authorization.ApplyTo(&c.Authorization); err != nil {
			return err
		}
	}

	c.Client, err = clientset.NewForConfig(restclient.AddUserAgent(c.Kubeconfig, userAgent))
	if err != nil {
		return err
	}

	c.EventBroadcaster = record.NewBroadcaster(record.WithContext(context.TODO())) // TODO: move broadcaster construction to a place where there is a proper context.
	c.EventRecorder = c.EventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: userAgent})

	rootClientBuilder := clientbuilder.SimpleControllerClientBuilder{
		ClientConfig: c.Kubeconfig,
	}
	if c.ComponentConfig.KubeCloudShared.UseServiceAccountCredentials {
		c.ClientBuilder = clientbuilder.NewDynamicClientBuilder(
			restclient.AnonymousClientConfig(c.Kubeconfig),
			c.Client.CoreV1(),
			metav1.NamespaceSystem)
	} else {
		c.ClientBuilder = rootClientBuilder
	}
	c.VersionedClient = rootClientBuilder.ClientOrDie("shared-informers")
	c.SharedInformers = informers.NewSharedInformerFactory(c.VersionedClient, resyncPeriod(c)())

	// sync back to component config
	// TODO: find more elegant way than syncing back the values.
	c.ComponentConfig.NodeStatusUpdateFrequency = o.NodeStatusUpdateFrequency
	c.ComponentConfig.NodeController.ConcurrentNodeSyncs = o.NodeController.ConcurrentNodeSyncs

	return nil
}

// Validate is used to validate config before launching the cloud controller manager
func (o *CloudControllerManagerOptions) Validate(allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string, allWebhooks, disabledByDefaultWebhooks []string) error {
	errors := []error{}

	errors = append(errors, o.Generic.Validate(allControllers, disabledByDefaultControllers, controllerAliases)...)
	errors = append(errors, o.KubeCloudShared.Validate()...)
	errors = append(errors, o.ServiceController.Validate()...)
	errors = append(errors, o.SecureServing.Validate()...)
	errors = append(errors, o.Authentication.Validate()...)
	errors = append(errors, o.Authorization.Validate()...)

	if o.Webhook != nil {
		errors = append(errors, o.Webhook.Validate(allWebhooks, disabledByDefaultWebhooks)...)
	}
	if o.WebhookServing != nil {
		errors = append(errors, o.WebhookServing.Validate()...)

		if o.WebhookServing.BindPort == o.SecureServing.BindPort && o.WebhookServing.BindPort != 0 {
			errors = append(errors, fmt.Errorf("--webhook-secure-port cannot be the same value as --secure-port"))
		}
	}
	if len(o.KubeCloudShared.CloudProvider.Name) == 0 {
		errors = append(errors, fmt.Errorf("--cloud-provider cannot be empty"))
	}

	return utilerrors.NewAggregate(errors)
}

// resyncPeriod computes the time interval a shared informer waits before resyncing with the api server
func resyncPeriod(c *config.Config) func() time.Duration {
	return func() time.Duration {
		factor := rand.Float64() + 1
		return time.Duration(float64(c.ComponentConfig.Generic.MinResyncPeriod.Nanoseconds()) * factor)
	}
}

// Config return a cloud controller manager config objective
func (o *CloudControllerManagerOptions) Config(allControllers []string, disabledByDefaultControllers []string, controllerAliases map[string]string, allWebhooks, disabledByDefaultWebhooks []string) (*config.Config, error) {
	if err := o.Validate(allControllers, disabledByDefaultControllers, controllerAliases, allWebhooks, disabledByDefaultWebhooks); err != nil {
		return nil, err
	}

	if err := o.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{netutils.ParseIPSloppy("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	if o.WebhookServing != nil {
		if err := o.WebhookServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{netutils.ParseIPSloppy("127.0.0.1")}); err != nil {
			return nil, fmt.Errorf("error creating self-signed certificates for webhook: %v", err)
		}
	}

	c := &config.Config{}
	if err := o.ApplyTo(c, allControllers, disabledByDefaultControllers, controllerAliases, CloudControllerManagerUserAgent); err != nil {
		return nil, err
	}

	return c, nil
}
