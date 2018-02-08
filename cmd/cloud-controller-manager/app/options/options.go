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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	cloudcontrollerconfig "k8s.io/kubernetes/cmd/cloud-controller-manager/app/config"
	cmoptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
	"k8s.io/kubernetes/pkg/master/ports"

	// add the kubernetes feature gates
	_ "k8s.io/kubernetes/pkg/features"

	"github.com/spf13/pflag"
)

// CloudControllerManagerOptions is the main context object for the controller manager.
type CloudControllerManagerOptions struct {
	Generic cmoptions.GenericControllerManagerOptions

	// NodeStatusUpdateFrequency is the frequency at which the controller updates nodes' status
	NodeStatusUpdateFrequency metav1.Duration
}

// NewCloudControllerManagerOptions creates a new ExternalCMServer with a default config.
func NewCloudControllerManagerOptions() *CloudControllerManagerOptions {
	componentConfig := cmoptions.NewDefaultControllerManagerComponentConfig(ports.InsecureCloudControllerManagerPort)

	s := CloudControllerManagerOptions{
		// The common/default are kept in 'cmd/kube-controller-manager/app/options/util.go'.
		// Please make common changes there and put anything cloud specific here.
		Generic:                   cmoptions.NewGenericControllerManagerOptions(componentConfig),
		NodeStatusUpdateFrequency: metav1.Duration{Duration: 5 * time.Minute},
	}
	s.Generic.ComponentConfig.LeaderElection.LeaderElect = true

	s.Generic.SecureServing.ServerCert.CertDirectory = "/var/run/kubernetes"
	s.Generic.SecureServing.ServerCert.PairName = "cloud-controller-manager"

	return &s
}

// AddFlags adds flags for a specific ExternalCMServer to the specified FlagSet
func (o *CloudControllerManagerOptions) AddFlags(fs *pflag.FlagSet) {
	o.Generic.AddFlags(fs)

	fs.StringVar(&o.Generic.ComponentConfig.CloudProvider, "cloud-provider", o.Generic.ComponentConfig.CloudProvider, "The provider of cloud services. Cannot be empty.")
	fs.DurationVar(&o.NodeStatusUpdateFrequency.Duration, "node-status-update-frequency", o.NodeStatusUpdateFrequency.Duration, "Specifies how often the controller updates nodes' status.")
	// TODO: remove --service-account-private-key-file 6 months after 1.8 is released (~1.10)
	fs.StringVar(&o.Generic.ComponentConfig.ServiceAccountKeyFile, "service-account-private-key-file", o.Generic.ComponentConfig.ServiceAccountKeyFile, "Filename containing a PEM-encoded private RSA or ECDSA key used to sign service account tokens.")
	fs.MarkDeprecated("service-account-private-key-file", "This flag is currently no-op and will be deleted.")
	fs.Int32Var(&o.Generic.ComponentConfig.ConcurrentServiceSyncs, "concurrent-service-syncs", o.Generic.ComponentConfig.ConcurrentServiceSyncs, "The number of services that are allowed to sync concurrently. Larger number = more responsive service management, but more CPU (and network) load")

	leaderelectionconfig.BindFlags(&o.Generic.ComponentConfig.LeaderElection, fs)

	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo fills up cloud controller manager config with options.
func (o *CloudControllerManagerOptions) ApplyTo(c *cloudcontrollerconfig.Config) error {
	if err := o.Generic.ApplyTo(&c.Generic, "cloud-controller-manager"); err != nil {
		return err
	}

	c.Extra.NodeStatusUpdateFrequency = o.NodeStatusUpdateFrequency.Duration

	return nil
}

// Validate is used to validate config before launching the cloud controller manager
func (o *CloudControllerManagerOptions) Validate() error {
	errors := []error{}
	errors = append(errors, o.Generic.Validate()...)

	if len(o.Generic.ComponentConfig.CloudProvider) == 0 {
		errors = append(errors, fmt.Errorf("--cloud-provider cannot be empty"))
	}

	return utilerrors.NewAggregate(errors)
}

// Config return a cloud controller manager config objective
func (o CloudControllerManagerOptions) Config() (*cloudcontrollerconfig.Config, error) {
	if err := o.Validate(); err != nil {
		return nil, err
	}

	c := &cloudcontrollerconfig.Config{}
	if err := o.ApplyTo(c); err != nil {
		return nil, err
	}

	return c, nil
}
