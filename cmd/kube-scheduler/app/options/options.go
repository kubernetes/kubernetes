/*
Copyright 2018 The Kubernetes Authors.

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
	"strconv"

	"github.com/golang/glog"
	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/runtime"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	controlleroptions "k8s.io/kubernetes/cmd/controller-manager/app/options"
	schedulerappconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/client/leaderelectionconfig"
)

// Options has all the params needed to run a Scheduler
type Options struct {
	// The default values. These are overridden if ConfigFile is set or by values in InsecureServing.
	ComponentConfig componentconfig.KubeSchedulerConfiguration

	SecureServing           *apiserveroptions.SecureServingOptions
	CombinedInsecureServing *CombinedInsecureServingOptions
	Authentication          *apiserveroptions.DelegatingAuthenticationOptions
	Authorization           *apiserveroptions.DelegatingAuthorizationOptions
	Deprecated              *DeprecatedOptions

	// ConfigFile is the location of the scheduler server's configuration file.
	ConfigFile string
	Master     string
}

// NewOptions returns default scheduler app options.
func NewOptions() (*Options, error) {
	cfg, err := newDefaultComponentConfig()
	if err != nil {
		return nil, err
	}

	hhost, hport, err := splitHostIntPort(cfg.HealthzBindAddress)
	if err != nil {
		return nil, err
	}

	o := &Options{
		ComponentConfig: *cfg,
		SecureServing:   nil, // TODO: enable with apiserveroptions.NewSecureServingOptions()
		CombinedInsecureServing: &CombinedInsecureServingOptions{
			Healthz: &controlleroptions.InsecureServingOptions{
				BindNetwork: "tcp",
			},
			Metrics: &controlleroptions.InsecureServingOptions{
				BindNetwork: "tcp",
			},
			BindPort:    hport,
			BindAddress: hhost,
		},
		Authentication: nil, // TODO: enable with apiserveroptions.NewDelegatingAuthenticationOptions()
		Authorization:  nil, // TODO: enable with apiserveroptions.NewDelegatingAuthorizationOptions()
		Deprecated:     &DeprecatedOptions{},
	}

	return o, nil
}

func splitHostIntPort(s string) (string, int, error) {
	host, port, err := net.SplitHostPort(s)
	if err != nil {
		return "", 0, err
	}
	portInt, err := strconv.Atoi(port)
	if err != nil {
		return "", 0, err
	}
	return host, portInt, err
}

func newDefaultComponentConfig() (*componentconfig.KubeSchedulerConfiguration, error) {
	scheme := runtime.NewScheme()
	componentconfig.AddToScheme(scheme)
	componentconfigv1alpha1.AddToScheme(scheme)

	cfgv1alpha1 := componentconfigv1alpha1.KubeSchedulerConfiguration{}
	scheme.Default(&cfgv1alpha1)
	cfg := componentconfig.KubeSchedulerConfiguration{}
	if err := scheme.Convert(&cfgv1alpha1, &cfg, nil); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// AddFlags adds flags for the scheduler options.
func (o *Options) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file. Flags override values in this file.")
	fs.StringVar(&o.Master, "master", o.Master, "The address of the Kubernetes API server (overrides any value in kubeconfig)")

	o.SecureServing.AddFlags(fs)
	o.CombinedInsecureServing.AddFlags(fs)
	o.Authentication.AddFlags(fs)
	o.Authorization.AddFlags(fs)
	o.Deprecated.AddFlags(fs, &o.ComponentConfig)

	leaderelectionconfig.BindFlags(&o.ComponentConfig.LeaderElection.LeaderElectionConfiguration, fs)
	utilfeature.DefaultFeatureGate.AddFlag(fs)
}

// ApplyTo applies the scheduler options to the given scheduler app configuration.
func (o *Options) ApplyTo(c *schedulerappconfig.Config) error {
	if len(o.ConfigFile) == 0 {
		glog.Warning("WARNING: all flags other than --config are deprecated. Please begin using a config file ASAP.")

		c.ComponentConfig = o.ComponentConfig

		// only apply deprecated flags if no config file is loaded (this is the old behaviour).
		if err := o.Deprecated.ApplyTo(&c.ComponentConfig); err != nil {
			return err
		}
		if err := o.CombinedInsecureServing.ApplyTo(c, &c.ComponentConfig); err != nil {
			return err
		}
	} else {
		cfg, err := loadConfigFromFile(o.ConfigFile)
		if err != nil {
			return err
		}

		// use the loaded config file only, with the exception of --address and --port. This means that
		// none of the deprectated flags in o.Deprecated are taken into consideration. This is the old
		// behaviour of the flags we have to keep.
		c.ComponentConfig = *cfg

		if err := o.CombinedInsecureServing.ApplyToFromLoadedConfig(c, &c.ComponentConfig); err != nil {
			return err
		}
	}

	if err := o.SecureServing.ApplyTo(&c.SecureServing); err != nil {
		return err
	}
	if err := o.Authentication.ApplyTo(&c.Authentication, c.SecureServing, nil); err != nil {
		return err
	}
	return o.Authorization.ApplyTo(&c.Authorization)
}

// Validate validates all the required options.
func (o *Options) Validate() []error {
	var errs []error

	errs = append(errs, o.SecureServing.Validate()...)
	errs = append(errs, o.CombinedInsecureServing.Validate()...)
	errs = append(errs, o.Authentication.Validate()...)
	errs = append(errs, o.Authorization.Validate()...)
	errs = append(errs, o.Deprecated.Validate()...)

	return errs
}
