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
	"fmt"
	"net"
	"strconv"

	"github.com/spf13/pflag"

	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	schedulerappconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

// CombinedInsecureServingOptions sets up to two insecure listeners for healthz and metrics. The flags
// override the ComponentConfig and DeprecatedInsecureServingOptions values for both.
type CombinedInsecureServingOptions struct {
	Healthz *apiserveroptions.DeprecatedInsecureServingOptionsWithLoopback
	Metrics *apiserveroptions.DeprecatedInsecureServingOptionsWithLoopback

	BindPort    int    // overrides the structs above on ApplyTo, ignored on ApplyToFromLoadedConfig
	BindAddress string // overrides the structs above on ApplyTo, ignored on ApplyToFromLoadedConfig
}

// AddFlags adds flags for the insecure serving options.
func (o *CombinedInsecureServingOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.BindAddress, "address", o.BindAddress, "DEPRECATED: the IP address on which to listen for the --port port (set to 0.0.0.0 for all IPv4 interfaces and :: for all IPv6 interfaces). See --bind-address instead.")
	// MarkDeprecated hides the flag from the help. We don't want that:
	// fs.MarkDeprecated("address", "see --bind-address instead.")
	fs.IntVar(&o.BindPort, "port", o.BindPort, "DEPRECATED: the port on which to serve HTTP insecurely without authentication and authorization. If 0, don't serve plain HTTP at all. See --secure-port instead.")
	// MarkDeprecated hides the flag from the help. We don't want that:
	// fs.MarkDeprecated("port", "see --secure-port instead.")
}

func (o *CombinedInsecureServingOptions) applyTo(c *schedulerappconfig.Config, componentConfig *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	if err := updateAddressFromDeprecatedInsecureServingOptions(&componentConfig.HealthzBindAddress, o.Healthz); err != nil {
		return err
	}
	if err := updateAddressFromDeprecatedInsecureServingOptions(&componentConfig.MetricsBindAddress, o.Metrics); err != nil {
		return err
	}

	if err := o.Healthz.ApplyTo(&c.InsecureServing, &c.LoopbackClientConfig); err != nil {
		return err
	}
	if o.Metrics != nil && (c.ComponentConfig.MetricsBindAddress != c.ComponentConfig.HealthzBindAddress || o.Healthz == nil) {
		if err := o.Metrics.ApplyTo(&c.InsecureMetricsServing, &c.LoopbackClientConfig); err != nil {
			return err
		}
	}

	return nil
}

// ApplyTo applies the insecure serving options to the given scheduler app configuration, and updates the componentConfig.
func (o *CombinedInsecureServingOptions) ApplyTo(c *schedulerappconfig.Config, componentConfig *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	if o == nil {
		componentConfig.HealthzBindAddress = ""
		componentConfig.MetricsBindAddress = ""
		return nil
	}

	if o.Healthz != nil {
		o.Healthz.BindPort = o.BindPort
		o.Healthz.BindAddress = net.ParseIP(o.BindAddress)
	}
	if o.Metrics != nil {
		o.Metrics.BindPort = o.BindPort
		o.Metrics.BindAddress = net.ParseIP(o.BindAddress)
	}

	return o.applyTo(c, componentConfig)
}

// ApplyToFromLoadedConfig updates the insecure serving options from the component config and then applies it to the given scheduler app configuration.
func (o *CombinedInsecureServingOptions) ApplyToFromLoadedConfig(c *schedulerappconfig.Config, componentConfig *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	if o == nil {
		return nil
	}

	if err := updateDeprecatedInsecureServingOptionsFromAddress(o.Healthz, componentConfig.HealthzBindAddress); err != nil {
		return fmt.Errorf("invalid healthz address: %v", err)
	}
	if err := updateDeprecatedInsecureServingOptionsFromAddress(o.Metrics, componentConfig.MetricsBindAddress); err != nil {
		return fmt.Errorf("invalid metrics address: %v", err)
	}

	return o.applyTo(c, componentConfig)
}

func updateAddressFromDeprecatedInsecureServingOptions(addr *string, is *apiserveroptions.DeprecatedInsecureServingOptionsWithLoopback) error {
	if is == nil {
		*addr = ""
	} else {
		if is.Listener != nil {
			*addr = is.Listener.Addr().String()
		} else if is.BindPort == 0 {
			*addr = ""
		} else {
			*addr = net.JoinHostPort(is.BindAddress.String(), strconv.Itoa(is.BindPort))
		}
	}

	return nil
}

func updateDeprecatedInsecureServingOptionsFromAddress(is *apiserveroptions.DeprecatedInsecureServingOptionsWithLoopback, addr string) error {
	if is == nil {
		return nil
	}
	if len(addr) == 0 {
		is.BindPort = 0
		return nil
	}

	host, portInt, err := splitHostIntPort(addr)
	if err != nil {
		return fmt.Errorf("invalid address %q", addr)
	}

	is.BindAddress = net.ParseIP(host)
	is.BindPort = portInt

	return nil
}

// Validate validates the insecure serving options.
func (o *CombinedInsecureServingOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errors := []error{}

	if o.BindPort < 0 || o.BindPort > 65535 {
		errors = append(errors, fmt.Errorf("--port %v must be between 0 and 65535, inclusive. 0 for turning off insecure (HTTP) port", o.BindPort))
	}

	if len(o.BindAddress) > 0 && net.ParseIP(o.BindAddress) == nil {
		errors = append(errors, fmt.Errorf("--address %v is an invalid IP address", o.BindAddress))
	}

	return errors
}
