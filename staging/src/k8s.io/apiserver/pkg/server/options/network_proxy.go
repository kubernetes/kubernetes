/*
Copyright 2019 The Kubernetes Authors.
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

	"github.com/spf13/pflag"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/networkproxy"
)

// NetworkProxyOptions holds the api server network proxy options.
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190226-network-proxy.md
type NetworkProxyOptions struct {
	// ConfigFile is the file path with api-server network proxy configuration.
	ConfigFile string
}

// NewNetworkProxyOptions creates a new instance of NetworkProxyOptions
//
//  Provides the list of XXX that holds sane values
//  that can be used by servers that don't care about the network proxy.
//  Servers that do care can overwrite/append that field after creation.
func NewNetworkProxyOptions() *NetworkProxyOptions {
	options := &NetworkProxyOptions{

	}
	// server.RegisterAllAdmissionPlugins(options.Plugins) // TODO: WRF
	return options
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (o *NetworkProxyOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ConfigFile, "network-proxy-config-file", o.ConfigFile,
		"File with apiserver network proxy configuration.")
}

// ApplyTo adds the network proxy settings to the server configuration.
// In case network proxy settings were not provided by a custer-admin
// they will be prepared from the recommended/default/no-op values.
func (o *NetworkProxyOptions) ApplyTo(c *server.Config) error {
	if o == nil {
		return nil
	}

	npConfig, err := networkproxy.ReadNetworkProxyConfiguration(o.ConfigFile, configScheme)
	if err != nil {
		return fmt.Errorf("failed to read network proxy config: %v", err)
	}

	server.SetupConnectivityService(npConfig)
	return nil
}

// Validate verifies flags passed to NetworkProxyOptions.
func (o *NetworkProxyOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	return errs
}
