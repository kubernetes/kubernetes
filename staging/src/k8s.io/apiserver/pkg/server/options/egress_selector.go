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
	"k8s.io/utils/path"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/egressselector"
)

// EgressSelectorOptions holds the api server egress selector options.
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/1281-network-proxy/README.md
type EgressSelectorOptions struct {
	// ConfigFile is the file path with api-server egress selector configuration.
	ConfigFile string
}

// NewEgressSelectorOptions creates a new instance of EgressSelectorOptions
//
// The option is to point to a configuration file for egress/konnectivity.
// This determines which types of requests use egress/konnectivity and how they use it.
// If empty the API Server will attempt to connect directly using the network.
func NewEgressSelectorOptions() *EgressSelectorOptions {
	return &EgressSelectorOptions{}
}

// AddFlags adds flags related to admission for a specific APIServer to the specified FlagSet
func (o *EgressSelectorOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ConfigFile, "egress-selector-config-file", o.ConfigFile,
		"File with apiserver egress selector configuration.")
}

// ApplyTo adds the egress selector settings to the server configuration.
// In case egress selector settings were not provided by a cluster-admin
// they will be prepared from the recommended/default/no-op values.
func (o *EgressSelectorOptions) ApplyTo(c *server.Config) error {
	if o == nil {
		return nil
	}

	npConfig, err := egressselector.ReadEgressSelectorConfiguration(o.ConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read egress selector config: %v", err)
	}
	errs := egressselector.ValidateEgressSelectorConfiguration(npConfig)
	if len(errs) > 0 {
		return fmt.Errorf("failed to validate egress selector configuration: %v", errs.ToAggregate())
	}

	cs, err := egressselector.NewEgressSelector(npConfig)
	if err != nil {
		return fmt.Errorf("failed to setup egress selector with config %#v: %v", npConfig, err)
	}
	c.EgressSelector = cs
	return nil
}

// Validate verifies flags passed to EgressSelectorOptions.
func (o *EgressSelectorOptions) Validate() []error {
	if o == nil || o.ConfigFile == "" {
		return nil
	}

	errs := []error{}

	if exists, err := path.Exists(path.CheckFollowSymlink, o.ConfigFile); !exists || err != nil {
		errs = append(errs, fmt.Errorf("egress-selector-config-file %s does not exist", o.ConfigFile))
	}

	return errs
}
