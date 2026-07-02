/*
Copyright 2026 The Kubernetes Authors.

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
	"github.com/spf13/pflag"

	podcertssignerconfig "k8s.io/kubernetes/pkg/controller/certificates/podcertssigner/config"
)

// ServiceCAControllerOptions holds the PodCertsSignerConfiguration options for the kube service CA signer.
type ServiceCAControllerOptions struct {
	*podcertssignerconfig.PodCertsSignerConfiguration
}

// AddFlags adds flags related to ServiceCASignerControllers for controller manager to the specified FlagSet
func (o *ServiceCAControllerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	fs.StringVar(&o.ServiceCACertPath, "service-ca-cert", o.ServiceCACertPath, "FIXME:")
	fs.StringVar(&o.ServiceCAKeyPath, "service-ca-key", o.ServiceCAKeyPath, "FIXME:")
	fs.StringVar(&o.ClusterServiceDomain, "cluster-service-domain", o.ClusterServiceDomain, "FIXME:")
}

// ApplyTo fills up PodCertsSignerConfiguration config with options.
func (o *ServiceCAControllerOptions) ApplyTo(cfg *podcertssignerconfig.PodCertsSignerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.ServiceCACertPath = o.ServiceCACertPath
	cfg.ServiceCAKeyPath = o.ServiceCAKeyPath
	cfg.ClusterServiceDomain = o.ClusterServiceDomain

	return nil
}

// Validate checks validation of ServiceAccountControllerOptions.
func (o *ServiceCAControllerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	// FIXME: validate both cert/key are set if at least one is set, validate
	// cluster domain is set

	errs := []error{}
	return errs
}
