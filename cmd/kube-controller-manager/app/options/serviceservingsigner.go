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
	"fmt"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/validation"
	podcertssignerconfig "k8s.io/kubernetes/pkg/controller/certificates/podcertssigner/config"
)

// ServiceServingSignerOptions holds the PodCertsSignerConfiguration options for the kube service CA signer.
type ServiceServingSignerOptions struct {
	*podcertssignerconfig.ServiceServingSignerConfiguration
}

// AddFlags adds flags related to ServiceCASignerControllers for controller manager to the specified FlagSet
func (o *ServiceServingSignerOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}

	if len(o.CABundlePath) == 0 {
		o.ClusterServiceDomain = "cluster.local"
	}

	fs.StringVar(&o.CACertPath, "service-serving-cert", o.CACertPath, "Path to the signer cert for the service serving signer. Expects PEM format. If there are multiple certificates in the bundle, the first one is used.")
	fs.StringVar(&o.CAKeyPath, "service-serving-key", o.CAKeyPath, "Path to the signer key for the service serving signer. Expects a single key in PEM format.")
	fs.StringVar(&o.CABundlePath, "service-serving-trust-bundle", o.CABundlePath, "Path to the trust bundle for the service serving signer. Used for trust distrubution within the cluster. Expects PEM format. If unset, the file given in --service-serving-cert is used as the trust bundle.")
	fs.StringVar(&o.ClusterServiceDomain, "cluster-service-domain", o.ClusterServiceDomain, "Cluster's service domain name. Defaults to 'cluster.local'.")
}

// ApplyTo fills up PodCertsSignerConfiguration config with options.
func (o *ServiceServingSignerOptions) ApplyTo(cfg *podcertssignerconfig.ServiceServingSignerConfiguration) error {
	if o == nil {
		return nil
	}

	cfg.CACertPath = o.CACertPath
	cfg.CAKeyPath = o.CAKeyPath
	cfg.CABundlePath = o.CABundlePath
	cfg.ClusterServiceDomain = o.ClusterServiceDomain

	return nil
}

// Validate checks validation of ServiceAccountControllerOptions.
func (o *ServiceServingSignerOptions) Validate() []error {
	if o == nil {
		return nil
	}

	errs := []error{}

	// if both are unset, assume the controller is disabled
	if (len(o.CACertPath) == 0) != (len(o.CAKeyPath) == 0) {
		errs = append(errs, fmt.Errorf("service-serving-cert and service-serving-key must be set together"))
	}

	if len(o.ClusterServiceDomain) == 0 {
		errs = append(errs, fmt.Errorf("cluster-service-domain must not be empty"))
	} else {
		for _, msg := range validation.IsDNS1123Subdomain(o.ClusterServiceDomain) {
			errs = append(errs, fmt.Errorf("cluster-service-domain must be a valid DNS subdomain, but got %q: %s", o.ClusterServiceDomain, msg))
		}
	}

	return errs
}
