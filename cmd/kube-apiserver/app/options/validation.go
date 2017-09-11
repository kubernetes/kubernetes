/*
Copyright 2014 The Kubernetes Authors.

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
)

// TODO: Longer term we should read this from some config store, rather than a flag.
func validateClusterIPFlags(options *ServerRunOptions) []error {
	errors := []error{}
	if options.ServiceClusterIPRange.IP == nil {
		errors = append(errors, fmt.Errorf("no --service-cluster-ip-range specified"))
	}
	var ones, bits = options.ServiceClusterIPRange.Mask.Size()
	if bits-ones > 20 {
		errors = append(errors, fmt.Errorf("specified --service-cluster-ip-range is too large"))
	}
	return errors
}

func validateServiceNodePort(options *ServerRunOptions) []error {
	errors := []error{}
	if options.KubernetesServiceNodePort < 0 || options.KubernetesServiceNodePort > 65535 {
		errors = append(errors, fmt.Errorf("--kubernetes-service-node-port %v must be between 0 and 65535, inclusive. If 0, the Kubernetes master service will be of type ClusterIP", options.KubernetesServiceNodePort))
	}

	if options.KubernetesServiceNodePort > 0 && !options.ServiceNodePortRange.Contains(options.KubernetesServiceNodePort) {
		errors = append(errors, fmt.Errorf("kubernetes service port range %v doesn't contain %v", options.ServiceNodePortRange, (options.KubernetesServiceNodePort)))
	}
	return errors
}

// Validate checks ServerRunOptions and return a slice of found errors.
func (options *ServerRunOptions) Validate() []error {
	var errors []error
	if errs := options.Etcd.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := validateClusterIPFlags(options); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := validateServiceNodePort(options); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := options.SecureServing.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := options.Authentication.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := options.Audit.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := options.InsecureServing.Validate("insecure-port"); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if options.MasterCount <= 0 {
		errors = append(errors, fmt.Errorf("--apiserver-count should be a positive number, but value '%d' provided", options.MasterCount))
	}
	return errors
}
