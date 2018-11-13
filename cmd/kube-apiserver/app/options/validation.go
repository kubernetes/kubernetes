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

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"
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

func validateTokenRequest(options *ServerRunOptions) []error {
	errors := []error{}

	enableAttempted := options.ServiceAccountSigningKeyFile != "" ||
		options.Authentication.ServiceAccounts.Issuer != "" ||
		len(options.Authentication.APIAudiences) != 0

	enableSucceeded := options.ServiceAccountIssuer != nil

	if enableAttempted && !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		errors = append(errors, fmt.Errorf("the TokenRequest feature is not enabled but --service-account-signing-key-file, --service-account-issuer and/or --api-audiences flags were passed"))
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) && !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		errors = append(errors, fmt.Errorf("the BoundServiceAccountTokenVolume feature depends on the TokenRequest feature, but the TokenRequest features is not enabled"))
	}

	if !enableAttempted && utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) {
		errors = append(errors, fmt.Errorf("--service-account-signing-key-file and --service-account-issuer are required flags"))
	}

	if enableAttempted && !enableSucceeded {
		errors = append(errors, fmt.Errorf("--service-account-signing-key-file, --service-account-issuer, and --api-audiences should be specified together"))
	}

	return errors
}

// Validate checks ServerRunOptions and return a slice of found errors.
func (s *ServerRunOptions) Validate() []error {
	var errors []error
	if errs := s.Etcd.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := validateClusterIPFlags(s); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := validateServiceNodePort(s); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.SecureServing.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.Authentication.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.Authorization.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.Audit.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.Admission.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := s.InsecureServing.Validate(); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if s.MasterCount <= 0 {
		errors = append(errors, fmt.Errorf("--apiserver-count should be a positive number, but value '%d' provided", s.MasterCount))
	}
	if errs := s.APIEnablement.Validate(legacyscheme.Scheme, apiextensionsapiserver.Scheme, aggregatorscheme.Scheme); len(errs) > 0 {
		errors = append(errors, errs...)
	}
	if errs := validateTokenRequest(s); len(errs) > 0 {
		errors = append(errors, errs...)
	}

	return errors
}
