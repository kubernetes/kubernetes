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
	"errors"
	"fmt"

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/features"
)

// TODO: Longer term we should read this from some config store, rather than a flag.
func validateClusterIPFlags(options *ServerRunOptions) []error {
	var errs []error

	if options.ServiceClusterIPRange.IP == nil {
		errs = append(errs, errors.New("no --service-cluster-ip-range specified"))
	}
	var ones, bits = options.ServiceClusterIPRange.Mask.Size()
	if bits-ones > 20 {
		errs = append(errs, errors.New("specified --service-cluster-ip-range is too large"))
	}

	return errs
}

func validateServiceNodePort(options *ServerRunOptions) []error {
	var errs []error

	if options.KubernetesServiceNodePort < 0 || options.KubernetesServiceNodePort > 65535 {
		errs = append(errs, fmt.Errorf("--kubernetes-service-node-port %v must be between 0 and 65535, inclusive. If 0, the Kubernetes master service will be of type ClusterIP", options.KubernetesServiceNodePort))
	}

	if options.KubernetesServiceNodePort > 0 && !options.ServiceNodePortRange.Contains(options.KubernetesServiceNodePort) {
		errs = append(errs, fmt.Errorf("kubernetes service port range %v doesn't contain %v", options.ServiceNodePortRange, (options.KubernetesServiceNodePort)))
	}
	return errs
}

func validateTokenRequest(options *ServerRunOptions) []error {
	var errs []error

	enableAttempted := options.ServiceAccountSigningKeyFile != "" ||
		options.Authentication.ServiceAccounts.Issuer != "" ||
		len(options.Authentication.APIAudiences) != 0

	enableSucceeded := options.ServiceAccountIssuer != nil

	if enableAttempted && !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		errs = append(errs, errors.New("the TokenRequest feature is not enabled but --service-account-signing-key-file, --service-account-issuer and/or --api-audiences flags were passed"))
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) && !utilfeature.DefaultFeatureGate.Enabled(features.TokenRequest) {
		errs = append(errs, errors.New("the BoundServiceAccountTokenVolume feature depends on the TokenRequest feature, but the TokenRequest features is not enabled"))
	}

	if !enableAttempted && utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) {
		errs = append(errs, errors.New("--service-account-signing-key-file and --service-account-issuer are required flags"))
	}

	if enableAttempted && !enableSucceeded {
		errs = append(errs, errors.New("--service-account-signing-key-file, --service-account-issuer, and --api-audiences should be specified together"))
	}

	return errs
}

// Validate checks ServerRunOptions and return a slice of found errs.
func (s *ServerRunOptions) Validate() []error {
	var errs []error
	if s.MasterCount <= 0 {
		errs = append(errs, fmt.Errorf("--apiserver-count should be a positive number, but value '%d' provided", s.MasterCount))
	}
	errs = append(errs, s.Etcd.Validate()...)
	errs = append(errs, validateClusterIPFlags(s)...)
	errs = append(errs, validateServiceNodePort(s)...)
	errs = append(errs, s.SecureServing.Validate()...)
	errs = append(errs, s.Authentication.Validate()...)
	errs = append(errs, s.Authorization.Validate()...)
	errs = append(errs, s.Audit.Validate()...)
	errs = append(errs, s.Admission.Validate()...)
	errs = append(errs, s.InsecureServing.Validate()...)
	errs = append(errs, s.APIEnablement.Validate(legacyscheme.Scheme, apiextensionsapiserver.Scheme, aggregatorscheme.Scheme)...)
	errs = append(errs, validateTokenRequest(s)...)

	return errs
}
