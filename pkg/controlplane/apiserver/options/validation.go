/*
Copyright 2023 The Kubernetes Authors.

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
	"strings"

	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

func validateTokenRequest(options *Options) []error {
	var errs []error

	enableAttempted := options.ServiceAccountSigningKeyFile != "" ||
		(len(options.Authentication.ServiceAccounts.Issuers) != 0 && options.Authentication.ServiceAccounts.Issuers[0] != "") ||
		len(options.Authentication.APIAudiences) != 0

	enableSucceeded := options.ServiceAccountIssuer != nil

	if !enableAttempted {
		errs = append(errs, errors.New("--service-account-signing-key-file and --service-account-issuer are required flags"))
	}

	if enableAttempted && !enableSucceeded {
		errs = append(errs, errors.New("--service-account-signing-key-file, --service-account-issuer, and --api-audiences should be specified together"))
	}

	return errs
}

func validateAPIPriorityAndFairness(options *Options) []error {
	if options.Features.EnablePriorityAndFairness {
		// If none of the following runtime config options are specified,
		// APF is assumed to be turned on. The internal APF controller uses
		// v1 so it should be enabled.
		enabledAPIString := options.APIEnablement.RuntimeConfig.String()
		testConfigs := []string{"flowcontrol.apiserver.k8s.io/v1", "api/ga", "api/all"} // in the order of precedence
		for _, testConfig := range testConfigs {
			if strings.Contains(enabledAPIString, fmt.Sprintf("%s=false", testConfig)) {
				return []error{fmt.Errorf("--runtime-config=%s=false conflicts with --enable-priority-and-fairness=true", testConfig)}
			}
			if strings.Contains(enabledAPIString, fmt.Sprintf("%s=true", testConfig)) {
				return nil
			}
		}
	}

	return nil
}

func validateNodeSelectorAuthorizationFeature() []error {
	if utilfeature.DefaultFeatureGate.Enabled(features.AuthorizeNodeWithSelectors) && !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		return []error{fmt.Errorf("AuthorizeNodeWithSelectors feature requires AuthorizeWithSelectors feature to be enabled")}
	}
	return nil
}

func validateDRAControlPlaneControllerFeature() []error {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAControlPlaneController) && !utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		return []error{fmt.Errorf("DRAControlPlaneController feature requires DynamicResourceAllocation feature to be enabled")}
	}
	return nil
}

func validateUnknownVersionInteroperabilityProxyFeature() []error {
	if utilfeature.DefaultFeatureGate.Enabled(features.UnknownVersionInteroperabilityProxy) {
		if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.StorageVersionAPI) {
			return nil
		}
		return []error{fmt.Errorf("UnknownVersionInteroperabilityProxy feature requires StorageVersionAPI feature flag to be enabled")}
	}
	return nil
}

func validateUnknownVersionInteroperabilityProxyFlags(options *Options) []error {
	err := []error{}
	if !utilfeature.DefaultFeatureGate.Enabled(features.UnknownVersionInteroperabilityProxy) {
		if options.PeerCAFile != "" {
			err = append(err, fmt.Errorf("--peer-ca-file requires UnknownVersionInteroperabilityProxy feature to be turned on"))
		}
		if options.PeerAdvertiseAddress.PeerAdvertiseIP != "" {
			err = append(err, fmt.Errorf("--peer-advertise-ip requires UnknownVersionInteroperabilityProxy feature to be turned on"))
		}
		if options.PeerAdvertiseAddress.PeerAdvertisePort != "" {
			err = append(err, fmt.Errorf("--peer-advertise-port requires UnknownVersionInteroperabilityProxy feature to be turned on"))
		}
	}
	return err
}

// Validate checks Options and return a slice of found errs.
func (s *Options) Validate() []error {
	var errs []error

	errs = append(errs, s.GenericServerRunOptions.Validate()...)
	errs = append(errs, s.Etcd.Validate()...)
	errs = append(errs, validateAPIPriorityAndFairness(s)...)
	errs = append(errs, s.SecureServing.Validate()...)
	errs = append(errs, s.Authentication.Validate()...)
	errs = append(errs, s.Authorization.Validate()...)
	errs = append(errs, s.Audit.Validate()...)
	errs = append(errs, s.Admission.Validate()...)
	errs = append(errs, s.APIEnablement.Validate(legacyscheme.Scheme, apiextensionsapiserver.Scheme, aggregatorscheme.Scheme)...)
	errs = append(errs, validateTokenRequest(s)...)
	errs = append(errs, s.Metrics.Validate()...)
	errs = append(errs, validateUnknownVersionInteroperabilityProxyFeature()...)
	errs = append(errs, validateUnknownVersionInteroperabilityProxyFlags(s)...)
	errs = append(errs, validateNodeSelectorAuthorizationFeature()...)
	errs = append(errs, validateDRAControlPlaneControllerFeature()...)

	return errs
}
