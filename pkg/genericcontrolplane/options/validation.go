/*
Copyright 2022 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/genericcontrolplanescheme"
)

func validateAPIServerIdentity(options *CompletedServerRunOptions) []error {
	var errs []error
	if options.IdentityLeaseDurationSeconds <= 0 {
		errs = append(errs, fmt.Errorf("--identity-lease-duration-seconds should be a positive number, but value '%d' provided", options.IdentityLeaseDurationSeconds))
	}
	if options.IdentityLeaseRenewIntervalSeconds <= 0 {
		errs = append(errs, fmt.Errorf("--identity-lease-renew-interval-seconds should be a positive number, but value '%d' provided", options.IdentityLeaseRenewIntervalSeconds))
	}
	return errs
}

// Validate checks Options and return a slice of found errs.
func (s *CompletedServerRunOptions) Validate() []error {
	var errs []error
	errs = append(errs, s.Etcd.Validate()...)
	errs = append(errs, s.SecureServing.Validate()...)
	errs = append(errs, s.Authentication.Validate()...)
	errs = append(errs, s.Audit.Validate()...)
	errs = append(errs, s.Admission.Validate()...)
	errs = append(errs, s.APIEnablement.Validate(genericcontrolplanescheme.Scheme, apiextensionsapiserver.Scheme)...)
	errs = append(errs, s.Metrics.Validate()...)
	errs = append(errs, validateAPIServerIdentity(s)...)

	return errs
}
