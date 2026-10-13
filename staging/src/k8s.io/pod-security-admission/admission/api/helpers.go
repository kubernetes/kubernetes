/*
Copyright 2021 The Kubernetes Authors.

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

package api

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/errors"
	policyapi "k8s.io/pod-security-admission/api"
)

var requiredErr = fmt.Errorf("required")

// TODO: deduplicate against PolicyToEvaluate
func ToPolicy(defaults PodSecurityDefaults) (policyapi.Policy, error) {
	var (
		err  error
		errs []error
		p    policyapi.Policy
	)

	if len(defaults.Enforce) == 0 {
		errs = appendErr(errs, requiredErr, "enforce")
	} else {
		p.Enforce.Level, err = policyapi.ParseLevel(defaults.Enforce)
		errs = appendErr(errs, err, "enforce")
	}

	if len(defaults.EnforceVersion) == 0 {
		errs = appendErr(errs, requiredErr, "enforce-version")
	} else {
		p.Enforce.Version, err = policyapi.ParseVersion(defaults.EnforceVersion)
		errs = appendErr(errs, err, "enforce-version")
	}

	if len(defaults.Audit) == 0 {
		errs = appendErr(errs, requiredErr, "audit")
	} else {
		p.Audit.Level, err = policyapi.ParseLevel(defaults.Audit)
		errs = appendErr(errs, err, "audit")
	}

	if len(defaults.AuditVersion) == 0 {
		errs = appendErr(errs, requiredErr, "audit-version")
	} else {
		p.Audit.Version, err = policyapi.ParseVersion(defaults.AuditVersion)
		errs = appendErr(errs, err, "audit-version")
	}

	if len(defaults.Warn) == 0 {
		errs = appendErr(errs, requiredErr, "warn")
	} else {
		p.Warn.Level, err = policyapi.ParseLevel(defaults.Warn)
		errs = appendErr(errs, err, "warn")
	}

	if len(defaults.WarnVersion) == 0 {
		errs = appendErr(errs, requiredErr, "warn-version")
	} else {
		p.Warn.Version, err = policyapi.ParseVersion(defaults.WarnVersion)
		errs = appendErr(errs, err, "warn-version")
	}

	return p, errors.NewAggregate(errs)
}

// appendErr is a helper function to collect field-specific errors.
func appendErr(errs []error, err error, field string) []error {
	if err != nil {
		return append(errs, fmt.Errorf("%s: %s", field, err.Error()))
	}
	return errs
}
