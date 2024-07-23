/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	"slices"
	"strings"

	"github.com/blang/semver/v4"
	"k8s.io/apimachinery/pkg/api/validation"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/kubernetes/pkg/apis/coordination"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
)

var validLeaseStrategies = []coordination.CoordinatedLeaseStrategy{coordination.OldestEmulationVersion}

// ValidateLease validates a Lease.
func ValidateLease(lease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseUpdate validates an update of Lease object.
func ValidateLeaseUpdate(lease, oldLease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&lease.ObjectMeta, &oldLease.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseSpec validates spec of Lease.
func ValidateLeaseSpec(spec *coordination.LeaseSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.LeaseDurationSeconds != nil && *spec.LeaseDurationSeconds <= 0 {
		fld := fldPath.Child("leaseDurationSeconds")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseDurationSeconds, "must be greater than 0"))
	}
	if spec.LeaseTransitions != nil && *spec.LeaseTransitions < 0 {
		fld := fldPath.Child("leaseTransitions")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseTransitions, "must be greater than or equal to 0"))
	}
	if spec.Strategy != nil {
		allErrs = append(allErrs, ValidateCoordinatedLeaseStrategy(*spec.Strategy, fldPath.Child("strategy"))...)
	}
	if spec.PreferredHolder != nil && *spec.PreferredHolder != "" && (spec.Strategy == nil || *spec.Strategy == "") {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("preferredHolder"), "may only be specified if `strategy` is defined"))
	}
	// spec.RenewTime is a MicroTime and doesn't need further validation
	return allErrs
}

// ValidateLeaseCandidate validates a LeaseCandidate.
func ValidateLeaseCandidate(lease *coordination.LeaseCandidate) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, ValidLeaseCandidateName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseCandidateSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidLeaseCandidateName(name string, prefix bool) []string {
	// prefix is already handled by IsConfigMapKey, a trailing - is permitted.
	return utilvalidation.IsConfigMapKey(name)
}

func ValidateLeaseCandidateSpecUpdate(leaseCandidateSpec, oldLeaseCandidateSpec *coordination.LeaseCandidateSpec) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(leaseCandidateSpec.LeaseName, oldLeaseCandidateSpec.LeaseName, field.NewPath("spec").Child("leaseName"))...)
	return allErrs
}

// ValidateLeaseCandidateUpdate validates an update of LeaseCandidate object.
func ValidateLeaseCandidateUpdate(leaseCandidate, oldLeaseCandidate *coordination.LeaseCandidate) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&leaseCandidate.ObjectMeta, &oldLeaseCandidate.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseCandidateSpec(&leaseCandidate.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateLeaseCandidateSpecUpdate(&leaseCandidate.Spec, &oldLeaseCandidate.Spec)...)
	return allErrs
}

// ValidateLeaseCandidateSpec validates spec of LeaseCandidate.
func ValidateLeaseCandidateSpec(spec *coordination.LeaseCandidateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.LeaseName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("leaseName"), ""))
	}

	ev := semver.Version{}
	if spec.EmulationVersion != "" {
		var err error
		ev, err = semver.Parse(spec.EmulationVersion)
		if err != nil {
			fld := fldPath.Child("emulationVersion")
			allErrs = append(allErrs, field.Invalid(fld, spec.EmulationVersion, "must be a valid semantic version"))
		}
	}
	bv := semver.Version{}
	if spec.BinaryVersion != "" {
		var err error
		bv, err = semver.Parse(spec.BinaryVersion)
		if err != nil {
			fld := fldPath.Child("binaryVersion")
			allErrs = append(allErrs, field.Invalid(fld, spec.BinaryVersion, "must be a valid semantic version"))
		}
	}
	if spec.BinaryVersion != "" && spec.EmulationVersion != "" && bv.LT(ev) {
		fld := fldPath.Child("binaryVersion")
		allErrs = append(allErrs, field.Invalid(fld, spec.BinaryVersion, "must be greater than or equal to `emulationVersion`"))
	}

	if len(spec.PreferredStrategies) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("preferredStrategies"), "must contain at least one strategy"))
	} else {
		for i, strategy := range spec.PreferredStrategies {
			fld := fldPath.Child("preferredStrategies").Index(i)
			if strategy == coordination.OldestEmulationVersion {
				zeroVersion := semver.Version{}
				if bv.EQ(zeroVersion) {
					allErrs = append(allErrs, field.Required(fldPath.Child("binaryVersion"), "must be specified when `strategy` is 'OldestEmulationVersion'"))
				}
				if ev.EQ(zeroVersion) {
					allErrs = append(allErrs, field.Required(fldPath.Child("emulationVersion"), "must be specified when `strategy` is 'OldestEmulationVersion'"))
				}
			}

			allErrs = append(allErrs, ValidateCoordinatedLeaseStrategy(strategy, fld)...)
		}
	}

	// spec.PingTime is a MicroTime and doesn't need further validation
	// spec.RenewTime is a MicroTime and doesn't need further validation

	return allErrs
}

// ValidateCoordinatedLeaseStrategy validates the Strategy field in both the Lease and LeaseCandidate
func ValidateCoordinatedLeaseStrategy(strategy coordination.CoordinatedLeaseStrategy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	parts := strings.Split(string(strategy), "/")
	switch len(parts) {
	case 1:
		// Must be a Kubernetes-defined name.
		if !slices.Contains(validLeaseStrategies, coordination.CoordinatedLeaseStrategy(parts[0])) {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("strategy"), strategy, validLeaseStrategies))
		}
	default:
		if msgs := utilvalidation.IsQualifiedName(string(strategy)); len(msgs) > 0 {
			for _, msg := range msgs {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("strategy"), strategy, msg))
			}
		}
	}
	return allErrs
}
