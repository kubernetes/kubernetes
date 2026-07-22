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

// Package validation contains methods to validate kinds in the
// authentication.k8s.io API group.
package validation

import (
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	admissionregistrationv1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

const MinTokenAgeSec = 10 * 60 // 10 minutes

// ValidateTokenRequest validates a TokenRequest.
func ValidateTokenRequest(tr *authentication.TokenRequest) field.ErrorList {
	allErrs := field.ErrorList{}
	specPath := field.NewPath("spec")

	if tr.Spec.ExpirationSeconds < MinTokenAgeSec {
		allErrs = append(allErrs, field.Invalid(specPath.Child("expirationSeconds"), tr.Spec.ExpirationSeconds, "may not specify a duration less than 10 minutes"))
	}
	if tr.Spec.ExpirationSeconds > 1<<32 {
		allErrs = append(allErrs, field.Invalid(specPath.Child("expirationSeconds"), tr.Spec.ExpirationSeconds, "may not specify a duration larger than 2^32 seconds"))
	}

	switch {
	case isWebhookRef(tr.Spec.BoundObjectRef):
		if len(tr.Spec.Audiences) != 1 {
			allErrs = append(allErrs, field.Invalid(specPath.Child("audiences"), tr.Spec.Audiences, "must be length 1 when bound to a webhook config"))
		}
		allErrs = append(allErrs, validateWebhookAttestations(tr.Spec.Attestations, specPath.Child("attestations"))...)
	default:
		if len(tr.Spec.Attestations) != 0 {
			return field.ErrorList{field.Invalid(specPath.Child("attestations"), tr.Spec.Attestations, "attestations may only be specified with a webhook bound object reference")}
		}
	}

	return allErrs
}

func validateWebhookAttestations(attestations map[string]authentication.AttestationValue, attestationsPath *field.Path) field.ErrorList {
	if len(attestations) != 1 {
		return field.ErrorList{field.Invalid(attestationsPath, attestations, "webhook bound requests require exactly one admissionReviewAPIGroups attestation")}
	}

	// Check for incorrect keys and values
	errs := field.ErrorList{}
	for key, values := range attestations {
		errs = append(errs, validateWebhookKeysAndValues(key, values, attestationsPath.Key(key))...)
	}

	return errs
}

func validateWebhookKeysAndValues(key string, values authentication.AttestationValue, path *field.Path) field.ErrorList {
	switch key {
	case authentication.AttestationAdmissionReviewAPIGroups:
		if len(values) != 1 {
			return field.ErrorList{field.Invalid(path, values, "must specify a single value")}
		}
		return validateAdmissionReviewAPIGroupsValue(values[0], path)
	default:
		return field.ErrorList{field.NotSupported(path, key, []string{authentication.AttestationAdmissionReviewAPIGroups})}
	}
}

func validateAdmissionReviewAPIGroupsValue(group string, path *field.Path) field.ErrorList {
	var errs field.ErrorList

	// The value "" cannot be allowed for the "admissionReviewAPIGroups"
	// because "" means two different things depending on the context in
	// which it is presented. In the context of a resource name in an
	// authorization request, "" means "all resource names". In the
	// context of an API Group, "" means "the core API group".
	//
	// The only valid requester of an admission review token for the
	// core API group is the Kubernetes API Server itself. Kubernetes
	// API Server will always use "admissionReviewAPIGroups": "*" when
	// it needs a token to authenticate to a webhook, so there is never
	// a need to allow the value "" for this attestation.
	if len(group) == 0 {
		errs = append(errs, field.Invalid(path, group, "may not be an empty string"))
		return errs
	}

	// the * group is valid for kube-apiserver.
	if group == "*" {
		return nil
	}

	// valid subdomain, e.g. "admissionregistration.k8s.io" or "apps"
	for _, reason := range validation.IsDNS1123Subdomain(group) {
		errs = append(errs, field.Invalid(path, group, "must specify a valid API Group: "+reason))
	}

	return errs
}

func isWebhookRef(ref *authentication.BoundObjectReference) bool {
	if ref == nil {
		return false
	}
	switch ref.Kind {
	case "ValidatingWebhookConfiguration", "MutatingWebhookConfiguration":
		return ref.APIVersion == admissionregistrationv1.SchemeGroupVersion.String()
	default:
		return false
	}
}
