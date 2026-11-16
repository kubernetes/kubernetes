/*
Copyright The Kubernetes Authors.

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

package v1

import (
	"context"
	"fmt"
	"path"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/distribution/reference"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateCustom_EvictionResponder_Name is wired into the generated declarative validation by
// +k8s:customValidation on corev1.EvictionResponder.Name. It enforces that the
// k8s-prefixed-label-key value, is not prefixed with a k8s.io or kubernetes.io domain.
func ValidateCustom_EvictionResponder_Name(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *string) field.ErrorList {
	if value != nil {
		return ValidateForbiddenReservedDomainSuffixes(fldPath, *value, []string{".k8s.io", ".kubernetes.io"})

	}
	return nil
}

// ValidateCustom_EvictionResponder_Priority is wired into the generated declarative validation by
// +k8s:customValidation on corev1.EvictionResponder.Priority. It ensures that k8s priority
// intervals are reserved for k8s responder use.
func ValidateCustom_EvictionResponder_Priority(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *int32) field.ErrorList {
	const reservedKubernetesRespondersPriority = 1000
	if value != nil && *value >= 0 && *value < reservedKubernetesRespondersPriority {
		// There is no need to check the responder name since we do not yet allow k8s responders here.
		// The default imperative-eviction responder is set to the Eviction object directly.
		// Once we allow core responders, they should have assigned priorities.
		return field.ErrorList{field.Invalid(fldPath, *value, "priorities 0-999 are reserved for responders with *.k8s.io suffix")}
	}
	return nil
}

// ValidateForbiddenReservedDomainSuffixes checks that the reservedSuffixes are not being used
func ValidateForbiddenReservedDomainSuffixes(fldPath *field.Path, value string, reservedSuffixes []string) field.ErrorList {
	var allErrors field.ErrorList
	segments := strings.SplitN(value, "/", 2)
	if len(segments) == 0 {
		return allErrors
	}
	cleanValue := strings.TrimRight(strings.TrimSpace(strings.ToLower(segments[0])), ".")
	for _, suffix := range reservedSuffixes {
		if strings.HasSuffix(cleanValue, suffix) || cleanValue == strings.Trim(suffix, ".") {
			var userFormattedPrefixes []string
			for _, reservedSuffix := range reservedSuffixes {
				userFormattedPrefixes = append(userFormattedPrefixes, "*"+reservedSuffix)
			}
			return append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("domain names %s are reserved", strings.Join(userFormattedPrefixes, ", "))))
		}
	}
	return allErrors
}

// The custom validators below return early for values longer than the
// +k8s:maxLength of their field, so that an oversized value is neither parsed
// nor echoed in an error; the maxLength rule reports it. Like that rule, they
// count characters, and the constants must match the tags in
// staging/src/k8s.io/api/core/v1/types.go.
const (
	securityProfileOCIRefMaxLength                  = 1024
	securityProfileOCIBaseLocalhostProfileMaxLength = 4096
)

// ValidateCustom_SecurityProfileOCI_Ref is wired into the generated declarative
// validation by +k8s:customValidation on corev1.SecurityProfileOCI.Ref. It
// requires a canonical, digest-pinned OCI reference without a tag, with a
// sha256, sha384, or sha512 digest. Empty and oversized values are left to
// +k8s:required and +k8s:maxLength.
func ValidateCustom_SecurityProfileOCI_Ref(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *string) field.ErrorList {
	if value == nil || len(*value) == 0 || utf8.RuneCountInString(*value) > securityProfileOCIRefMaxLength {
		return nil
	}
	named, err := reference.ParseNamed(*value)
	if err != nil {
		return field.ErrorList{field.Invalid(fldPath, *value, fmt.Sprintf("must be a digest-pinned reference in canonical form (<registry>/<repository>@<algorithm>:<digest>): %v", err))}
	}
	if _, ok := named.(reference.Tagged); ok {
		return field.ErrorList{field.Invalid(fldPath, *value, "must not contain a tag")}
	}
	digested, ok := named.(reference.Digested)
	if !ok {
		return field.ErrorList{field.Invalid(fldPath, *value, "must be pinned by digest (<registry>/<repository>@<algorithm>:<digest>)")}
	}
	// The parser accepts any algorithm registered with go-digest; the API
	// documents these three.
	switch string(digested.Digest().Algorithm()) {
	case "sha256", "sha384", "sha512":
		return nil
	}
	return field.ErrorList{field.Invalid(fldPath, *value, "digest algorithm must be sha256, sha384, or sha512")}
}

// ValidateCustom_SecurityProfileOCIBase_LocalhostProfile is wired into the
// generated declarative validation by +k8s:customValidation on
// corev1.SecurityProfileOCIBase.LocalhostProfile. It requires a non-empty
// relative path without ".." elements, like the localhostProfile of a seccomp
// profile of type Localhost. Only "/" separates path elements: seccomp is
// Linux-only and the kubelet resolves the path on the node, where a backslash
// is an ordinary file name character. Whether the field may be set is decided
// by the union with the base profile type, and oversized values are left to
// +k8s:maxLength.
func ValidateCustom_SecurityProfileOCIBase_LocalhostProfile(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *string) field.ErrorList {
	if value == nil || utf8.RuneCountInString(*value) > securityProfileOCIBaseLocalhostProfileMaxLength {
		return nil
	}
	if len(*value) == 0 {
		return field.ErrorList{field.Required(fldPath, "must not be empty")}
	}
	var allErrs field.ErrorList
	if path.IsAbs(*value) {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, "must be a relative path"))
	}
	if slices.Contains(strings.Split(*value, "/"), "..") {
		allErrs = append(allErrs, field.Invalid(fldPath, *value, "must not contain '..'"))
	}
	return allErrs
}
