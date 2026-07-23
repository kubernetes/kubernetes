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
	"strings"

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
