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

package v1alpha1

import (
	"context"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apiscorev1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

// ValidateCustom_EvictionRequestSpec_Requester is wired into the generated declarative validation by
// +k8s:customValidation on corev1.EvictionRequestSpec.Requester. It enforces that the
// k8s-prefixed-label-key value, is not prefixed with a k8s.io or kubernetes.io domain.
func ValidateCustom_EvictionRequestSpec_Requester(ctx context.Context, op operation.Operation, fldPath *field.Path, value, oldValue *string) field.ErrorList {
	if value != nil {
		// Once, we have k8s requesters, we need to check a set of supported requesters that can bypass this validation.
		return apiscorev1.ValidateForbiddenReservedDomainSuffixes(fldPath, *value, []string{".k8s.io", ".kubernetes.io"})

	}
	return nil
}
