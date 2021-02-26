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

package fieldmanager

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateManagedFields by checking the integrity of every entry and trying to decode
// them to the internal format
func ValidateManagedFields(managedFields []metav1.ManagedFieldsEntry) error {
	validationErrs := v1validation.ValidateManagedFields(managedFields, field.NewPath("metadata").Child("managedFields"))
	if len(validationErrs) > 0 {
		return validationErrs.ToAggregate()
	}

	if _, err := DecodeManagedFields(managedFields); err != nil {
		return err
	}

	return nil
}
