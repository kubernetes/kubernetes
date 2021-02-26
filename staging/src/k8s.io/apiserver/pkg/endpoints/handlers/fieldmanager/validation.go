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
