package managedfields

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
)

// ManagedInterface groups a fieldpath.ManagedFields together with the timestamps associated with each operation.
type ManagedInterface = internal.ManagedInterface

// DecodeManagedFields converts ManagedFields from the wire format (api format)
// to the format used by sigs.k8s.io/structured-merge-diff
func DecodeManagedFields(encodedManagedFields []metav1.ManagedFieldsEntry) (ManagedInterface, error) {
	return internal.DecodeManagedFields(encodedManagedFields)
}
