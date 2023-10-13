package schemavalidation

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Naive comparison in first alpha. This will change in a future release
func diffNativeToDeclarativeErrors(
	nativeResult field.ErrorList,
	declarativeResult field.ErrorList,
) (additions, deletions field.ErrorList) {
	// Naive error matching requires exact match. Will be replaced/improved
	// over time.
	existingErrors := sets.New[string]()
	declarativeErrors := sets.New[string]()
	for _, v := range nativeResult {
		existingErrors.Insert(v.Error())
	}

	for _, v := range declarativeResult {
		st := v.Error()
		declarativeErrors.Insert(st)
		if _, ok := existingErrors[st]; !ok {
			additions = append(additions, v)
		}
	}

	// Warn for all native errors that were in schema
	for _, v := range nativeResult {
		if _, ok := declarativeErrors[v.Error()]; !ok {
			deletions = append(deletions, v)
		}

	}

	return additions, deletions
}
