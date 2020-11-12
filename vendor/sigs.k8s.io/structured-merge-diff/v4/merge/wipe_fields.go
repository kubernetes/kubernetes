/*
Copyright 2020 The Kubernetes Authors.

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
package merge

import (
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// WipeManagedFields clears the managedFields for the current manager from any fields that
// have been/will be removed from the object during preparation.
// The fields to clear are determined based on the fields that were not added or modified in the preparedObject
// compared to the liveObject
func WipeManagedFields(
	liveManagedFields, newManagedFields fieldpath.ManagedFields,
	manager string,
	liveObject, preparedObject *typed.TypedValue,
) (fieldpath.ManagedFields, error) {
	newManaged, exists := newManagedFields[manager]
	if !exists {
		return newManagedFields, nil
	}

	liveManaged, exists := liveManagedFields[manager]
	if !exists {
		liveManaged = fieldpath.NewVersionedSet(fieldpath.NewSet(), newManaged.APIVersion(), newManaged.Applied())
	}

	comparisonLivePrepared, err := liveObject.Compare(preparedObject)
	if err != nil {
		return nil, fmt.Errorf("comparing live and prepared: %w", err)
	}

	addedManagedFields := newManaged.Set().Difference(liveManaged.Set())
	// remove fields from addedManagedFields that are gone after the object got prepared
	addedManagedFields = addedManagedFields.Difference(comparisonLivePrepared.Added)
	addedManagedFields = addedManagedFields.Difference(comparisonLivePrepared.Modified)

	newManagedFields[manager] = fieldpath.NewVersionedSet(
		newManaged.Set().Difference(addedManagedFields),
		newManaged.APIVersion(),
		newManaged.Applied(),
	)

	return newManagedFields, nil
}
