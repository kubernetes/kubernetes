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

package apply

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// DecodeObjectManagedFields extracts and converts the objects ManagedFields into a fieldpath.ManagedFields.
func DecodeObjectManagedFields(from runtime.Object) (fieldpath.ManagedFields, error) {
	if from == nil {
		return make(map[string]*fieldpath.VersionedSet), nil
	}
	accessor, err := meta.Accessor(from)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	accessor.SetManagedFields(nil)

	managed, err := DecodeManagedFields(accessor.GetManagedFields())
	if err != nil {
		return nil, fmt.Errorf("failed to convert managed fields from API: %v", err)
	}
	return managed, err
}

// EncodeObjectManagedFields converts and stores the fieldpathManagedFields into the objects ManagedFields
func EncodeObjectManagedFields(obj runtime.Object, fields fieldpath.ManagedFields) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("couldn't get accessor: %v", err)
	}

	managed, err := EncodeManagedFields(fields)
	if err != nil {
		return fmt.Errorf("failed to convert back managed fields to API: %v", err)
	}
	accessor.SetManagedFields(managed)

	return nil
}
