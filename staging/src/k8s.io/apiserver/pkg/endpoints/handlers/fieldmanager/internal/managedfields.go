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

package internal

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// RemoveObjectManagedFields removes the ManagedFields from the object
// before we merge so that it doesn't appear in the ManagedFields
// recursively.
func RemoveObjectManagedFields(obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("couldn't get accessor: %v", err)
	}
	accessor.SetManagedFields(nil)
	return nil
}

// DecodeObjectManagedFields extracts and converts the objects ManagedFields into a fieldpath.ManagedFields.
func DecodeObjectManagedFields(from runtime.Object) (fieldpath.ManagedFields, error) {
	if from == nil {
		return make(map[string]*fieldpath.VersionedSet), nil
	}
	accessor, err := meta.Accessor(from)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}

	managed, err := decodeManagedFields(accessor.GetManagedFields())
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

	managed, err := encodeManagedFields(fields)
	if err != nil {
		return fmt.Errorf("failed to convert back managed fields to API: %v", err)
	}
	accessor.SetManagedFields(managed)

	return nil
}

// decodeManagedFields converts ManagedFields from the wire format (api format)
// to the format used by sigs.k8s.io/structured-merge-diff
func decodeManagedFields(encodedManagedFields map[string]metav1.VersionedFields) (managedFields fieldpath.ManagedFields, err error) {
	managedFields = make(map[string]*fieldpath.VersionedSet, len(encodedManagedFields))
	for manager, encodedVersionedSet := range encodedManagedFields {
		managedFields[manager], err = decodeVersionedSet(&encodedVersionedSet)
		if err != nil {
			return nil, fmt.Errorf("error decoding versioned set for %v: %v", manager, err)
		}
	}
	return managedFields, nil
}

func decodeVersionedSet(encodedVersionedSet *metav1.VersionedFields) (versionedSet *fieldpath.VersionedSet, err error) {
	versionedSet = &fieldpath.VersionedSet{}
	versionedSet.APIVersion = fieldpath.APIVersion(encodedVersionedSet.APIVersion)
	set, err := FieldsToSet(encodedVersionedSet.Fields)
	if err != nil {
		return nil, fmt.Errorf("error decoding set: %v", err)
	}
	versionedSet.Set = &set
	return versionedSet, nil
}

// encodeManagedFields converts ManagedFields from the the format used by
// sigs.k8s.io/structured-merge-diff to the the wire format (api format)
func encodeManagedFields(managedFields fieldpath.ManagedFields) (encodedManagedFields map[string]metav1.VersionedFields, err error) {
	encodedManagedFields = make(map[string]metav1.VersionedFields, len(managedFields))
	for manager, versionedSet := range managedFields {
		v, err := encodeVersionedSet(versionedSet)
		if err != nil {
			return nil, fmt.Errorf("error encoding versioned set for %v: %v", manager, err)
		}
		encodedManagedFields[manager] = *v
	}
	return encodedManagedFields, nil
}

func encodeVersionedSet(versionedSet *fieldpath.VersionedSet) (encodedVersionedSet *metav1.VersionedFields, err error) {
	encodedVersionedSet = &metav1.VersionedFields{}
	encodedVersionedSet.APIVersion = string(versionedSet.APIVersion)
	encodedVersionedSet.Fields, err = SetToFields(*versionedSet.Set)
	if err != nil {
		return nil, fmt.Errorf("error encoding set: %v", err)
	}
	return encodedVersionedSet, nil
}
