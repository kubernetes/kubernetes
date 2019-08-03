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
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
)

// RemoveObjectManagedFields removes the ManagedFields from the object
// before we merge so that it doesn't appear in the ManagedFields
// recursively.
func RemoveObjectManagedFields(obj runtime.Object) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}
	accessor.SetManagedFields(nil)
}

// DecodeObjectManagedFields extracts and converts the objects ManagedFields into a fieldpath.ManagedFields.
func DecodeObjectManagedFields(from runtime.Object) (fieldpath.ManagedFields, error) {
	if from == nil {
		return fieldpath.ManagedFields{}, nil
	}
	accessor, err := meta.Accessor(from)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
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
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
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
func decodeManagedFields(encodedManagedFields []metav1.ManagedFieldsEntry) (managedFields fieldpath.ManagedFields, err error) {
	managedFields = make(fieldpath.ManagedFields, len(encodedManagedFields))
	for _, encodedVersionedSet := range encodedManagedFields {
		manager, err := BuildManagerIdentifier(&encodedVersionedSet)
		if err != nil {
			return nil, fmt.Errorf("error decoding manager from %v: %v", encodedVersionedSet, err)
		}
		managedFields[manager], err = decodeVersionedSet(&encodedVersionedSet)
		if err != nil {
			return nil, fmt.Errorf("error decoding versioned set from %v: %v", encodedVersionedSet, err)
		}
	}
	return managedFields, nil
}

// BuildManagerIdentifier creates a manager identifier string from a ManagedFieldsEntry
func BuildManagerIdentifier(encodedManager *metav1.ManagedFieldsEntry) (manager string, err error) {
	encodedManagerCopy := *encodedManager

	// Never include the fields in the manager identifier
	encodedManagerCopy.Fields = nil

	// For appliers, don't include the APIVersion or Time in the manager identifier,
	// so it will always have the same manager identifier each time it applied.
	if encodedManager.Operation == metav1.ManagedFieldsOperationApply {
		encodedManagerCopy.APIVersion = ""
		encodedManagerCopy.Time = nil
	}

	// Use the remaining fields to build the manager identifier
	b, err := json.Marshal(&encodedManagerCopy)
	if err != nil {
		return "", fmt.Errorf("error marshalling manager identifier: %v", err)
	}

	return string(b), nil
}

func decodeVersionedSet(encodedVersionedSet *metav1.ManagedFieldsEntry) (versionedSet fieldpath.VersionedSet, err error) {
	fields := EmptyFields
	if encodedVersionedSet.Fields != nil {
		fields = *encodedVersionedSet.Fields
	}
	set, err := FieldsToSet(fields)
	if err != nil {
		return nil, fmt.Errorf("error decoding set: %v", err)
	}
	return fieldpath.NewVersionedSet(&set, fieldpath.APIVersion(encodedVersionedSet.APIVersion), encodedVersionedSet.Operation == metav1.ManagedFieldsOperationApply), nil
}

// encodeManagedFields converts ManagedFields from the format used by
// sigs.k8s.io/structured-merge-diff to the wire format (api format)
func encodeManagedFields(managedFields fieldpath.ManagedFields) (encodedManagedFields []metav1.ManagedFieldsEntry, err error) {
	// Sort the keys so a predictable order will be used.
	managers := []string{}
	for manager := range managedFields {
		managers = append(managers, manager)
	}
	sort.Strings(managers)

	encodedManagedFields = []metav1.ManagedFieldsEntry{}
	for _, manager := range managers {
		versionedSet := managedFields[manager]
		v, err := encodeManagerVersionedSet(manager, versionedSet)
		if err != nil {
			return nil, fmt.Errorf("error encoding versioned set for %v: %v", manager, err)
		}
		encodedManagedFields = append(encodedManagedFields, *v)
	}
	return sortEncodedManagedFields(encodedManagedFields)
}

func sortEncodedManagedFields(encodedManagedFields []metav1.ManagedFieldsEntry) (sortedManagedFields []metav1.ManagedFieldsEntry, err error) {
	sort.Slice(encodedManagedFields, func(i, j int) bool {
		p, q := encodedManagedFields[i], encodedManagedFields[j]

		if p.Operation != q.Operation {
			return p.Operation < q.Operation
		}

		ntime := &metav1.Time{Time: time.Time{}}
		if p.Time == nil {
			p.Time = ntime
		}
		if q.Time == nil {
			q.Time = ntime
		}
		if !p.Time.Equal(q.Time) {
			return p.Time.Before(q.Time)
		}

		return p.Manager < q.Manager
	})

	return encodedManagedFields, nil
}

func encodeManagerVersionedSet(manager string, versionedSet fieldpath.VersionedSet) (encodedVersionedSet *metav1.ManagedFieldsEntry, err error) {
	encodedVersionedSet = &metav1.ManagedFieldsEntry{}

	// Get as many fields as we can from the manager identifier
	err = json.Unmarshal([]byte(manager), encodedVersionedSet)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling manager identifier %v: %v", manager, err)
	}

	// Get the APIVersion, Operation, and Fields from the VersionedSet
	encodedVersionedSet.APIVersion = string(versionedSet.APIVersion())
	if versionedSet.Applied() {
		encodedVersionedSet.Operation = metav1.ManagedFieldsOperationApply
	}
	fields, err := SetToFields(*versionedSet.Set())
	if err != nil {
		return nil, fmt.Errorf("error encoding set: %v", err)
	}
	encodedVersionedSet.Fields = &fields

	return encodedVersionedSet, nil
}
