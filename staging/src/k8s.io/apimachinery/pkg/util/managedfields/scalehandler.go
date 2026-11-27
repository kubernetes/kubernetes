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

package managedfields

import (
	"fmt"
	"maps"

	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
)

var (
	scaleGroupVersion   = schema.GroupVersion{Group: "autoscaling", Version: "v1"}
	replicasPathInScale = fieldpath.MakePathOrDie("spec", "replicas")
)

// ResourcePathMappings maps a group/version to its replicas path. The
// assumption is that all the paths correspond to leaf fields.
type ResourcePathMappings map[string]fieldpath.Path

// ScaleHandler manages the conversion of managed fields between a main
// resource and the scale subresource
type ScaleHandler struct {
	parentEntries []metav1.ManagedFieldsEntry
	groupVersion  schema.GroupVersion
	mappings      ResourcePathMappings
}

// NewScaleHandler creates a new ScaleHandler
func NewScaleHandler(parentEntries []metav1.ManagedFieldsEntry, groupVersion schema.GroupVersion, mappings ResourcePathMappings) *ScaleHandler {
	return &ScaleHandler{
		parentEntries: parentEntries,
		groupVersion:  groupVersion,
		mappings:      mappings,
	}
}

// Older scale updates could split one Update entry into an untimed entry
// owning only replicas and another entry with the same identifier owning the
// remaining fields. DecodeManagedFields retains only the last such entry.
func (h *ScaleHandler) decodeParentEntries() (internal.ManagedInterface, error) {
	managed, err := internal.DecodeManagedFields(h.parentEntries)
	if err != nil || len(managed.Fields()) == len(h.parentEntries) {
		return managed, err
	}
	for _, entry := range h.parentEntries {
		path := h.mappings[entry.APIVersion]
		if entry.Operation != metav1.ManagedFieldsOperationUpdate || path == nil || entry.FieldsV1 == nil {
			continue
		}
		manager, err := internal.BuildManagerIdentifier(&entry)
		if err != nil {
			return nil, err
		}
		fields, err := internal.FieldsToSet(*entry.FieldsV1)
		if err != nil {
			return nil, err
		}
		retained := managed.Fields()[manager]
		replicas := fieldpath.NewSet(path)
		switch {
		case entry.Time == nil && fields.Equals(replicas) && !retained.Set().Has(path):
			retained.Set().Insert(path)
		case managed.Times()[manager] == nil && retained.Set().Equals(replicas) && !fields.Has(path):
			fields.Insert(path)
			managed.Fields()[manager] = fieldpath.NewVersionedSet(&fields, retained.APIVersion(), false)
			managed.Times()[manager] = entry.Time
		}
	}
	return managed, nil
}

// ToSubresource filter the managed fields of the main resource and convert
// them so that they can be handled by scale.
// For the managed fields that have a replicas path it performs two changes:
//  1. APIVersion is changed to the APIVersion of the scale subresource
//  2. Replicas path of the main resource is transformed to the replicas path of
//     the scale subresource
func (h *ScaleHandler) ToSubresource() ([]metav1.ManagedFieldsEntry, error) {
	managed, err := h.decodeParentEntries()
	if err != nil {
		return nil, err
	}

	f := fieldpath.ManagedFields{}
	t := map[string]*metav1.Time{}
	for manager, versionedSet := range managed.Fields() {
		path, ok := h.mappings[string(versionedSet.APIVersion())]
		// Skip the entry if the APIVersion is unknown
		if !ok || path == nil {
			continue
		}

		if versionedSet.Set().Has(path) {
			newVersionedSet := fieldpath.NewVersionedSet(
				fieldpath.NewSet(replicasPathInScale),
				fieldpath.APIVersion(scaleGroupVersion.String()),
				versionedSet.Applied(),
			)

			f[manager] = newVersionedSet
			t[manager] = managed.Times()[manager]
		}
	}

	return managedFieldsEntries(internal.NewManaged(f, t))
}

// ToParent merges `scaleEntries` with the entries of the main resource and
// transforms them accordingly
func (h *ScaleHandler) ToParent(scaleEntries []metav1.ManagedFieldsEntry) ([]metav1.ManagedFieldsEntry, error) {
	decodedParentEntries, err := h.decodeParentEntries()
	if err != nil {
		return nil, err
	}
	parentFields := decodedParentEntries.Fields()

	decodedScaleEntries, err := internal.DecodeManagedFields(scaleEntries)
	if err != nil {
		return nil, err
	}
	scaleFields := decodedScaleEntries.Fields()
	// Multiple parent API versions can project to the same scale manager.
	unmatchedScaleFields := maps.Clone(scaleFields)

	f := fieldpath.ManagedFields{}
	t := map[string]*metav1.Time{}

	for _, entry := range h.parentEntries {
		manager, err := internal.BuildManagerIdentifier(&entry)
		if err != nil {
			return nil, err
		}
		versionedSet := parentFields[manager]
		// Get the main resource "replicas" path
		path, ok := h.mappings[string(versionedSet.APIVersion())]
		// Drop the entry if the APIVersion is unknown.
		if !ok {
			continue
		}

		// If the parent entry does not have the replicas path or it is nil, just
		// keep it as it is. The path is nil for Custom Resources without scale
		// subresource.
		if path == nil || !versionedSet.Set().Has(path) {
			f[manager] = versionedSet
			t[manager] = decodedParentEntries.Times()[manager]
			continue
		}

		// Update manager identifiers include the API version. Look up the
		// projected owner under the version used by the scale subresource.
		entry.APIVersion = scaleGroupVersion.String()
		scaleManager, err := internal.BuildManagerIdentifier(&entry)
		if err != nil {
			return nil, err
		}
		if fields, ok := scaleFields[scaleManager]; !ok || !fields.Set().Has(replicasPathInScale) {
			// "Steal" the replicas path from the main resource entry
			newSet := versionedSet.Set().Difference(fieldpath.NewSet(path))

			if !newSet.Empty() {
				newVersionedSet := fieldpath.NewVersionedSet(
					newSet,
					versionedSet.APIVersion(),
					versionedSet.Applied(),
				)
				f[manager] = newVersionedSet
				t[manager] = decodedParentEntries.Times()[manager]
			}
		} else {
			// Field wasn't stolen, let's keep the entry as it is.
			f[manager] = versionedSet
			t[manager] = decodedParentEntries.Times()[manager]
			delete(unmatchedScaleFields, scaleManager)
		}
	}

	for manager, versionedSet := range unmatchedScaleFields {
		if !versionedSet.Set().Has(replicasPathInScale) {
			continue
		}
		newVersionedSet := fieldpath.NewVersionedSet(
			fieldpath.NewSet(h.mappings[h.groupVersion.String()]),
			fieldpath.APIVersion(h.groupVersion.String()),
			versionedSet.Applied(),
		)
		f[manager] = newVersionedSet
		t[manager] = decodedParentEntries.Times()[manager]
	}

	return managedFieldsEntries(internal.NewManaged(f, t))
}

func managedFieldsEntries(entries internal.ManagedInterface) ([]metav1.ManagedFieldsEntry, error) {
	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := internal.EncodeObjectManagedFields(obj, entries); err != nil {
		return nil, err
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}
	return accessor.GetManagedFields(), nil
}
