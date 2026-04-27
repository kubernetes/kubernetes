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

package internal

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/merge"
)

type lastAppliedManager struct {
	fieldManager    Manager
	typeConverter   TypeConverter
	objectConverter runtime.ObjectConvertor
	groupVersion    schema.GroupVersion
}

var _ Manager = &lastAppliedManager{}

// NewLastAppliedManager converts the client-side apply annotation to
// server-side apply managed fields
func NewLastAppliedManager(fieldManager Manager, typeConverter TypeConverter, objectConverter runtime.ObjectConvertor, groupVersion schema.GroupVersion) Manager {
	return &lastAppliedManager{
		fieldManager:    fieldManager,
		typeConverter:   typeConverter,
		objectConverter: objectConverter,
		groupVersion:    groupVersion,
	}
}

// Update implements Manager.
func (f *lastAppliedManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	return f.fieldManager.Update(liveObj, newObj, managed, manager)
}

// Apply will consider the last-applied annotation
// for upgrading an object managed by client-side apply to server-side apply
// without conflicts.
func (f *lastAppliedManager) Apply(liveObj, newObj runtime.Object, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	newLiveObj, newManaged, newErr := f.fieldManager.Apply(liveObj, newObj, managed, manager, force)
	// Upgrade the client-side apply annotation only from kubectl server-side-apply.
	// To opt-out of this behavior, users may specify a different field manager.
	if manager != "kubectl" {
		return newLiveObj, newManaged, newErr
	}

	// Check if we have conflicts
	if newErr == nil {
		return newLiveObj, newManaged, newErr
	}
	conflicts, ok := newErr.(merge.Conflicts)
	if !ok {
		return newLiveObj, newManaged, newErr
	}
	conflictSet := conflictsToSet(conflicts)

	// Check if conflicts are allowed due to client-side apply,
	// and if so, then force apply
	allowedConflictSet, err := f.allowedConflictsFromLastApplied(liveObj)
	if err != nil {
		return newLiveObj, newManaged, newErr
	}
	if !conflictSet.Difference(allowedConflictSet).Empty() {
		newConflicts := conflictsDifference(conflicts, allowedConflictSet)
		return newLiveObj, newManaged, newConflicts
	}

	return f.fieldManager.Apply(liveObj, newObj, managed, manager, true)
}

func (f *lastAppliedManager) allowedConflictsFromLastApplied(liveObj runtime.Object) (*fieldpath.Set, error) {
	var accessor, err = meta.Accessor(liveObj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}

	// If there is no client-side apply annotation, then there is nothing to do
	var annotations = accessor.GetAnnotations()
	if annotations == nil {
		return nil, fmt.Errorf("no last applied annotation")
	}
	var lastApplied, ok = annotations[LastAppliedConfigAnnotation]
	if !ok || lastApplied == "" {
		return nil, fmt.Errorf("no last applied annotation")
	}

	liveObjVersioned, err := f.objectConverter.ConvertToVersion(liveObj, f.groupVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live obj to versioned: %v", err)
	}

	liveObjTyped, err := f.typeConverter.ObjectToTyped(liveObjVersioned)
	if err != nil {
		return nil, fmt.Errorf("failed to convert live obj to typed: %v", err)
	}

	var lastAppliedObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	err = json.Unmarshal([]byte(lastApplied), lastAppliedObj)
	if err != nil {
		return nil, fmt.Errorf("failed to decode last applied obj: %v in '%s'", err, lastApplied)
	}

	if lastAppliedObj.GetAPIVersion() != f.groupVersion.String() {
		return nil, fmt.Errorf("expected version of last applied to match live object '%s', but got '%s': %v", f.groupVersion.String(), lastAppliedObj.GetAPIVersion(), err)
	}

	lastAppliedObjTyped, err := f.typeConverter.ObjectToTyped(lastAppliedObj)
	if err != nil {
		return nil, fmt.Errorf("failed to convert last applied to typed: %v", err)
	}

	lastAppliedObjFieldSet, err := lastAppliedObjTyped.ToFieldSet()
	if err != nil {
		return nil, fmt.Errorf("failed to create fieldset for last applied object: %v", err)
	}

	comparison, err := lastAppliedObjTyped.Compare(liveObjTyped)
	if err != nil {
		return nil, fmt.Errorf("failed to compare last applied object and live object: %v", err)
	}

	// Remove fields in last applied that are different, added, or missing in
	// the live object.
	// Because last-applied fields don't match the live object fields,
	// then we don't own these fields.
	lastAppliedObjFieldSet = lastAppliedObjFieldSet.
		Difference(comparison.Modified).
		Difference(comparison.Added).
		Difference(comparison.Removed)

	return lastAppliedObjFieldSet, nil
}

// TODO: replace with merge.Conflicts.ToSet()
func conflictsToSet(conflicts merge.Conflicts) *fieldpath.Set {
	conflictSet := fieldpath.NewSet()
	for _, conflict := range []merge.Conflict(conflicts) {
		conflictSet.Insert(conflict.Path)
	}
	return conflictSet
}

func conflictsDifference(conflicts merge.Conflicts, s *fieldpath.Set) merge.Conflicts {
	newConflicts := []merge.Conflict{}
	for _, conflict := range []merge.Conflict(conflicts) {
		if !s.Has(conflict.Path) {
			newConflicts = append(newConflicts, conflict)
		}
	}
	return newConflicts
}
