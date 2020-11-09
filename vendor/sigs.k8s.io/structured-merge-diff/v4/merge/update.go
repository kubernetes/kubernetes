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

package merge

import (
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// Converter is an interface to the conversion logic. The converter
// needs to be able to convert objects from one version to another.
type Converter interface {
	Convert(object *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error)
	IsMissingVersionError(error) bool
}

// Updater is the object used to compute updated FieldSets and also
// merge the object on Apply.
type Updater struct {
	Converter     Converter
	IgnoredFields map[fieldpath.APIVersion]*fieldpath.Set

	enableUnions bool
}

// EnableUnionFeature turns on union handling. It is disabled by default until the
// feature is complete.
func (s *Updater) EnableUnionFeature() {
	s.enableUnions = true
}

func (s *Updater) update(oldObject, newObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, workflow string, force bool) (fieldpath.ManagedFields, *typed.Comparison, error) {
	conflicts := fieldpath.ManagedFields{}
	removed := fieldpath.ManagedFields{}
	compare, err := oldObject.Compare(newObject)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to compare objects: %v", err)
	}

	versions := map[fieldpath.APIVersion]*typed.Comparison{
		version: compare.ExcludeFields(s.IgnoredFields[version]),
	}

	for manager, managerSet := range managers {
		if manager == workflow {
			continue
		}
		compare, ok := versions[managerSet.APIVersion()]
		if !ok {
			var err error
			versionedOldObject, err := s.Converter.Convert(oldObject, managerSet.APIVersion())
			if err != nil {
				if s.Converter.IsMissingVersionError(err) {
					delete(managers, manager)
					continue
				}
				return nil, nil, fmt.Errorf("failed to convert old object: %v", err)
			}
			versionedNewObject, err := s.Converter.Convert(newObject, managerSet.APIVersion())
			if err != nil {
				if s.Converter.IsMissingVersionError(err) {
					delete(managers, manager)
					continue
				}
				return nil, nil, fmt.Errorf("failed to convert new object: %v", err)
			}
			compare, err = versionedOldObject.Compare(versionedNewObject)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to compare objects: %v", err)
			}
			versions[managerSet.APIVersion()] = compare.ExcludeFields(s.IgnoredFields[managerSet.APIVersion()])
		}

		conflictSet := managerSet.Set().Intersection(compare.Modified.Union(compare.Added))
		if !conflictSet.Empty() {
			conflicts[manager] = fieldpath.NewVersionedSet(conflictSet, managerSet.APIVersion(), false)
		}

		if !compare.Removed.Empty() {
			removed[manager] = fieldpath.NewVersionedSet(compare.Removed, managerSet.APIVersion(), false)
		}
	}

	if !force && len(conflicts) != 0 {
		return nil, nil, ConflictsFromManagers(conflicts)
	}

	for manager, conflictSet := range conflicts {
		managers[manager] = fieldpath.NewVersionedSet(managers[manager].Set().Difference(conflictSet.Set()), managers[manager].APIVersion(), managers[manager].Applied())
	}

	for manager, removedSet := range removed {
		managers[manager] = fieldpath.NewVersionedSet(managers[manager].Set().Difference(removedSet.Set()), managers[manager].APIVersion(), managers[manager].Applied())
	}

	for manager := range managers {
		if managers[manager].Set().Empty() {
			delete(managers, manager)
		}
	}

	return managers, compare, nil
}

// Update is the method you should call once you've merged your final
// object on CREATE/UPDATE/PATCH verbs. newObject must be the object
// that you intend to persist (after applying the patch if this is for a
// PATCH call), and liveObject must be the original object (empty if
// this is a CREATE call).
func (s *Updater) Update(liveObject, newObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string) (*typed.TypedValue, fieldpath.ManagedFields, error) {
	var err error
	managers, err = s.reconcileManagedFieldsWithSchemaChanges(liveObject, managers)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	if s.enableUnions {
		newObject, err = liveObject.NormalizeUnions(newObject)
		if err != nil {
			return nil, fieldpath.ManagedFields{}, err
		}
	}
	managers, compare, err := s.update(liveObject, newObject, version, managers, manager, true)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	if _, ok := managers[manager]; !ok {
		managers[manager] = fieldpath.NewVersionedSet(fieldpath.NewSet(), version, false)
	}

	ignored := s.IgnoredFields[version]
	if ignored == nil {
		ignored = fieldpath.NewSet()
	}
	managers[manager] = fieldpath.NewVersionedSet(
		managers[manager].Set().Union(compare.Modified).Union(compare.Added).Difference(compare.Removed).RecursiveDifference(ignored),
		version,
		false,
	)
	if managers[manager].Set().Empty() {
		delete(managers, manager)
	}
	return newObject, managers, nil
}

// Apply should be called when Apply is run, given the current object as
// well as the configuration that is applied. This will merge the object
// and return it. If the object hasn't changed, nil is returned (the
// managers can still have changed though).
func (s *Updater) Apply(liveObject, configObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string, force bool) (*typed.TypedValue, fieldpath.ManagedFields, error) {
	var err error
	managers, err = s.reconcileManagedFieldsWithSchemaChanges(liveObject, managers)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	if s.enableUnions {
		configObject, err = configObject.NormalizeUnionsApply(configObject)
		if err != nil {
			return nil, fieldpath.ManagedFields{}, err
		}
	}
	newObject, err := liveObject.Merge(configObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to merge config: %v", err)
	}
	if s.enableUnions {
		newObject, err = configObject.NormalizeUnionsApply(newObject)
		if err != nil {
			return nil, fieldpath.ManagedFields{}, err
		}
	}
	lastSet := managers[manager]
	set, err := configObject.ToFieldSet()
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to get field set: %v", err)
	}

	ignored := s.IgnoredFields[version]
	if ignored != nil {
		set = set.RecursiveDifference(ignored)
		// TODO: is this correct. If we don't remove from lastSet pruning might remove the fields?
		if lastSet != nil {
			lastSet.Set().RecursiveDifference(ignored)
		}
	}
	managers[manager] = fieldpath.NewVersionedSet(set, version, true)
	newObject, err = s.prune(newObject, managers, manager, lastSet)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to prune fields: %v", err)
	}
	managers, compare, err := s.update(liveObject, newObject, version, managers, manager, force)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	if compare.IsSame() {
		newObject = nil
	}
	return newObject, managers, nil
}

// prune will remove a field, list or map item, iff:
// * applyingManager applied it last time
// * applyingManager didn't apply it this time
// * no other applier claims to manage it
func (s *Updater) prune(merged *typed.TypedValue, managers fieldpath.ManagedFields, applyingManager string, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error) {
	if lastSet == nil || lastSet.Set().Empty() {
		return merged, nil
	}
	convertedMerged, err := s.Converter.Convert(merged, lastSet.APIVersion())
	if err != nil {
		if s.Converter.IsMissingVersionError(err) {
			return merged, nil
		}
		return nil, fmt.Errorf("failed to convert merged object to last applied version: %v", err)
	}

	pruned := convertedMerged.RemoveItems(lastSet.Set())
	pruned, err = s.addBackOwnedItems(convertedMerged, pruned, managers, applyingManager)
	if err != nil {
		return nil, fmt.Errorf("failed add back owned items: %v", err)
	}
	pruned, err = s.addBackDanglingItems(convertedMerged, pruned, lastSet)
	if err != nil {
		return nil, fmt.Errorf("failed add back dangling items: %v", err)
	}
	return s.Converter.Convert(pruned, managers[applyingManager].APIVersion())
}

// addBackOwnedItems adds back any fields, list and map items that were removed by prune,
// but other appliers or updaters (or the current applier's new config) claim to own.
func (s *Updater) addBackOwnedItems(merged, pruned *typed.TypedValue, managedFields fieldpath.ManagedFields, applyingManager string) (*typed.TypedValue, error) {
	var err error
	managedAtVersion := map[fieldpath.APIVersion]*fieldpath.Set{}
	for _, managerSet := range managedFields {
		if _, ok := managedAtVersion[managerSet.APIVersion()]; !ok {
			managedAtVersion[managerSet.APIVersion()] = fieldpath.NewSet()
		}
		managedAtVersion[managerSet.APIVersion()] = managedAtVersion[managerSet.APIVersion()].Union(managerSet.Set())
	}
	for version, managed := range managedAtVersion {
		merged, err = s.Converter.Convert(merged, version)
		if err != nil {
			if s.Converter.IsMissingVersionError(err) {
				continue
			}
			return nil, fmt.Errorf("failed to convert merged object at version %v: %v", version, err)
		}
		pruned, err = s.Converter.Convert(pruned, version)
		if err != nil {
			if s.Converter.IsMissingVersionError(err) {
				continue
			}
			return nil, fmt.Errorf("failed to convert pruned object at version %v: %v", version, err)
		}
		mergedSet, err := merged.ToFieldSet()
		if err != nil {
			return nil, fmt.Errorf("failed to create field set from merged object at version %v: %v", version, err)
		}
		prunedSet, err := pruned.ToFieldSet()
		if err != nil {
			return nil, fmt.Errorf("failed to create field set from pruned object at version %v: %v", version, err)
		}
		pruned = merged.RemoveItems(mergedSet.Difference(prunedSet.Union(managed)))
	}
	return pruned, nil
}

// addBackDanglingItems makes sure that the fields list and map items removed by prune were
// previously owned by the currently applying manager. This will add back fields list and map items
// that are unowned or that are owned by Updaters and shouldn't be removed.
func (s *Updater) addBackDanglingItems(merged, pruned *typed.TypedValue, lastSet fieldpath.VersionedSet) (*typed.TypedValue, error) {
	convertedPruned, err := s.Converter.Convert(pruned, lastSet.APIVersion())
	if err != nil {
		if s.Converter.IsMissingVersionError(err) {
			return merged, nil
		}
		return nil, fmt.Errorf("failed to convert pruned object to last applied version: %v", err)
	}
	prunedSet, err := convertedPruned.ToFieldSet()
	if err != nil {
		return nil, fmt.Errorf("failed to create field set from pruned object in last applied version: %v", err)
	}
	mergedSet, err := merged.ToFieldSet()
	if err != nil {
		return nil, fmt.Errorf("failed to create field set from merged object in last applied version: %v", err)
	}
	return merged.RemoveItems(mergedSet.Difference(prunedSet).Intersection(lastSet.Set())), nil
}

// reconcileManagedFieldsWithSchemaChanges reconciles the managed fields with any changes to the
// object's schema since the managed fields were written.
//
// Supports:
// - changing types from atomic to granular
// - changing types from granular to atomic
func (s *Updater) reconcileManagedFieldsWithSchemaChanges(liveObject *typed.TypedValue, managers fieldpath.ManagedFields) (fieldpath.ManagedFields, error) {
	result := fieldpath.ManagedFields{}
	for manager, versionedSet := range managers {
		tv, err := s.Converter.Convert(liveObject, versionedSet.APIVersion())
		if s.Converter.IsMissingVersionError(err) { // okay to skip, obsolete versions will be deleted automatically anyway
			continue
		}
		if err != nil {
			return nil, err
		}
		reconciled, err := typed.ReconcileFieldSetWithSchema(versionedSet.Set(), tv)
		if err != nil {
			return nil, err
		}
		if reconciled != nil {
			result[manager] = fieldpath.NewVersionedSet(reconciled, versionedSet.APIVersion(), versionedSet.Applied())
		} else {
			result[manager] = versionedSet
		}
	}
	return result, nil
}
