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

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/typed"
)

// Converter is an interface to the conversion logic. The converter
// needs to be able to convert objects from one version to another.
type Converter interface {
	Convert(object typed.TypedValue, version fieldpath.APIVersion) (typed.TypedValue, error)
}

// Updater is the object used to compute updated FieldSets and also
// merge the object on Apply.
type Updater struct {
	Converter Converter
}

func (s *Updater) update(oldObject, newObject typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, workflow string, force bool) (fieldpath.ManagedFields, error) {
	if managers == nil {
		managers = fieldpath.ManagedFields{}
	}
	conflicts := fieldpath.ManagedFields{}
	type Versioned struct {
		oldObject typed.TypedValue
		newObject typed.TypedValue
	}
	versions := map[fieldpath.APIVersion]Versioned{
		version: Versioned{
			oldObject: oldObject,
			newObject: newObject,
		},
	}

	for manager, managerSet := range managers {
		if manager == workflow {
			continue
		}
		versioned, ok := versions[managerSet.APIVersion]
		if !ok {
			var err error
			versioned.oldObject, err = s.Converter.Convert(oldObject, managerSet.APIVersion)
			if err != nil {
				return nil, fmt.Errorf("failed to convert old object: %v", err)
			}
			versioned.newObject, err = s.Converter.Convert(newObject, managerSet.APIVersion)
			if err != nil {
				return nil, fmt.Errorf("failed to convert new object: %v", err)
			}
			versions[managerSet.APIVersion] = versioned
		}
		compare, err := versioned.oldObject.Compare(versioned.newObject)
		if err != nil {
			return nil, fmt.Errorf("failed to compare objects: %v", err)
		}

		conflictSet := managerSet.Intersection(compare.Modified.Union(compare.Added))
		if !conflictSet.Empty() {
			conflicts[manager] = &fieldpath.VersionedSet{
				Set:        conflictSet,
				APIVersion: managerSet.APIVersion,
			}
		}
	}

	if !force && len(conflicts) != 0 {
		return nil, ConflictsFromManagers(conflicts)
	}

	for manager, conflictSet := range conflicts {
		managers[manager].Set = managers[manager].Set.Difference(conflictSet.Set)
		if managers[manager].Set.Empty() {
			delete(managers, manager)
		}
	}

	if _, ok := managers[workflow]; !ok {
		managers[workflow] = &fieldpath.VersionedSet{
			Set: fieldpath.NewSet(),
		}
	}

	return managers, nil
}

// Update is the method you should call once you've merged your final
// object on CREATE/UPDATE/PATCH verbs. newObject must be the object
// that you intend to persist (after applying the patch if this is for a
// PATCH call), and liveObject must be the original object (empty if
// this is a CREATE call).
func (s *Updater) Update(liveObject, newObject typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string) (fieldpath.ManagedFields, error) {
	var err error
	managers, err = s.update(liveObject, newObject, version, managers, manager, true)
	if err != nil {
		return fieldpath.ManagedFields{}, err
	}
	compare, err := liveObject.Compare(newObject)
	if err != nil {
		return fieldpath.ManagedFields{}, fmt.Errorf("failed to compare live and new objects: %v", err)
	}
	managers[manager].Set = managers[manager].Set.Union(compare.Modified).Union(compare.Added).Difference(compare.Removed)
	managers[manager].APIVersion = version
	if managers[manager].Set.Empty() {
		delete(managers, manager)
	}
	return managers, nil
}

// Apply should be called when Apply is run, given the current object as
// well as the configuration that is applied. This will merge the object
// and return it.
func (s *Updater) Apply(liveObject, configObject typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string, force bool) (typed.TypedValue, fieldpath.ManagedFields, error) {
	newObject, err := liveObject.Merge(configObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to merge config: %v", err)
	}
	managers, err = s.update(liveObject, newObject, version, managers, manager, force)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	newObject, err = s.removeDisownedItems(newObject, configObject, managers[manager])
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to remove fields: %v", err)
	}
	set, err := configObject.ToFieldSet()
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to get field set: %v", err)
	}
	managers[manager] = &fieldpath.VersionedSet{
		Set:        set,
		APIVersion: version,
	}
	if managers[manager].Set.Empty() {
		delete(managers, manager)
	}
	return newObject, managers, nil
}

func (s *Updater) removeDisownedItems(merged, applied typed.TypedValue, lastSet *fieldpath.VersionedSet) (typed.TypedValue, error) {
	if lastSet.Set.Empty() {
		return merged, nil
	}
	convertedApplied, err := s.Converter.Convert(applied, lastSet.APIVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert applied config to last applied version: %v", err)
	}
	appliedSet, err := convertedApplied.ToFieldSet()
	if err != nil {
		return nil, fmt.Errorf("failed to create field set from applied config in last applied version: %v", err)
	}
	return merged.RemoveItems(lastSet.Set.Difference(appliedSet)), nil
}
