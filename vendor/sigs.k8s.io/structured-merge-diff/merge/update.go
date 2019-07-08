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
	Convert(object *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error)
	IsMissingVersionError(error) bool
}

// Updater is the object used to compute updated FieldSets and also
// merge the object on Apply.
type Updater struct {
	Converter Converter
}

func (s *Updater) update(oldObject, newObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, workflow string, force bool) (fieldpath.ManagedFields, error) {
	conflicts := fieldpath.ManagedFields{}
	removed := fieldpath.ManagedFields{}
	compare, err := oldObject.Compare(newObject)
	if err != nil {
		return nil, fmt.Errorf("failed to compare objects: %v", err)
	}

	versions := map[fieldpath.APIVersion]*typed.Comparison{
		version: compare,
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
				return nil, fmt.Errorf("failed to convert old object: %v", err)
			}
			versionedNewObject, err := s.Converter.Convert(newObject, managerSet.APIVersion())
			if err != nil {
				if s.Converter.IsMissingVersionError(err) {
					delete(managers, manager)
					continue
				}
				return nil, fmt.Errorf("failed to convert new object: %v", err)
			}
			compare, err = versionedOldObject.Compare(versionedNewObject)
			if err != nil {
				return nil, fmt.Errorf("failed to compare objects: %v", err)
			}
			versions[managerSet.APIVersion()] = compare
		}

		conflictSet := fieldpath.Intersection(managerSet.Set(), fieldpath.Union(compare.Modified.Iterator(), compare.Added.Iterator()).Iterator())
		if !conflictSet.Empty() {
			conflicts[manager] = fieldpath.NewVersionedSet(conflictSet, managerSet.APIVersion(), false)
		}

		if !compare.Removed.Empty() {
			removed[manager] = fieldpath.NewVersionedSet(compare.Removed, managerSet.APIVersion(), false)
		}
	}

	if !force && len(conflicts) != 0 {
		return nil, ConflictsFromManagers(conflicts)
	}

	for manager, conflictSet := range conflicts {
		managers[manager] = fieldpath.NewVersionedSet(fieldpath.Difference(managers[manager].Set(), conflictSet.Set()), managers[manager].APIVersion(), managers[manager].Applied())
	}

	for manager, removedSet := range removed {
		managers[manager] = fieldpath.NewVersionedSet(fieldpath.Difference(managers[manager].Set(), removedSet.Set()), managers[manager].APIVersion(), managers[manager].Applied())
	}

	for manager := range managers {
		if managers[manager].Set().Empty() {
			delete(managers, manager)
		}
	}

	return managers, nil
}

// Update is the method you should call once you've merged your final
// object on CREATE/UPDATE/PATCH verbs. newObject must be the object
// that you intend to persist (after applying the patch if this is for a
// PATCH call), and liveObject must be the original object (empty if
// this is a CREATE call).
func (s *Updater) Update(liveObject, newObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string) (*typed.TypedValue, fieldpath.ManagedFields, error) {
	newObject, err := liveObject.NormalizeUnions(newObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	managers = shallowCopyManagers(managers)
	managers, err = s.update(liveObject, newObject, version, managers, manager, true)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	compare, err := liveObject.Compare(newObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to compare live and new objects: %v", err)
	}
	if _, ok := managers[manager]; !ok {
		managers[manager] = fieldpath.NewVersionedSet(fieldpath.NewSetAsList(), version, false)
	}
	managers[manager] = fieldpath.NewVersionedSet(
		fieldpath.Difference(
			fieldpath.Union(
				fieldpath.Union(
					managers[manager].Set(),
					compare.Modified.Iterator(),
				).Iterator(),
				compare.Added.Iterator(),
			).Iterator(),
			compare.Removed.Iterator(),
		),
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
// and return it.
func (s *Updater) Apply(liveObject, configObject *typed.TypedValue, version fieldpath.APIVersion, managers fieldpath.ManagedFields, manager string, force bool) (*typed.TypedValue, fieldpath.ManagedFields, error) {
	managers = shallowCopyManagers(managers)
	configObject, err := configObject.NormalizeUnionsApply(configObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	newObject, err := liveObject.Merge(configObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to merge config: %v", err)
	}
	newObject, err = configObject.NormalizeUnionsApply(newObject)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	set, err := configObject.ToFieldSet()
	if err != nil {
		return nil, fieldpath.ManagedFields{}, fmt.Errorf("failed to get field set: %v", err)
	}
	managers[manager] = fieldpath.NewVersionedSet(set, version, true)
	managers, err = s.update(liveObject, newObject, version, managers, manager, force)
	if err != nil {
		return nil, fieldpath.ManagedFields{}, err
	}
	return newObject, managers, nil
}

func shallowCopyManagers(managers fieldpath.ManagedFields) fieldpath.ManagedFields {
	newManagers := fieldpath.ManagedFields{}
	for manager, set := range managers {
		newManagers[manager] = set
	}
	return newManagers
}
