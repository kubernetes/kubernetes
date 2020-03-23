/*
Copyright 2019 The Kubernetes Authors.

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

package fieldmanager

import (
	"fmt"
	"sort"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/structured-merge-diff/v3/fieldpath"
)

type capManagersManager struct {
	fieldManager          Manager
	maxUpdateManagers     int
	oldUpdatesManagerName string
}

var _ Manager = &capManagersManager{}

// NewCapManagersManager creates a new wrapped FieldManager which ensures that the number of managers from updates
// does not exceed maxUpdateManagers, by merging some of the oldest entries on each update.
func NewCapManagersManager(fieldManager Manager, maxUpdateManagers int) Manager {
	return &capManagersManager{
		fieldManager:          fieldManager,
		maxUpdateManagers:     maxUpdateManagers,
		oldUpdatesManagerName: "ancient-changes",
	}
}

// Update implements Manager.
func (f *capManagersManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	object, managed, err := f.fieldManager.Update(liveObj, newObj, managed, manager)
	if err != nil {
		return object, managed, err
	}
	if managed, err = f.capUpdateManagers(managed); err != nil {
		return nil, nil, fmt.Errorf("failed to cap update managers: %v", err)
	}
	return object, managed, nil
}

// Apply implements Manager.
func (f *capManagersManager) Apply(liveObj, appliedObj runtime.Object, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error) {
	return f.fieldManager.Apply(liveObj, appliedObj, managed, fieldManager, force)
}

// capUpdateManagers merges a number of the oldest update entries into versioned buckets,
// such that the number of entries from updates does not exceed f.maxUpdateManagers.
func (f *capManagersManager) capUpdateManagers(managed Managed) (newManaged Managed, err error) {
	// Gather all entries from updates
	updaters := []string{}
	for manager, fields := range managed.Fields() {
		if fields.Applied() == false {
			updaters = append(updaters, manager)
		}
	}
	if len(updaters) <= f.maxUpdateManagers {
		return managed, nil
	}

	// If we have more than the maximum, sort the update entries by time, oldest first.
	sort.Slice(updaters, func(i, j int) bool {
		iTime, jTime, iSeconds, jSeconds := managed.Times()[updaters[i]], managed.Times()[updaters[j]], int64(0), int64(0)
		if iTime != nil {
			iSeconds = iTime.Unix()
		}
		if jTime != nil {
			jSeconds = jTime.Unix()
		}
		if iSeconds != jSeconds {
			return iSeconds < jSeconds
		}
		return updaters[i] < updaters[j]
	})

	// Merge the oldest updaters with versioned bucket managers until the number of updaters is under the cap
	versionToFirstManager := map[string]string{}
	for i, length := 0, len(updaters); i < len(updaters) && length > f.maxUpdateManagers; i++ {
		manager := updaters[i]
		vs := managed.Fields()[manager]
		time := managed.Times()[manager]
		version := string(vs.APIVersion())

		// Create a new manager identifier for the versioned bucket entry.
		// The version for this manager comes from the version of the update being merged into the bucket.
		bucket, err := internal.BuildManagerIdentifier(&metav1.ManagedFieldsEntry{
			Manager:    f.oldUpdatesManagerName,
			Operation:  metav1.ManagedFieldsOperationUpdate,
			APIVersion: version,
		})
		if err != nil {
			return managed, fmt.Errorf("failed to create bucket manager for version %v: %v", version, err)
		}

		// Merge the fieldets if this is not the first time the version was seen.
		// Otherwise just record the manager name in versionToFirstManager
		if first, ok := versionToFirstManager[version]; ok {
			// If the bucket doesn't exists yet, create one.
			if _, ok := managed.Fields()[bucket]; !ok {
				s := managed.Fields()[first]
				delete(managed.Fields(), first)
				managed.Fields()[bucket] = s
			}

			managed.Fields()[bucket] = fieldpath.NewVersionedSet(vs.Set().Union(managed.Fields()[bucket].Set()), vs.APIVersion(), vs.Applied())
			delete(managed.Fields(), manager)
			length--

			// Use the time from the update being merged into the bucket, since it is more recent.
			managed.Times()[bucket] = time
		} else {
			versionToFirstManager[version] = manager
		}
	}

	return managed, nil
}
