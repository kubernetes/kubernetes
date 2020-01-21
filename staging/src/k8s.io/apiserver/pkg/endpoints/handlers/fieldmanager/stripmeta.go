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

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/structured-merge-diff/v3/fieldpath"
)

type stripMetaManager struct {
	fieldManager Manager

	// stripSet is the list of fields that should never be part of a mangedFields.
	stripSet *fieldpath.Set
}

var _ Manager = &stripMetaManager{}

// NewStripMetaManager creates a new Manager that strips metadata and typemeta fields from the manager's fieldset.
func NewStripMetaManager(fieldManager Manager) Manager {
	return &stripMetaManager{
		fieldManager: fieldManager,
		stripSet: fieldpath.NewSet(
			fieldpath.MakePathOrDie("apiVersion"),
			fieldpath.MakePathOrDie("kind"),
			fieldpath.MakePathOrDie("metadata"),
			fieldpath.MakePathOrDie("metadata", "name"),
			fieldpath.MakePathOrDie("metadata", "namespace"),
			fieldpath.MakePathOrDie("metadata", "creationTimestamp"),
			fieldpath.MakePathOrDie("metadata", "selfLink"),
			fieldpath.MakePathOrDie("metadata", "uid"),
			fieldpath.MakePathOrDie("metadata", "clusterName"),
			fieldpath.MakePathOrDie("metadata", "generation"),
			fieldpath.MakePathOrDie("metadata", "managedFields"),
			fieldpath.MakePathOrDie("metadata", "resourceVersion"),
		),
	}
}

// Update implements Manager.
func (f *stripMetaManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	newObj, managed, err := f.fieldManager.Update(liveObj, newObj, managed, manager)
	if err != nil {
		return nil, nil, err
	}
	f.stripFields(managed.Fields(), manager)
	return newObj, managed, nil
}

// Apply implements Manager.
func (f *stripMetaManager) Apply(liveObj, appliedObj runtime.Object, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	newObj, managed, err := f.fieldManager.Apply(liveObj, appliedObj, managed, manager, force)
	if err != nil {
		return nil, nil, err
	}
	f.stripFields(managed.Fields(), manager)
	return newObj, managed, nil
}

// stripFields removes a predefined set of paths found in typed from managed
func (f *stripMetaManager) stripFields(managed fieldpath.ManagedFields, manager string) {
	vs, ok := managed[manager]
	if ok {
		if vs == nil {
			panic(fmt.Sprintf("Found unexpected nil manager which should never happen: %s", manager))
		}
		newSet := vs.Set().Difference(f.stripSet)
		if newSet.Empty() {
			delete(managed, manager)
		} else {
			managed[manager] = fieldpath.NewVersionedSet(newSet, vs.APIVersion(), vs.Applied())
		}
	}
}
