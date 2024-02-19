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

package fieldmanager

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

type managedFieldsUpdater struct {
	fieldManager Manager
}

var _ Manager = &managedFieldsUpdater{}

// NewManagedFieldsUpdater is responsible for updating the managedfields
// in the object, updating the time of the operation as necessary. For
// updates, it uses a hard-coded manager to detect if things have
// changed, and swaps back the correct manager after the operation is
// done.
func NewManagedFieldsUpdater(fieldManager Manager) Manager {
	return &managedFieldsUpdater{
		fieldManager: fieldManager,
	}
}

// Update implements Manager.
func (f *managedFieldsUpdater) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	self := "current-operation"
	object, managed, err := f.fieldManager.Update(liveObj, newObj, managed, self)
	if err != nil {
		return object, managed, err
	}

	// If the current operation took any fields from anything, it means the object changed,
	// so update the timestamp of the managedFieldsEntry and merge with any previous updates from the same manager
	if vs, ok := managed.Fields()[self]; ok {
		delete(managed.Fields(), self)

		if previous, ok := managed.Fields()[manager]; ok {
			managed.Fields()[manager] = fieldpath.NewVersionedSet(vs.Set().Union(previous.Set()), vs.APIVersion(), vs.Applied())
		} else {
			managed.Fields()[manager] = vs
		}

		managed.Times()[manager] = &metav1.Time{Time: time.Now().UTC()}
	}

	return object, managed, nil
}

// Apply implements Manager.
func (f *managedFieldsUpdater) Apply(liveObj, appliedObj runtime.Object, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error) {
	object, managed, err := f.fieldManager.Apply(liveObj, appliedObj, managed, fieldManager, force)
	if err != nil {
		return object, managed, err
	}
	if object != nil {
		managed.Times()[fieldManager] = &metav1.Time{Time: time.Now().UTC()}
	} else {
		object = liveObj.DeepCopyObject()
		internal.RemoveObjectManagedFields(object)
	}
	return object, managed, nil
}
