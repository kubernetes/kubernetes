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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type skipNonAppliedManager struct {
	fieldManager  FieldManager
	objectCreater runtime.ObjectCreater
	gvk           schema.GroupVersionKind
}

var _ FieldManager = &skipNonAppliedManager{}

// NewSkipNonAppliedManager creates a new wrapped FieldManager that only starts tracking managers after the first apply
func NewSkipNonAppliedManager(fieldManager FieldManager, objectCreater runtime.ObjectCreater, gvk schema.GroupVersionKind) FieldManager {
	return &skipNonAppliedManager{
		fieldManager:  fieldManager,
		objectCreater: objectCreater,
		gvk:           gvk,
	}
}

// Update implements FieldManager.
func (f *skipNonAppliedManager) Update(liveObj, newObj runtime.Object, manager string) (runtime.Object, error) {
	liveObjAccessor, err := meta.Accessor(liveObj)
	if err != nil {
		return newObj, nil
	}
	newObjAccessor, err := meta.Accessor(newObj)
	if err != nil {
		return newObj, nil
	}
	if len(liveObjAccessor.GetManagedFields()) == 0 && len(newObjAccessor.GetManagedFields()) == 0 {
		return newObj, nil
	}

	return f.fieldManager.Update(liveObj, newObj, manager)
}

// Apply implements FieldManager.
func (f *skipNonAppliedManager) Apply(liveObj runtime.Object, patch []byte, fieldManager string, force bool) (runtime.Object, error) {
	liveObjAccessor, err := meta.Accessor(liveObj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	if len(liveObjAccessor.GetManagedFields()) == 0 {
		emptyObj, err := f.objectCreater.New(f.gvk)
		if err != nil {
			return nil, fmt.Errorf("failed to create empty object of type %v: %v", f.gvk, err)
		}
		liveObj, err = f.fieldManager.Update(emptyObj, liveObj, "before-first-apply")
		if err != nil {
			return nil, fmt.Errorf("failed to create manager for existing fields: %v", err)
		}
	}

	return f.fieldManager.Apply(liveObj, patch, fieldManager, force)
}
