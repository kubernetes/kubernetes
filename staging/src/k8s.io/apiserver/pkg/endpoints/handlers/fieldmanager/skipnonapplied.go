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
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type skipNonAppliedManager struct {
	fieldManager           Manager
	objectCreater          runtime.ObjectCreater
	gvk                    schema.GroupVersionKind
	beforeApplyManagerName string
}

var _ Manager = &skipNonAppliedManager{}

// NewSkipNonAppliedManager creates a new wrapped FieldManager that only starts tracking managers after the first apply.
func NewSkipNonAppliedManager(fieldManager Manager, objectCreater runtime.ObjectCreater, gvk schema.GroupVersionKind) Manager {
	return &skipNonAppliedManager{
		fieldManager:           fieldManager,
		objectCreater:          objectCreater,
		gvk:                    gvk,
		beforeApplyManagerName: "before-first-apply",
	}
}

// Update implements Manager.
func (f *skipNonAppliedManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	if len(managed.Fields()) == 0 {
		return newObj, managed, nil
	}
	return f.fieldManager.Update(liveObj, newObj, managed, manager)
}

// Apply implements Manager.
func (f *skipNonAppliedManager) Apply(liveObj runtime.Object, patch []byte, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error) {
	if len(managed.Fields()) == 0 {
		emptyObj, err := f.objectCreater.New(f.gvk)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create empty object of type %v: %v", f.gvk, err)
		}
		liveObj, managed, err = f.fieldManager.Update(emptyObj, liveObj, managed, f.beforeApplyManagerName)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create manager for existing fields: %v", err)
		}
	}
	return f.fieldManager.Apply(liveObj, patch, managed, fieldManager, force)
}
