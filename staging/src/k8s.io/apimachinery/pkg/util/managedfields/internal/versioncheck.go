/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type versionCheckManager struct {
	fieldManager Manager
	gvk          schema.GroupVersionKind
}

var _ Manager = &versionCheckManager{}

// NewVersionCheckManager creates a manager that makes sure that the
// applied object is in the proper version.
func NewVersionCheckManager(fieldManager Manager, gvk schema.GroupVersionKind) Manager {
	return &versionCheckManager{fieldManager: fieldManager, gvk: gvk}
}

// Update implements Manager.
func (f *versionCheckManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	// Nothing to do for updates, this is checked in many other places.
	return f.fieldManager.Update(liveObj, newObj, managed, manager)
}

// Apply implements Manager.
func (f *versionCheckManager) Apply(liveObj, appliedObj runtime.Object, managed Managed, fieldManager string, force bool) (runtime.Object, Managed, error) {
	if gvk := appliedObj.GetObjectKind().GroupVersionKind(); gvk != f.gvk {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid object type: %v", gvk))
	}
	return f.fieldManager.Apply(liveObj, appliedObj, managed, fieldManager, force)
}
