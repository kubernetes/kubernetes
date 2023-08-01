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

package internal

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type buildManagerInfoManager struct {
	fieldManager Manager
	groupVersion schema.GroupVersion
	subresource  string
}

var _ Manager = &buildManagerInfoManager{}

// NewBuildManagerInfoManager creates a new Manager that converts the manager name into a unique identifier
// combining operation and version for update requests, and just operation for apply requests.
func NewBuildManagerInfoManager(f Manager, gv schema.GroupVersion, subresource string) Manager {
	return &buildManagerInfoManager{
		fieldManager: f,
		groupVersion: gv,
		subresource:  subresource,
	}
}

// Update implements Manager.
func (f *buildManagerInfoManager) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	manager, err := f.buildManagerInfo(manager, metav1.ManagedFieldsOperationUpdate)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build manager identifier: %v", err)
	}
	return f.fieldManager.Update(liveObj, newObj, managed, manager)
}

// Apply implements Manager.
func (f *buildManagerInfoManager) Apply(liveObj, appliedObj runtime.Object, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	manager, err := f.buildManagerInfo(manager, metav1.ManagedFieldsOperationApply)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build manager identifier: %v", err)
	}
	return f.fieldManager.Apply(liveObj, appliedObj, managed, manager, force)
}

func (f *buildManagerInfoManager) buildManagerInfo(prefix string, operation metav1.ManagedFieldsOperationType) (string, error) {
	managerInfo := metav1.ManagedFieldsEntry{
		Manager:     prefix,
		Operation:   operation,
		APIVersion:  f.groupVersion.String(),
		Subresource: f.subresource,
	}
	if managerInfo.Manager == "" {
		managerInfo.Manager = "unknown"
	}
	return BuildManagerIdentifier(&managerInfo)
}
