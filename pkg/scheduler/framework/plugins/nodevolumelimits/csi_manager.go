/*
Copyright 2025 The Kubernetes Authors.

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

package nodevolumelimits

import (
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/labels"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// DefaultCSIManager is an implementation of the CSIManager interface.
type DefaultCSIManager struct {
	csiNodeLister *csiNodeListerWrapper
}

var _ fwk.CSIManager = &DefaultCSIManager{}

func NewCSIManager(csiNodeLister storagelisters.CSINodeLister) *DefaultCSIManager {
	return &DefaultCSIManager{csiNodeLister: NewCSINodeLister(csiNodeLister)}
}

func (m *DefaultCSIManager) CSINodes() fwk.CSINodeLister {
	return m.csiNodeLister
}

type csiNodeListerWrapper struct {
	csiNodeLister storagelisters.CSINodeLister
}

var _ fwk.CSINodeLister = &csiNodeListerWrapper{}

func NewCSINodeLister(csiNodeLister storagelisters.CSINodeLister) *csiNodeListerWrapper {
	return &csiNodeListerWrapper{csiNodeLister: csiNodeLister}
}

func (l *csiNodeListerWrapper) List() ([]*storagev1.CSINode, error) {
	return l.csiNodeLister.List(labels.Everything())
}

func (l *csiNodeListerWrapper) Get(name string) (*storagev1.CSINode, error) {
	return l.csiNodeLister.Get(name)
}
