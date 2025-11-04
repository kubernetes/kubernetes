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
	defaultCSINodeLister *csiNodeLister
}

var _ fwk.CSIManager = &DefaultCSIManager{}

func NewCSIManager(csiNodeLister storagelisters.CSINodeLister) *DefaultCSIManager {
	return &DefaultCSIManager{defaultCSINodeLister: NewCSINodeLister(csiNodeLister)}
}

func (m *DefaultCSIManager) CSINodes() fwk.CSINodeLister {
	return m.defaultCSINodeLister
}

type csiNodeLister struct {
	csiNodeLister storagelisters.CSINodeLister
}

var _ fwk.CSINodeLister = &csiNodeLister{}

func NewCSINodeLister(csinodeLister storagelisters.CSINodeLister) *csiNodeLister {
	return &csiNodeLister{csinodeLister: csinodeLister}
}

func (l *csiNodeLister) List() ([]*storagev1.CSINode, error) {
	return l.csinodeLister.List(labels.Everything())
}

func (l *csiNodeLister) Get(name string) (*storagev1.CSINode, error) {
	return l.csinodeLister.Get(name)
}
