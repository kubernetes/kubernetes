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

// CSIManagerImpl is a implementation of the CSIManager interface.
type CSIManagerImpl struct {
	csiNodeListerImpl *csiNodeLister
}

var _ fwk.CSIManager = &CSIManagerImpl{}

func NewCSIManager(csinodeLister storagelisters.CSINodeLister) *CSIManagerImpl {
	return &CSIManagerImpl{csiNodeListerImpl: NewCsiNodeLister(csinodeLister)}
}

func (m *CSIManagerImpl) CSINodes() fwk.CSINodeLister {
	return m.csiNodeListerImpl
}

type csiNodeLister struct {
	csinodeLister storagelisters.CSINodeLister
}

var _ fwk.CSINodeLister = &csiNodeLister{}

func NewCsiNodeLister(csinodeLister storagelisters.CSINodeLister) *csiNodeLister {
	return &csiNodeLister{csinodeLister: csinodeLister}
}

func (l *csiNodeLister) List() ([]*storagev1.CSINode, error) {
	return l.csinodeLister.List(labels.Everything())
}

func (l *csiNodeLister) Get(name string) (*storagev1.CSINode, error) {
	return l.csinodeLister.Get(name)
}
