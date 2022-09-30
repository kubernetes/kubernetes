/*
Copyright 2022 The Kubernetes Authors.

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

// This file helps to detect whether we changed the interface of Listers.
// This is important for downstream projects who import the Listers to not
// breaking their codes.

package contract

import (
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

var _ framework.NodeInfoLister = &nodeInfoListerContract{}
var _ framework.StorageInfoLister = &storageInfoListerContract{}
var _ framework.SharedLister = &shareListerContract{}

type nodeInfoListerContract struct{}

func (c *nodeInfoListerContract) List() ([]*framework.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) HavePodsWithAffinityList() ([]*framework.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) HavePodsWithRequiredAntiAffinityList() ([]*framework.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) Get(_ string) (*framework.NodeInfo, error) {
	return nil, nil
}

type storageInfoListerContract struct{}

func (c *storageInfoListerContract) IsPVCUsedByPods(_ string) bool {
	return false
}

type shareListerContract struct{}

func (c *shareListerContract) NodeInfos() framework.NodeInfoLister {
	return nil
}

func (c *shareListerContract) StorageInfos() framework.StorageInfoLister {
	return nil
}
