/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

type NodeMapper struct {
}

type NodeInfo struct {
	Name              string
	DataCenterRef     types.ManagedObjectReference
	VirtualMachineRef types.ManagedObjectReference
	VSphere           *VSphere
}

var (
	nameToNodeInfo = make(map[string]*NodeInfo)
)

// GenerateNodeMap populates node name to node info map
func (nm *NodeMapper) GenerateNodeMap(vSphereInstances map[string]*VSphere, nodeList v1.NodeList) error {
	type VmSearch struct {
		vs         *VSphere
		datacenter *object.Datacenter
	}

	var wg sync.WaitGroup
	var queueChannel []*VmSearch

	var datacenters []*object.Datacenter
	var err error
	for _, vs := range vSphereInstances {

		// Create context
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		if vs.Config.Datacenters == "" {
			datacenters, err = vs.GetAllDatacenter(ctx)
			if err != nil {
				framework.Logf("NodeMapper error: %v", err)
				continue
			}
		} else {
			dcName := strings.Split(vs.Config.Datacenters, ",")
			for _, dc := range dcName {
				dc = strings.TrimSpace(dc)
				if dc == "" {
					continue
				}
				datacenter, err := vs.GetDatacenter(ctx, dc)
				if err != nil {
					framework.Logf("NodeMapper error dc: %s \n err: %v", dc, err)

					continue
				}
				datacenters = append(datacenters, datacenter)
			}
		}

		for _, dc := range datacenters {
			framework.Logf("Search candidates vc=%s and datacenter=%s", vs.Config.Hostname, dc.Name())
			queueChannel = append(queueChannel, &VmSearch{vs: vs, datacenter: dc})
		}
	}

	for _, node := range nodeList.Items {
		n := node
		go func() {
			nodeUUID := getUUIDFromProviderID(n.Spec.ProviderID)
			framework.Logf("Searching for node with UUID: %s", nodeUUID)
			for _, res := range queueChannel {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				vm, err := res.vs.GetVMByUUID(ctx, nodeUUID, res.datacenter)
				if err != nil {
					framework.Logf("Error %v while looking for node=%s in vc=%s and datacenter=%s",
						err, n.Name, res.vs.Config.Hostname, res.datacenter.Name())
					continue
				}
				if vm != nil {
					framework.Logf("Found node %s as vm=%+v in vc=%s and datacenter=%s",
						n.Name, vm, res.vs.Config.Hostname, res.datacenter.Name())
					nodeInfo := &NodeInfo{Name: n.Name, DataCenterRef: res.datacenter.Reference(), VirtualMachineRef: vm.Reference(), VSphere: res.vs}
					nm.SetNodeInfo(n.Name, nodeInfo)
					break
				}
			}
			wg.Done()
		}()
		wg.Add(1)
	}
	wg.Wait()

	if len(nameToNodeInfo) != len(nodeList.Items) {
		return errors.New("all nodes not mapped to respective vSphere")
	}
	return nil
}

// GetNodeInfo return NodeInfo for given nodeName
func (nm *NodeMapper) GetNodeInfo(nodeName string) *NodeInfo {
	return nameToNodeInfo[nodeName]
}

// SetNodeInfo sets NodeInfo for given nodeName. This function is not thread safe. Users need to handle concurrency.
func (nm *NodeMapper) SetNodeInfo(nodeName string, nodeInfo *NodeInfo) {
	nameToNodeInfo[nodeName] = nodeInfo
}
