//go:build !providerless
// +build !providerless

/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"k8s.io/klog/v2"
	"k8s.io/legacy-cloud-providers/vsphere/vclib"
)

type sharedDatastore struct {
	nodeManager         *NodeManager
	candidateDatastores []*vclib.DatastoreInfo
}

type hostInfo struct {
	hostUUID   string
	hostMOID   string
	datacenter string
}

const (
	summary       = "summary"
	runtimeHost   = "summary.runtime.host"
	hostsProperty = "host"
	nameProperty  = "name"
)

func (shared *sharedDatastore) getSharedDatastore(ctcx context.Context) (*vclib.DatastoreInfo, error) {
	nodes := shared.nodeManager.getNodes()

	// Segregate nodes according to VC-DC
	dcNodes := make(map[string][]NodeInfo)
	nodeHosts := make(map[string]hostInfo)

	for nodeName, node := range nodes {
		nodeInfo, err := shared.nodeManager.GetNodeInfoWithNodeObject(node)
		if err != nil {
			return nil, fmt.Errorf("unable to find node %s: %v", nodeName, err)
		}
		vcDC := nodeInfo.vcServer + nodeInfo.dataCenter.String()
		dcNodes[vcDC] = append(dcNodes[vcDC], nodeInfo)
	}

	for vcDC, nodes := range dcNodes {
		var hostInfos []hostInfo
		var err error
		hostInfos, err = shared.getNodeHosts(ctcx, nodes, vcDC)
		if err != nil {
			if vclib.IsManagedObjectNotFoundError(err) {
				klog.Warningf("SharedHost.getSharedDatastore: batch fetching of hosts failed - switching to fetching them individually.")
				hostInfos, err = shared.getEachNodeHost(ctcx, nodes, vcDC)
				if err != nil {
					klog.Errorf("SharedHost.getSharedDatastore: error fetching node hosts individually: %v", err)
					return nil, err
				}
			} else {
				return nil, err
			}
		}
		for _, host := range hostInfos {
			hostDCName := fmt.Sprintf("%s/%s", host.datacenter, host.hostMOID)
			nodeHosts[hostDCName] = host
		}
	}

	if len(nodeHosts) < 1 {
		msg := fmt.Sprintf("SharedHost.getSharedDatastore unable to find hosts associated with nodes")
		klog.Error(msg)
		return nil, fmt.Errorf("")
	}

	for _, datastoreInfo := range shared.candidateDatastores {
		dataStoreHosts, err := shared.getAttachedHosts(ctcx, datastoreInfo.Datastore)
		if err != nil {
			msg := fmt.Sprintf("error finding attached hosts to datastore %s: %v", datastoreInfo.Name(), err)
			klog.Error(msg)
			return nil, fmt.Errorf(msg)
		}
		if shared.isIncluded(dataStoreHosts, nodeHosts) {
			return datastoreInfo, nil
		}
	}
	return nil, fmt.Errorf("SharedHost.getSharedDatastore: unable to find any shared datastores")
}

// check if all of the nodeHosts are included in the dataStoreHosts
func (shared *sharedDatastore) isIncluded(dataStoreHosts []hostInfo, nodeHosts map[string]hostInfo) bool {
	result := true
	for _, host := range nodeHosts {
		hostFound := false
		for _, targetHost := range dataStoreHosts {
			if host.hostUUID == targetHost.hostUUID && host.hostMOID == targetHost.hostMOID {
				hostFound = true
			}
		}
		if !hostFound {
			result = false
		}
	}
	return result
}

func (shared *sharedDatastore) getEachNodeHost(ctx context.Context, nodes []NodeInfo, dcVC string) ([]hostInfo, error) {
	var hosts []hostInfo
	for _, node := range nodes {
		host, err := node.vm.GetHost(ctx)
		if err != nil {
			klog.Errorf("SharedHost.getEachNodeHost: unable to find host for vm %s: %v", node.vm.InventoryPath, err)
			return nil, err
		}
		hosts = append(hosts, hostInfo{
			hostUUID:   host.Summary.Hardware.Uuid,
			hostMOID:   host.Summary.Host.String(),
			datacenter: node.dataCenter.String(),
		})
	}
	return hosts, nil
}

func (shared *sharedDatastore) getNodeHosts(ctx context.Context, nodes []NodeInfo, dcVC string) ([]hostInfo, error) {
	var vmRefs []types.ManagedObjectReference
	if len(nodes) < 1 {
		return nil, fmt.Errorf("no nodes found for dc-vc: %s", dcVC)
	}
	var nodeInfo NodeInfo
	for _, n := range nodes {
		nodeInfo = n
		vmRefs = append(vmRefs, n.vm.Reference())
	}
	pc := property.DefaultCollector(nodeInfo.dataCenter.Client())
	var vmoList []mo.VirtualMachine
	err := pc.Retrieve(ctx, vmRefs, []string{nameProperty, runtimeHost}, &vmoList)
	if err != nil {
		klog.Errorf("SharedHost.getNodeHosts: unable to fetch vms from datacenter %s: %v", nodeInfo.dataCenter.String(), err)
		return nil, err
	}
	var hostMoList []mo.HostSystem
	var hostRefs []types.ManagedObjectReference
	for _, vmo := range vmoList {
		if vmo.Summary.Runtime.Host == nil {
			msg := fmt.Sprintf("SharedHost.getNodeHosts: no host associated with vm %s", vmo.Name)
			klog.Error(msg)
			return nil, fmt.Errorf(msg)
		}
		hostRefs = append(hostRefs, vmo.Summary.Runtime.Host.Reference())
	}
	pc = property.DefaultCollector(nodeInfo.dataCenter.Client())
	err = pc.Retrieve(ctx, hostRefs, []string{summary}, &hostMoList)
	if err != nil {
		klog.Errorf("SharedHost.getNodeHosts: unable to fetch hosts from datacenter %s: %v", nodeInfo.dataCenter.String(), err)
		return nil, err
	}
	var hosts []hostInfo
	for _, host := range hostMoList {
		hosts = append(hosts, hostInfo{hostMOID: host.Summary.Host.String(), hostUUID: host.Summary.Hardware.Uuid, datacenter: nodeInfo.dataCenter.String()})
	}
	return hosts, nil
}

func (shared *sharedDatastore) getAttachedHosts(ctx context.Context, datastore *vclib.Datastore) ([]hostInfo, error) {
	var ds mo.Datastore

	pc := property.DefaultCollector(datastore.Client())
	err := pc.RetrieveOne(ctx, datastore.Reference(), []string{hostsProperty}, &ds)
	if err != nil {
		return nil, err
	}

	mounts := make(map[types.ManagedObjectReference]types.DatastoreHostMount)
	var refs []types.ManagedObjectReference
	for _, host := range ds.Host {
		refs = append(refs, host.Key)
		mounts[host.Key] = host
	}

	var hs []mo.HostSystem
	err = pc.Retrieve(ctx, refs, []string{summary}, &hs)
	if err != nil {
		return nil, err
	}
	var hosts []hostInfo
	for _, h := range hs {
		hosts = append(hosts, hostInfo{hostUUID: h.Summary.Hardware.Uuid, hostMOID: h.Summary.Host.String()})
	}
	return hosts, nil

}
