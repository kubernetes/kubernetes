// +build !providerless

/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"
	"sync"

	"github.com/vmware/govmomi/object"
	v1 "k8s.io/api/core/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/klog/v2"
	"k8s.io/legacy-cloud-providers/vsphere/vclib"
)

// Stores info about the kubernetes node
type NodeInfo struct {
	dataCenter *vclib.Datacenter
	vm         *vclib.VirtualMachine
	vcServer   string
	vmUUID     string
	zone       *cloudprovider.Zone
}

func (n NodeInfo) String() string {
	return fmt.Sprintf("{datacenter: %v, vm: %v, vcServer: %s, vmUUID: %s, zone: %v}",
		*n.dataCenter, n.vm.Reference(), n.vcServer, n.vmUUID, *n.zone)
}

type NodeManager struct {
	// TODO: replace map with concurrent map when k8s supports go v1.9

	// Maps the VC server to VSphereInstance
	vsphereInstanceMap map[string]*VSphereInstance
	// Maps node name to node info.
	nodeInfoMap map[string]*NodeInfo
	// Maps node name to node structure
	registeredNodes map[string]*v1.Node
	//CredentialsManager
	credentialManager *SecretCredentialManager

	// Mutexes
	registeredNodesLock   sync.RWMutex
	nodeInfoLock          sync.RWMutex
	credentialManagerLock sync.Mutex
}

type NodeDetails struct {
	NodeName string
	vm       *vclib.VirtualMachine
	VMUUID   string
	Zone     *cloudprovider.Zone
}

// TODO: Make it configurable in vsphere.conf
const (
	POOL_SIZE  = 8
	QUEUE_SIZE = POOL_SIZE * 10
)

func (nm *NodeManager) DiscoverNode(node *v1.Node) error {
	type VmSearch struct {
		vc         string
		datacenter *vclib.Datacenter
	}

	var mutex = &sync.Mutex{}
	var globalErrMutex = &sync.Mutex{}
	var queueChannel chan *VmSearch
	var wg sync.WaitGroup
	var globalErr *error

	queueChannel = make(chan *VmSearch, QUEUE_SIZE)
	nodeUUID, err := GetNodeUUID(node)
	if err != nil {
		klog.Errorf("Node Discovery failed to get node uuid for node %s with error: %v", node.Name, err)
		return err
	}

	klog.V(4).Infof("Discovering node %s with uuid %s", node.ObjectMeta.Name, nodeUUID)

	vmFound := false
	globalErr = nil

	setGlobalErr := func(err error) {
		globalErrMutex.Lock()
		globalErr = &err
		globalErrMutex.Unlock()
	}

	setVMFound := func(found bool) {
		mutex.Lock()
		vmFound = found
		mutex.Unlock()
	}

	getVMFound := func() bool {
		mutex.Lock()
		found := vmFound
		mutex.Unlock()
		return found
	}

	go func() {
		var datacenterObjs []*vclib.Datacenter
		for vc, vsi := range nm.vsphereInstanceMap {

			found := getVMFound()
			if found == true {
				break
			}

			// Create context
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			err := nm.vcConnect(ctx, vsi)
			if err != nil {
				klog.V(4).Info("Discovering node error vc:", err)
				setGlobalErr(err)
				continue
			}

			if vsi.cfg.Datacenters == "" {
				datacenterObjs, err = vclib.GetAllDatacenter(ctx, vsi.conn)
				if err != nil {
					klog.V(4).Info("Discovering node error dc:", err)
					setGlobalErr(err)
					continue
				}
			} else {
				datacenters := strings.Split(vsi.cfg.Datacenters, ",")
				for _, dc := range datacenters {
					dc = strings.TrimSpace(dc)
					if dc == "" {
						continue
					}
					datacenterObj, err := vclib.GetDatacenter(ctx, vsi.conn, dc)
					if err != nil {
						klog.V(4).Info("Discovering node error dc:", err)
						setGlobalErr(err)
						continue
					}
					datacenterObjs = append(datacenterObjs, datacenterObj)
				}
			}

			for _, datacenterObj := range datacenterObjs {
				found := getVMFound()
				if found == true {
					break
				}

				klog.V(4).Infof("Finding node %s in vc=%s and datacenter=%s", node.Name, vc, datacenterObj.Name())
				queueChannel <- &VmSearch{
					vc:         vc,
					datacenter: datacenterObj,
				}
			}
		}
		close(queueChannel)
	}()

	for i := 0; i < POOL_SIZE; i++ {
		wg.Add(1)
		go func() {
			for res := range queueChannel {
				ctx, cancel := context.WithCancel(context.Background())
				vm, err := res.datacenter.GetVMByUUID(ctx, nodeUUID)
				if err != nil {
					klog.V(4).Infof("Error while looking for vm=%+v in vc=%s and datacenter=%s: %v",
						vm, res.vc, res.datacenter.Name(), err)
					if err != vclib.ErrNoVMFound {
						setGlobalErr(err)
					} else {
						klog.V(4).Infof("Did not find node %s in vc=%s and datacenter=%s",
							node.Name, res.vc, res.datacenter.Name())
					}
					cancel()
					continue
				}
				if vm != nil {
					klog.V(4).Infof("Found node %s as vm=%+v in vc=%s and datacenter=%s",
						node.Name, vm, res.vc, res.datacenter.Name())

					// Get the node zone information
					nodeFd := node.ObjectMeta.Labels[v1.LabelZoneFailureDomain]
					nodeRegion := node.ObjectMeta.Labels[v1.LabelZoneRegion]
					nodeZone := &cloudprovider.Zone{FailureDomain: nodeFd, Region: nodeRegion}
					nodeInfo := &NodeInfo{dataCenter: res.datacenter, vm: vm, vcServer: res.vc, vmUUID: nodeUUID, zone: nodeZone}
					nm.addNodeInfo(node.ObjectMeta.Name, nodeInfo)
					for range queueChannel {
					}
					setVMFound(true)
					cancel()
					break
				}
			}
			wg.Done()
		}()
	}
	wg.Wait()
	if vmFound {
		return nil
	}
	if globalErr != nil {
		return *globalErr
	}

	klog.V(4).Infof("Discovery Node: %q vm not found", node.Name)
	return vclib.ErrNoVMFound
}

func (nm *NodeManager) RegisterNode(node *v1.Node) error {
	nm.addNode(node)
	return nm.DiscoverNode(node)
}

func (nm *NodeManager) UnRegisterNode(node *v1.Node) error {
	nm.removeNode(node)
	return nil
}

func (nm *NodeManager) RediscoverNode(nodeName k8stypes.NodeName) error {
	node, err := nm.GetNode(nodeName)

	if err != nil {
		return err
	}
	return nm.DiscoverNode(&node)
}

func (nm *NodeManager) GetNode(nodeName k8stypes.NodeName) (v1.Node, error) {
	nm.registeredNodesLock.RLock()
	node := nm.registeredNodes[convertToString(nodeName)]
	nm.registeredNodesLock.RUnlock()
	if node == nil {
		return v1.Node{}, vclib.ErrNoVMFound
	}
	return *node, nil
}

func (nm *NodeManager) getNodes() map[string]*v1.Node {
	nm.registeredNodesLock.RLock()
	defer nm.registeredNodesLock.RUnlock()
	registeredNodes := make(map[string]*v1.Node, len(nm.registeredNodes))
	for nodeName, node := range nm.registeredNodes {
		registeredNodes[nodeName] = node
	}
	return registeredNodes
}

func (nm *NodeManager) addNode(node *v1.Node) {
	nm.registeredNodesLock.Lock()
	nm.registeredNodes[node.ObjectMeta.Name] = node
	nm.registeredNodesLock.Unlock()
}

func (nm *NodeManager) removeNode(node *v1.Node) {
	nm.registeredNodesLock.Lock()
	delete(nm.registeredNodes, node.ObjectMeta.Name)
	nm.registeredNodesLock.Unlock()

	nm.nodeInfoLock.Lock()
	delete(nm.nodeInfoMap, node.ObjectMeta.Name)
	nm.nodeInfoLock.Unlock()
}

// GetNodeInfo returns a NodeInfo which datacenter, vm and vc server ip address.
// This method returns an error if it is unable find node VCs and DCs listed in vSphere.conf
// NodeInfo returned may not be updated to reflect current VM location.
//
// This method is a getter but it can cause side-effect of updating NodeInfo object.
func (nm *NodeManager) GetNodeInfo(nodeName k8stypes.NodeName) (NodeInfo, error) {
	return nm.getRefreshedNodeInfo(nodeName)
}

// GetNodeDetails returns NodeDetails for all the discovered nodes.
//
// This method is a getter but it can cause side-effect of updating NodeInfo objects.
func (nm *NodeManager) GetNodeDetails() ([]NodeDetails, error) {
	var nodeDetails []NodeDetails

	for nodeName, nodeObj := range nm.getNodes() {
		nodeInfo, err := nm.GetNodeInfoWithNodeObject(nodeObj)
		if err != nil {
			return nil, err
		}
		klog.V(4).Infof("Updated NodeInfo %v for node %q.", nodeInfo, nodeName)
		nodeDetails = append(nodeDetails, NodeDetails{nodeName, nodeInfo.vm, nodeInfo.vmUUID, nodeInfo.zone})
	}
	return nodeDetails, nil
}

func (nm *NodeManager) refreshNodes() (errList []error) {
	for nodeName := range nm.getNodes() {
		nodeInfo, err := nm.getRefreshedNodeInfo(convertToK8sType(nodeName))
		if err != nil {
			errList = append(errList, err)
			continue
		}
		klog.V(4).Infof("Updated NodeInfo %v for node %q.", nodeInfo, nodeName)
	}
	return errList
}

func (nm *NodeManager) getRefreshedNodeInfo(nodeName k8stypes.NodeName) (NodeInfo, error) {
	nodeInfo := nm.getNodeInfo(nodeName)
	var err error
	if nodeInfo == nil {
		// Rediscover node if no NodeInfo found.
		klog.V(4).Infof("No VM found for node %q. Initiating rediscovery.", convertToString(nodeName))
		err = nm.RediscoverNode(nodeName)
		if err != nil {
			klog.Errorf("Error %q node info for node %q not found", err, convertToString(nodeName))
			return NodeInfo{}, err
		}
		nodeInfo = nm.getNodeInfo(nodeName)
	} else {
		// Renew the found NodeInfo to avoid stale vSphere connection.
		klog.V(4).Infof("Renewing NodeInfo %+v for node %q", nodeInfo, convertToString(nodeName))
		nodeInfo, err = nm.renewNodeInfo(nodeInfo, true)
		if err != nil {
			klog.Errorf("Error %q occurred while renewing NodeInfo for %q", err, convertToString(nodeName))
			return NodeInfo{}, err
		}
		nm.addNodeInfo(convertToString(nodeName), nodeInfo)
	}
	return *nodeInfo, nil
}

func (nm *NodeManager) addNodeInfo(nodeName string, nodeInfo *NodeInfo) {
	nm.nodeInfoLock.Lock()
	nm.nodeInfoMap[nodeName] = nodeInfo
	nm.nodeInfoLock.Unlock()
}

func (nm *NodeManager) getNodeInfo(nodeName k8stypes.NodeName) *NodeInfo {
	nm.nodeInfoLock.RLock()
	nodeInfo := nm.nodeInfoMap[convertToString(nodeName)]
	nm.nodeInfoLock.RUnlock()
	return nodeInfo
}

func (nm *NodeManager) GetVSphereInstance(nodeName k8stypes.NodeName) (VSphereInstance, error) {
	nodeInfo, err := nm.GetNodeInfo(nodeName)
	if err != nil {
		klog.V(4).Infof("node info for node %q not found", convertToString(nodeName))
		return VSphereInstance{}, err
	}
	vsphereInstance := nm.vsphereInstanceMap[nodeInfo.vcServer]
	if vsphereInstance == nil {
		return VSphereInstance{}, fmt.Errorf("vSphereInstance for vc server %q not found while looking for node %q", nodeInfo.vcServer, convertToString(nodeName))
	}
	return *vsphereInstance, nil
}

// renewNodeInfo renews vSphere connection, VirtualMachine and Datacenter for NodeInfo instance.
func (nm *NodeManager) renewNodeInfo(nodeInfo *NodeInfo, reconnect bool) (*NodeInfo, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vsphereInstance := nm.vsphereInstanceMap[nodeInfo.vcServer]
	if vsphereInstance == nil {
		err := fmt.Errorf("vSphereInstance for vSphere %q not found while refershing NodeInfo for VM %q", nodeInfo.vcServer, nodeInfo.vm)
		return nil, err
	}
	if reconnect {
		err := nm.vcConnect(ctx, vsphereInstance)
		if err != nil {
			return nil, err
		}
	}
	vm := nodeInfo.vm.RenewVM(vsphereInstance.conn.Client)
	return &NodeInfo{
		vm:         &vm,
		dataCenter: vm.Datacenter,
		vcServer:   nodeInfo.vcServer,
		vmUUID:     nodeInfo.vmUUID,
		zone:       nodeInfo.zone,
	}, nil
}

func (nodeInfo *NodeInfo) VM() *vclib.VirtualMachine {
	if nodeInfo == nil {
		return nil
	}
	return nodeInfo.vm
}

// vcConnect connects to vCenter with existing credentials
// If credentials are invalid:
// 		1. It will fetch credentials from credentialManager
//      2. Update the credentials
//		3. Connects again to vCenter with fetched credentials
func (nm *NodeManager) vcConnect(ctx context.Context, vsphereInstance *VSphereInstance) error {
	err := vsphereInstance.conn.Connect(ctx)
	if err == nil {
		return nil
	}

	credentialManager := nm.CredentialManager()
	if !vclib.IsInvalidCredentialsError(err) || credentialManager == nil {
		klog.Errorf("Cannot connect to vCenter with err: %v", err)
		return err
	}

	klog.V(4).Infof("Invalid credentials. Cannot connect to server %q. Fetching credentials from secrets.", vsphereInstance.conn.Hostname)

	// Get latest credentials from SecretCredentialManager
	credentials, err := credentialManager.GetCredential(vsphereInstance.conn.Hostname)
	if err != nil {
		klog.Errorf("Failed to get credentials from Secret Credential Manager with err: %v", err)
		return err
	}
	vsphereInstance.conn.UpdateCredentials(credentials.User, credentials.Password)
	return vsphereInstance.conn.Connect(ctx)
}

// GetNodeInfoWithNodeObject returns a NodeInfo which datacenter, vm and vc server ip address.
// This method returns an error if it is unable find node VCs and DCs listed in vSphere.conf
// NodeInfo returned may not be updated to reflect current VM location.
//
// This method is a getter but it can cause side-effect of updating NodeInfo object.
func (nm *NodeManager) GetNodeInfoWithNodeObject(node *v1.Node) (NodeInfo, error) {
	return nm.getRefreshedNodeInfo(convertToK8sType(node.Name))
}

func (nm *NodeManager) CredentialManager() *SecretCredentialManager {
	nm.credentialManagerLock.Lock()
	defer nm.credentialManagerLock.Unlock()
	return nm.credentialManager
}

func (nm *NodeManager) UpdateCredentialManager(credentialManager *SecretCredentialManager) {
	nm.credentialManagerLock.Lock()
	defer nm.credentialManagerLock.Unlock()
	nm.credentialManager = credentialManager
}

func (nm *NodeManager) GetHostsInZone(ctx context.Context, zoneFailureDomain string) ([]*object.HostSystem, error) {
	klog.V(9).Infof("GetHostsInZone called with registeredNodes: %v", nm.registeredNodes)
	nodeDetails, err := nm.GetNodeDetails()
	if err != nil {
		return nil, err
	}
	klog.V(4).Infof("Node Details: %v", nodeDetails)
	// Build a map of Host moRef to HostSystem
	hostMap := make(map[string]*object.HostSystem)
	for _, n := range nodeDetails {
		// Match the provided zone failure domain with the node.
		klog.V(9).Infof("Matching provided zone %s with node %s zone %s", zoneFailureDomain, n.NodeName, n.Zone.FailureDomain)
		if zoneFailureDomain == n.Zone.FailureDomain {
			host, err := n.vm.HostSystem(ctx)
			if err != nil {
				klog.Errorf("Failed to get host system for VM %s. err: %+v", n.vm, err)
				continue
			}
			hostMap[host.Reference().Value] = host
		}
	}
	// Build the unique list of hosts.
	hosts := make([]*object.HostSystem, 0)
	for _, value := range hostMap {
		hosts = append(hosts, value)
	}
	klog.V(4).Infof("GetHostsInZone %v returning: %v", zoneFailureDomain, hosts)
	return hosts, nil
}
