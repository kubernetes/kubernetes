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
	"net/http"
	"strings"
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8stypes "k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
)

// Stores info about the kubernetes node
type NodeInfo struct {
	dataCenter *vclib.Datacenter
	vm         *vclib.VirtualMachine
	vcServer   string
	vmUUID     string
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

	kubeClient clientset.Interface

	vmUUIDConfigMapName      string
	vmUUIDConfigMapNamespace string
	vmUUIDConfigMapCache     map[string]string
	vmUUIDConfigMapLock      sync.RWMutex
}

type NodeDetails struct {
	NodeName string
	vm       *vclib.VirtualMachine
	VMUUID   string
}

// TODO: Make it configurable in vsphere.conf
const (
	POOL_SIZE  = 8
	QUEUE_SIZE = POOL_SIZE * 10
)

func (nm *NodeManager) DiscoverNode(nodeName string) error {
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
	nodeUUID, err := nm.GetVMUUID(nodeName)
	if err != nil {
		glog.Errorf("Node Discovery failed to get node uuid for node %s with error: %v", nodeName, err)
		return err
	}

	glog.V(4).Infof("Discovering node %s with uuid %s", nodeName, nodeUUID)

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
				glog.V(4).Info("Discovering node error vc:", err)
				setGlobalErr(err)
				continue
			}

			if vsi.cfg.Datacenters == "" {
				datacenterObjs, err = vclib.GetAllDatacenter(ctx, vsi.conn)
				if err != nil {
					glog.V(4).Info("Discovering node error dc:", err)
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
						glog.V(4).Info("Discovering node error dc:", err)
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

				glog.V(4).Infof("Finding node %s in vc=%s and datacenter=%s", nodeName, vc, datacenterObj.Name())
				queueChannel <- &VmSearch{
					vc:         vc,
					datacenter: datacenterObj,
				}
			}
		}
		close(queueChannel)
	}()

	for i := 0; i < POOL_SIZE; i++ {
		go func() {
			for res := range queueChannel {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				vm, err := res.datacenter.GetVMByUUID(ctx, nodeUUID)
				if err != nil {
					glog.V(4).Infof("Error while looking for vm=%+v in vc=%s and datacenter=%s: %v",
						vm, res.vc, res.datacenter.Name(), err)
					if err != vclib.ErrNoVMFound {
						setGlobalErr(err)
					} else {
						glog.V(4).Infof("Did not find node %s in vc=%s and datacenter=%s",
							nodeName, res.vc, res.datacenter.Name())
					}
					continue
				}
				if vm != nil {
					glog.V(4).Infof("Found node %s as vm=%+v in vc=%s and datacenter=%s",
						nodeName, vm, res.vc, res.datacenter.Name())

					nodeInfo := &NodeInfo{dataCenter: res.datacenter, vm: vm, vcServer: res.vc, vmUUID: nodeUUID}
					nm.addNodeInfo(nodeName, nodeInfo)
					for range queueChannel {
					}
					setVMFound(true)
					break
				}
			}
			wg.Done()
		}()
		wg.Add(1)
	}
	wg.Wait()
	if vmFound {
		return nil
	}
	if globalErr != nil {
		return *globalErr
	}

	glog.V(4).Infof("Discovery Node: %q vm not found", nodeName)
	return vclib.ErrNoVMFound
}

func (nm *NodeManager) RegisterNode(node *v1.Node) error {
	nm.addNode(node)
	nodeUUID, err := GetNodeUUID(node)
	if err != nil {
		glog.Errorf("Failed to get vm uuid for node %s. err: %+v", node.Name, err)
		return err
	}
	nm.RegisterVMUUID(node.Name, nodeUUID)
	nm.DiscoverNode(node.Name)
	return nil
}

func (nm *NodeManager) UnRegisterNode(node *v1.Node) error {
	nm.removeNode(node)
	return nil
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

func (nm *NodeManager) RegisterVMUUID(nodeName string, vmUUID string) error {
	nm.vmUUIDConfigMapLock.Lock()
	defer nm.vmUUIDConfigMapLock.Unlock()

	configMap, err := nm.getVMUUIDConfigMap()
	if err != nil {
		glog.Errorf("Error while getting vm uuid configmap. err: %+v", err)
		return err
	}
	if configMap.Data == nil {
		configMap.Data = map[string]string{}
	}
	configMap.Data[nodeName] = vmUUID
	if err := nm.updateVMUUIDConfigMap(configMap); err != nil {
		glog.Errorf("Error while registering a new node. nodeName: %s, vmUUID: %s, err: %+v", nodeName, vmUUID, err)
		return err
	}
	return nil
}

func (nm *NodeManager) GetVMUUID(nodeName string) (string, error) {
	nm.vmUUIDConfigMapLock.RLock()
	defer nm.vmUUIDConfigMapLock.RUnlock()

	vmUUID, ok := nm.vmUUIDConfigMapCache[nodeName]
	if ok {
		return vmUUID, nil
	}
	glog.V(4).Infof("vm uuid for %s not found in cache, checking configmap", nodeName)
	configMap, err := nm.getVMUUIDConfigMap()
	if err != nil {
		glog.Errorf("Error while getting vm uuid configmap. err: %+v", err)
		return "", err
	}
	vmUUID, ok = configMap.Data[nodeName]
	if !ok {
		glog.Errorf("vm uuid for %s not found in configmap", nodeName)
		return "", err
	}
	glog.V(4).Infof("vm uuid for %s found in configmap. uuid: %s", nodeName, vmUUID)
	return vmUUID, nil
}

func (nm *NodeManager) getVMUUIDConfigMap() (*v1.ConfigMap, error) {
	if nm.kubeClient == nil {
		return nil, fmt.Errorf("kubeClient is not initialized yet")
	}
	configMap, err := nm.kubeClient.CoreV1().ConfigMaps(nm.vmUUIDConfigMapNamespace).Get(nm.vmUUIDConfigMapName, metav1.GetOptions{})
	if err != nil {
		statusErr, ok := err.(*apierrors.StatusError)
		if ok && statusErr.ErrStatus.Code == http.StatusNotFound {
			if err := nm.initializeVMUUIDConfigMap(); err != nil {
				return nil, err
			}
			return nm.getVMUUIDConfigMap()
		}
		glog.Errorf("Error while getting vm uuid configmap. err: %+v", err)
		return nil, err
	}
	if configMap.Data == nil {
		nm.vmUUIDConfigMapCache = make(map[string]string)
	} else {
		nm.vmUUIDConfigMapCache = configMap.Data
	}
	return configMap, nil
}
func (nm *NodeManager) updateVMUUIDConfigMap(configMap *v1.ConfigMap) error {
	if nm.kubeClient == nil {
		return fmt.Errorf("kubeClient is not initialized yet")
	}
	_, err := nm.kubeClient.CoreV1().ConfigMaps(nm.vmUUIDConfigMapNamespace).Update(configMap)
	if err != nil {
		glog.Errorf("Error while updating vm uuid configmap. err: %+v", err)
		return err
	}
	return nil
}

func (nm *NodeManager) initializeVMUUIDConfigMap() error {
	if nm.kubeClient == nil {
		return fmt.Errorf("kubeClient is not initialized yet")
	}
	_, err := nm.kubeClient.CoreV1().ConfigMaps(nm.vmUUIDConfigMapNamespace).Create(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: nm.vmUUIDConfigMapName,
		},
		Data: map[string]string{},
	})
	if err != nil {
		glog.Info("Failed to initialize vsphere vm uuid  configmap. err: %+v.", err)
		return err
	}
	return nil
}

// GetNodeInfo returns a NodeInfo which datacenter, vm and vc server ip address.
// This method returns an error if it is unable find node VCs and DCs listed in vSphere.conf
// NodeInfo returned may not be updated to reflect current VM location.
//
// This method is a getter but it can cause side-effect of updating NodeInfo object.
func (nm *NodeManager) GetNodeInfo(nodeName k8stypes.NodeName) (NodeInfo, error) {
	getNodeInfo := func(nodeName k8stypes.NodeName) *NodeInfo {
		nm.nodeInfoLock.RLock()
		nodeInfo := nm.nodeInfoMap[convertToString(nodeName)]
		nm.nodeInfoLock.RUnlock()
		return nodeInfo
	}
	nodeInfo := getNodeInfo(nodeName)
	var err error
	if nodeInfo == nil {
		// Rediscover node if no NodeInfo found.
		glog.V(4).Infof("No VM found for node %q. Initiating rediscovery.", convertToString(nodeName))
		err = nm.DiscoverNode(convertToString(nodeName))
		if err != nil {
			glog.Errorf("Error %q node info for node %q not found", err, convertToString(nodeName))
			return NodeInfo{}, err
		}
		nodeInfo = getNodeInfo(nodeName)
	} else {
		// Renew the found NodeInfo to avoid stale vSphere connection.
		glog.V(4).Infof("Renewing NodeInfo %+v for node %q", nodeInfo, convertToString(nodeName))
		nodeInfo, err = nm.renewNodeInfo(nodeInfo, true)
		if err != nil {
			glog.Errorf("Error %q occurred while renewing NodeInfo for %q", err, convertToString(nodeName))
			return NodeInfo{}, err
		}
		nm.addNodeInfo(convertToString(nodeName), nodeInfo)
	}
	return *nodeInfo, nil
}

// GetNodeDetails returns NodeDetails for all the discovered nodes.
//
// This method is a getter but it can cause side-effect of updating NodeInfo objects.
func (nm *NodeManager) GetNodeDetails() ([]NodeDetails, error) {
	nm.registeredNodesLock.Lock()
	defer nm.registeredNodesLock.Unlock()
	var nodeDetails []NodeDetails

	for nodeName := range nm.registeredNodes {
		nodeInfo, err := nm.GetNodeInfoWithNodeObject(nodeName)
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("Updated NodeInfo %v for node %q.", nodeInfo, nodeName)
		nodeDetails = append(nodeDetails, NodeDetails{nodeName, nodeInfo.vm, nodeInfo.vmUUID})
	}
	return nodeDetails, nil
}

func (nm *NodeManager) addNodeInfo(nodeName string, nodeInfo *NodeInfo) {
	nm.nodeInfoLock.Lock()
	nm.nodeInfoMap[nodeName] = nodeInfo
	nm.nodeInfoLock.Unlock()
}

func (nm *NodeManager) GetVSphereInstance(nodeName k8stypes.NodeName) (VSphereInstance, error) {
	nodeInfo, err := nm.GetNodeInfo(nodeName)
	if err != nil {
		glog.V(4).Infof("node info for node %q not found", convertToString(nodeName))
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
		glog.Errorf("Cannot connect to vCenter with err: %v", err)
		return err
	}

	glog.V(4).Infof("Invalid credentials. Cannot connect to server %q. "+
		"Fetching credentials from secrets.", vsphereInstance.conn.Hostname)

	// Get latest credentials from SecretCredentialManager
	credentials, err := credentialManager.GetCredential(vsphereInstance.conn.Hostname)
	if err != nil {
		glog.Errorf("Failed to get credentials from Secret Credential Manager with err: %v", err)
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
func (nm *NodeManager) GetNodeInfoWithNodeObject(nodeName string) (NodeInfo, error) {
	getNodeInfo := func(nodeName string) *NodeInfo {
		nm.nodeInfoLock.RLock()
		nodeInfo := nm.nodeInfoMap[nodeName]
		nm.nodeInfoLock.RUnlock()
		return nodeInfo
	}
	nodeInfo := getNodeInfo(nodeName)
	var err error
	if nodeInfo == nil {
		// Rediscover node if no NodeInfo found.
		glog.V(4).Infof("No VM found for node %q. Initiating rediscovery.", nodeName)
		err = nm.DiscoverNode(nodeName)
		if err != nil {
			glog.Errorf("Error %q node info for node %q not found", err, nodeName)
			return NodeInfo{}, err
		}
		nodeInfo = getNodeInfo(nodeName)
	} else {
		// Renew the found NodeInfo to avoid stale vSphere connection.
		glog.V(4).Infof("Renewing NodeInfo %+v for node %q", nodeInfo, nodeName)
		nodeInfo, err = nm.renewNodeInfo(nodeInfo, true)
		if err != nil {
			glog.Errorf("Error %q occurred while renewing NodeInfo for %q", err, nodeName)
			return NodeInfo{}, err
		}
		nm.addNodeInfo(nodeName, nodeInfo)
	}
	return *nodeInfo, nil
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
