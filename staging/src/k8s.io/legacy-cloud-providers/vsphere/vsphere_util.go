// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/legacy-cloud-providers/vsphere/vclib"
	"k8s.io/legacy-cloud-providers/vsphere/vclib/diskmanagers"
)

const (
	DatastoreProperty     = "datastore"
	DatastoreInfoProperty = "info"
	DatastoreNameProperty = "name"
	Folder                = "Folder"
	VirtualMachine        = "VirtualMachine"
	DummyDiskName         = "kube-dummyDisk.vmdk"
	ProviderPrefix        = "vsphere://"
	vSphereConfFileEnvVar = "VSPHERE_CONF_FILE"
	UUIDPrefix            = "VMware-"
)

// GetVSphere reads vSphere configuration from system environment and construct vSphere object
func GetVSphere() (*VSphere, error) {
	cfg, err := getVSphereConfig()
	if err != nil {
		return nil, err
	}
	vs, err := newControllerNode(*cfg)
	if err != nil {
		return nil, err
	}
	return vs, nil
}

func getVSphereConfig() (*VSphereConfig, error) {
	confFileLocation := os.Getenv(vSphereConfFileEnvVar)
	if confFileLocation == "" {
		return nil, fmt.Errorf("Env variable 'VSPHERE_CONF_FILE' is not set.")
	}
	confFile, err := os.Open(confFileLocation)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := confFile.Close(); err != nil {
			klog.Errorf("failed to close config file: %v", err)
		}
	}()

	cfg, err := readConfig(confFile)
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

// Returns the accessible datastores for the given node VM.
func getAccessibleDatastores(ctx context.Context, nodeVmDetail *NodeDetails, nodeManager *NodeManager) ([]*vclib.DatastoreInfo, error) {
	accessibleDatastores, err := nodeVmDetail.vm.GetAllAccessibleDatastores(ctx)
	if err != nil {
		// Check if the node VM is not found which indicates that the node info in the node manager is stale.
		// If so, rediscover the node and retry.
		if vclib.IsManagedObjectNotFoundError(err) {
			klog.V(4).Infof("error %q ManagedObjectNotFound for node %q. Rediscovering...", err, nodeVmDetail.NodeName)
			err = nodeManager.RediscoverNode(convertToK8sType(nodeVmDetail.NodeName))
			if err == nil {
				klog.V(4).Infof("Discovered node %s successfully", nodeVmDetail.NodeName)
				nodeInfo, err := nodeManager.GetNodeInfo(convertToK8sType(nodeVmDetail.NodeName))
				if err != nil {
					klog.V(4).Infof("error %q getting node info for node %+v", err, nodeVmDetail)
					return nil, err
				}

				accessibleDatastores, err = nodeInfo.vm.GetAllAccessibleDatastores(ctx)
				if err != nil {
					klog.V(4).Infof("error %q getting accessible datastores for node %+v", err, nodeVmDetail)
					return nil, err
				}
			} else {
				klog.V(4).Infof("error %q rediscovering node %+v", err, nodeVmDetail)
				return nil, err
			}
		} else {
			klog.V(4).Infof("error %q getting accessible datastores for node %+v", err, nodeVmDetail)
			return nil, err
		}
	}
	return accessibleDatastores, nil
}

// Get all datastores accessible for the virtual machine object.
func getSharedDatastoresInK8SCluster(ctx context.Context, nodeManager *NodeManager) ([]*vclib.DatastoreInfo, error) {
	nodeVmDetails, err := nodeManager.GetNodeDetails()
	if err != nil {
		klog.Errorf("Error while obtaining Kubernetes node nodeVmDetail details. error : %+v", err)
		return nil, err
	}

	if len(nodeVmDetails) == 0 {
		msg := fmt.Sprintf("Kubernetes node nodeVmDetail details is empty. nodeVmDetails : %+v", nodeVmDetails)
		klog.Error(msg)
		return nil, fmt.Errorf(msg)
	}
	var sharedDatastores []*vclib.DatastoreInfo
	for _, nodeVmDetail := range nodeVmDetails {
		klog.V(9).Infof("Getting accessible datastores for node %s", nodeVmDetail.NodeName)
		accessibleDatastores, err := getAccessibleDatastores(ctx, &nodeVmDetail, nodeManager)
		if err != nil {
			if err == vclib.ErrNoVMFound {
				klog.V(9).Infof("Got NoVMFound error for node %s", nodeVmDetail.NodeName)
				continue
			}
			return nil, err
		}

		if len(sharedDatastores) == 0 {
			sharedDatastores = accessibleDatastores
		} else {
			sharedDatastores = intersect(sharedDatastores, accessibleDatastores)
			if len(sharedDatastores) == 0 {
				return nil, fmt.Errorf("No shared datastores found in the Kubernetes cluster for nodeVmDetails: %+v", nodeVmDetails)
			}
		}
	}
	klog.V(9).Infof("sharedDatastores : %+v", sharedDatastores)
	return sharedDatastores, nil
}

func intersect(list1 []*vclib.DatastoreInfo, list2 []*vclib.DatastoreInfo) []*vclib.DatastoreInfo {
	klog.V(9).Infof("list1: %+v", list1)
	klog.V(9).Infof("list2: %+v", list2)
	var sharedDs []*vclib.DatastoreInfo
	for _, val1 := range list1 {
		// Check if val1 is found in list2
		for _, val2 := range list2 {
			// Intersection is performed based on the datastoreUrl as this uniquely identifies the datastore.
			if val1.Info.Url == val2.Info.Url {
				sharedDs = append(sharedDs, val1)
				break
			}
		}
	}
	return sharedDs
}

// getMostFreeDatastore gets the best fit compatible datastore by free space.
func getMostFreeDatastore(ctx context.Context, client *vim25.Client, dsInfoList []*vclib.DatastoreInfo) (*vclib.DatastoreInfo, error) {
	var curMax int64
	curMax = -1
	var index int
	for i, dsInfo := range dsInfoList {
		dsFreeSpace := dsInfo.Info.GetDatastoreInfo().FreeSpace
		if dsFreeSpace > curMax {
			curMax = dsFreeSpace
			index = i
		}
	}
	return dsInfoList[index], nil
}

func getPbmCompatibleDatastore(ctx context.Context, vcClient *vim25.Client, storagePolicyName string, nodeManager *NodeManager) (*vclib.DatastoreInfo, error) {
	pbmClient, err := vclib.NewPbmClient(ctx, vcClient)
	if err != nil {
		return nil, err
	}
	storagePolicyID, err := pbmClient.ProfileIDByName(ctx, storagePolicyName)
	if err != nil {
		klog.Errorf("Failed to get Profile ID by name: %s. err: %+v", storagePolicyName, err)
		return nil, err
	}
	sharedDs, err := getSharedDatastoresInK8SCluster(ctx, nodeManager)
	if err != nil {
		klog.Errorf("Failed to get shared datastores. err: %+v", err)
		return nil, err
	}
	if len(sharedDs) == 0 {
		msg := "No shared datastores found in the endpoint virtual center"
		klog.Errorf(msg)
		return nil, errors.New(msg)
	}
	compatibleDatastores, _, err := pbmClient.GetCompatibleDatastores(ctx, storagePolicyID, sharedDs)
	if err != nil {
		klog.Errorf("Failed to get compatible datastores from datastores : %+v with storagePolicy: %s. err: %+v",
			sharedDs, storagePolicyID, err)
		return nil, err
	}
	klog.V(9).Infof("compatibleDatastores : %+v", compatibleDatastores)
	datastore, err := getMostFreeDatastore(ctx, vcClient, compatibleDatastores)
	if err != nil {
		klog.Errorf("Failed to get most free datastore from compatible datastores: %+v. err: %+v", compatibleDatastores, err)
		return nil, err
	}
	klog.V(4).Infof("Most free datastore : %+s", datastore.Info.Name)
	return datastore, err
}

func getDatastoresForZone(ctx context.Context, nodeManager *NodeManager, selectedZones []string) ([]*vclib.DatastoreInfo, error) {

	var sharedDatastores []*vclib.DatastoreInfo

	for _, zone := range selectedZones {
		var sharedDatastoresPerZone []*vclib.DatastoreInfo
		hosts, err := nodeManager.GetHostsInZone(ctx, zone)
		if err != nil {
			return nil, err
		}
		klog.V(4).Infof("Hosts in zone %s : %s", zone, hosts)

		for _, host := range hosts {
			var hostSystemMo mo.HostSystem
			err = host.Properties(ctx, host.Reference(), []string{"datastore"}, &hostSystemMo)
			if err != nil {
				klog.Errorf("Failed to get datastore property for host %s. err : %+v", host, err)
				return nil, err
			}

			klog.V(4).Infof("Datastores mounted on host %s : %s", host, hostSystemMo.Datastore)
			var dsRefList []types.ManagedObjectReference
			for _, dsRef := range hostSystemMo.Datastore {
				dsRefList = append(dsRefList, dsRef)
			}

			var dsMoList []mo.Datastore
			pc := property.DefaultCollector(host.Client())
			properties := []string{DatastoreInfoProperty, DatastoreNameProperty}
			err = pc.Retrieve(ctx, dsRefList, properties, &dsMoList)
			if err != nil {
				klog.Errorf("Failed to get Datastore managed objects from datastore objects."+
					" dsObjList: %+v, properties: %+v, err: %+v", dsRefList, properties, err)
				return nil, err
			}
			klog.V(9).Infof("Datastore mo details: %+v", dsMoList)

			// find the Datacenter parent for this host
			mes, err := mo.Ancestors(ctx, host.Client(), pc.Reference(), host.Reference())
			if err != nil {
				return nil, err
			}
			var dcMoref *types.ManagedObjectReference
			for i := len(mes) - 1; i > 0; i-- {
				if mes[i].Self.Type == "Datacenter" {
					dcMoref = &mes[i].Self
					break
				}
			}
			if dcMoref == nil {
				return nil, fmt.Errorf("failed to find the Datacenter of host %s", host)
			}

			dc := object.NewDatacenter(host.Client(), *dcMoref)
			finder := find.NewFinder(host.Client(), false)
			finder.SetDatacenter(dc)
			var dsObjList []*vclib.DatastoreInfo
			for _, dsMo := range dsMoList {
				// use the finder so that InventoryPath is set correctly in dsObj
				dsObj, err := finder.Datastore(ctx, dsMo.Name)
				if err != nil {
					klog.Errorf("Failed to find datastore named %s in datacenter %s", dsMo.Name, dc)
					return nil, err
				}
				dsObjList = append(dsObjList,
					&vclib.DatastoreInfo{
						Datastore: &vclib.Datastore{Datastore: dsObj,
							Datacenter: &vclib.Datacenter{Datacenter: dc}},
						Info: dsMo.Info.GetDatastoreInfo()})
			}

			klog.V(9).Infof("DatastoreInfo details : %s", dsObjList)

			if len(sharedDatastoresPerZone) == 0 {
				sharedDatastoresPerZone = dsObjList
			} else {
				sharedDatastoresPerZone = intersect(sharedDatastoresPerZone, dsObjList)
				if len(sharedDatastoresPerZone) == 0 {
					klog.V(4).Infof("No shared datastores found among hosts %s", hosts)
					return nil, fmt.Errorf("No matching datastores found in the kubernetes cluster for zone %s", zone)
				}
			}
			klog.V(9).Infof("Shared datastore list after processing host %s : %s", host, sharedDatastoresPerZone)
		}
		klog.V(4).Infof("Shared datastore per zone %s is %s", zone, sharedDatastoresPerZone)
		if len(sharedDatastores) == 0 {
			sharedDatastores = sharedDatastoresPerZone
		} else {
			sharedDatastores = intersect(sharedDatastores, sharedDatastoresPerZone)
			if len(sharedDatastores) == 0 {
				return nil, fmt.Errorf("No matching datastores found in the kubernetes cluster across zones %s", selectedZones)
			}
		}
	}
	klog.V(1).Infof("Returning selected datastores : %s", sharedDatastores)
	return sharedDatastores, nil
}

func getPbmCompatibleZonedDatastore(ctx context.Context, vcClient *vim25.Client, storagePolicyName string, zonedDatastores []*vclib.DatastoreInfo) (*vclib.DatastoreInfo, error) {
	pbmClient, err := vclib.NewPbmClient(ctx, vcClient)
	if err != nil {
		return nil, err
	}
	storagePolicyID, err := pbmClient.ProfileIDByName(ctx, storagePolicyName)
	if err != nil {
		klog.Errorf("Failed to get Profile ID by name: %s. err: %+v", storagePolicyName, err)
		return nil, err
	}
	compatibleDatastores, _, err := pbmClient.GetCompatibleDatastores(ctx, storagePolicyID, zonedDatastores)
	if err != nil {
		klog.Errorf("Failed to get compatible datastores from datastores : %+v with storagePolicy: %s. err: %+v",
			zonedDatastores, storagePolicyID, err)
		return nil, err
	}
	klog.V(9).Infof("compatibleDatastores : %+v", compatibleDatastores)
	datastore, err := getMostFreeDatastore(ctx, vcClient, compatibleDatastores)
	if err != nil {
		klog.Errorf("Failed to get most free datastore from compatible datastores: %+v. err: %+v", compatibleDatastores, err)
		return nil, err
	}
	klog.V(4).Infof("Most free datastore : %+s", datastore.Info.Name)
	return datastore, err
}

func (vs *VSphere) setVMOptions(ctx context.Context, connection *vclib.VSphereConnection, ds *vclib.Datastore) (*vclib.VMOptions, error) {
	var vmOptions vclib.VMOptions
	dsHosts, err := ds.GetDatastoreHostMounts(ctx)
	if err != nil {
		klog.Errorf("Failed to get datastore host mounts for %v: %+v", ds, err)
		return nil, err
	}
	// pick a host at random to use for Volume creation
	dsHostMoref := dsHosts[rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(dsHosts))]
	dummyVMHost := object.NewHostSystem(connection.Client, dsHostMoref)
	resourcePool, err := dummyVMHost.ResourcePool(ctx)
	if err != nil {
		klog.Errorf("Failed to get resource pool from host %v", dummyVMHost)
		return nil, err
	}
	folder, err := ds.Datacenter.GetFolderByPath(ctx, vs.cfg.Workspace.Folder)
	if err != nil {
		return nil, err
	}
	vmOptions.VMFolder = folder
	vmOptions.VMResourcePool = resourcePool
	return &vmOptions, nil
}

// A background routine which will be responsible for deleting stale dummy VM's.
func (vs *VSphere) cleanUpDummyVMs(dummyVMPrefix string) {
	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for {
		time.Sleep(CleanUpDummyVMRoutineInterval * time.Minute)
		datacenters, err := vs.GetWorkspaceDatacenters(ctx)
		if err != nil {
			klog.V(4).Infof("Failed to get datacenters from VC. err: %+v", err)
			continue
		}
		// Clean up dummy VMs in each datacenter
		for _, dc := range datacenters {
			// Get the folder reference for global working directory where the dummy VM needs to be created.
			vmFolder, err := dc.GetFolderByPath(ctx, vs.cfg.Workspace.Folder)
			if err != nil {
				klog.V(4).Infof("Unable to get the kubernetes folder: %q reference. err: %+v", vs.cfg.Workspace.Folder, err)
				continue
			}
			// A write lock is acquired to make sure the cleanUp routine doesn't delete any VM's created by ongoing PVC requests.
			cleanUpDummyVMs := func() {
				cleanUpDummyVMLock.Lock()
				defer cleanUpDummyVMLock.Unlock()
				err = diskmanagers.CleanUpDummyVMs(ctx, vmFolder)
				if err != nil {
					klog.V(4).Infof("Unable to clean up dummy VM's in the kubernetes cluster: %q. err: %+v", vs.cfg.Workspace.Folder, err)
				}
			}
			cleanUpDummyVMs()
		}
	}
}

// Get canonical volume path for volume Path.
// Example1: The canonical path for volume path - [vsanDatastore] kubevols/volume.vmdk will be [vsanDatastore] 25d8b159-948c-4b73-e499-02001ad1b044/volume.vmdk
// Example2: The canonical path for volume path - [vsanDatastore] 25d8b159-948c-4b73-e499-02001ad1b044/volume.vmdk will be same as volume Path.
func getcanonicalVolumePath(ctx context.Context, dc *vclib.Datacenter, volumePath string) (string, error) {
	var folderID string
	var folderExists bool
	canonicalVolumePath := volumePath
	dsPathObj, err := vclib.GetDatastorePathObjFromVMDiskPath(volumePath)
	if err != nil {
		return "", err
	}
	dsPath := strings.Split(strings.TrimSpace(dsPathObj.Path), "/")
	if len(dsPath) <= 1 {
		return canonicalVolumePath, nil
	}
	datastore := dsPathObj.Datastore
	dsFolder := dsPath[0]
	folderNameIDMap, datastoreExists := datastoreFolderIDMap[datastore]
	if datastoreExists {
		folderID, folderExists = folderNameIDMap[dsFolder]
	}
	// Get the datastore folder ID if datastore or folder doesn't exist in datastoreFolderIDMap
	if !datastoreExists || !folderExists {
		if !vclib.IsValidUUID(dsFolder) {
			dummyDiskVolPath := "[" + datastore + "] " + dsFolder + "/" + DummyDiskName
			// Querying a non-existent dummy disk on the datastore folder.
			// It would fail and return an folder ID in the error message.
			_, err := dc.GetVirtualDiskPage83Data(ctx, dummyDiskVolPath)
			canonicalVolumePath, err = getPathFromFileNotFound(err)
			if err != nil {
				return "", fmt.Errorf("failed to get path from dummy request: %v", err)
			}
		}
		diskPath := vclib.GetPathFromVMDiskPath(canonicalVolumePath)
		if diskPath == "" {
			return "", fmt.Errorf("Failed to parse canonicalVolumePath: %s in getcanonicalVolumePath method", canonicalVolumePath)
		}
		folderID = strings.Split(strings.TrimSpace(diskPath), "/")[0]
		setdatastoreFolderIDMap(datastoreFolderIDMap, datastore, dsFolder, folderID)
	}
	canonicalVolumePath = strings.Replace(volumePath, dsFolder, folderID, 1)
	return canonicalVolumePath, nil
}

// getPathFromFileNotFound returns the path from a fileNotFound error
func getPathFromFileNotFound(err error) (string, error) {
	if soap.IsSoapFault(err) {
		fault := soap.ToSoapFault(err)
		f, ok := fault.VimFault().(types.FileNotFound)
		if !ok {
			return "", fmt.Errorf("%v is not a FileNotFound error", err)
		}
		return f.File, nil
	}
	return "", fmt.Errorf("%v is not a soap fault", err)
}

func setdatastoreFolderIDMap(
	datastoreFolderIDMap map[string]map[string]string,
	datastore string,
	folderName string,
	folderID string) {
	folderNameIDMap := datastoreFolderIDMap[datastore]
	if folderNameIDMap == nil {
		folderNameIDMap = make(map[string]string)
		datastoreFolderIDMap[datastore] = folderNameIDMap
	}
	folderNameIDMap[folderName] = folderID
}

func convertVolPathToDevicePath(ctx context.Context, dc *vclib.Datacenter, volPath string) (string, error) {
	volPath = vclib.RemoveStorageClusterORFolderNameFromVDiskPath(volPath)
	// Get the canonical volume path for volPath.
	canonicalVolumePath, err := getcanonicalVolumePath(ctx, dc, volPath)
	if err != nil {
		klog.Errorf("Failed to get canonical vsphere volume path for volume: %s. err: %+v", volPath, err)
		return "", err
	}
	// Check if the volume path contains .vmdk extension. If not, add the extension and update the nodeVolumes Map
	if len(canonicalVolumePath) > 0 && filepath.Ext(canonicalVolumePath) != ".vmdk" {
		canonicalVolumePath += ".vmdk"
	}
	return canonicalVolumePath, nil
}

// convertVolPathsToDevicePaths removes cluster or folder path from volPaths and convert to canonicalPath
func (vs *VSphere) convertVolPathsToDevicePaths(ctx context.Context, nodeVolumes map[k8stypes.NodeName][]string) (map[k8stypes.NodeName][]string, error) {
	vmVolumes := make(map[k8stypes.NodeName][]string)
	for nodeName, volPaths := range nodeVolumes {
		nodeInfo, err := vs.nodeManager.GetNodeInfo(nodeName)
		if err != nil {
			return nil, err
		}

		_, err = vs.getVSphereInstanceForServer(nodeInfo.vcServer, ctx)
		if err != nil {
			return nil, err
		}

		for i, volPath := range volPaths {
			deviceVolPath, err := convertVolPathToDevicePath(ctx, nodeInfo.dataCenter, volPath)
			if err != nil {
				klog.Errorf("Failed to convert vsphere volume path %s to device path for volume %s. err: %+v", volPath, deviceVolPath, err)
				return nil, err
			}
			volPaths[i] = deviceVolPath
		}
		vmVolumes[nodeName] = volPaths
	}
	return vmVolumes, nil
}

// checkDiskAttached verifies volumes are attached to the VMs which are in same vCenter and Datacenter
// Returns nodes if exist any for which VM is not found in that vCenter and Datacenter
func (vs *VSphere) checkDiskAttached(ctx context.Context, nodes []k8stypes.NodeName, nodeVolumes map[k8stypes.NodeName][]string, attached map[string]map[string]bool, retry bool) ([]k8stypes.NodeName, error) {
	var nodesToRetry []k8stypes.NodeName
	var vmList []*vclib.VirtualMachine
	var nodeInfo NodeInfo
	var err error

	for _, nodeName := range nodes {
		nodeInfo, err = vs.nodeManager.GetNodeInfo(nodeName)
		if err != nil {
			return nodesToRetry, err
		}
		vmList = append(vmList, nodeInfo.vm)
	}

	// Making sure session is valid
	_, err = vs.getVSphereInstanceForServer(nodeInfo.vcServer, ctx)
	if err != nil {
		return nodesToRetry, err
	}

	// If any of the nodes are not present property collector query will fail for entire operation
	vmMoList, err := nodeInfo.dataCenter.GetVMMoList(ctx, vmList, []string{"config.hardware.device", "name", "config.uuid"})
	if err != nil {
		if vclib.IsManagedObjectNotFoundError(err) && !retry {
			klog.V(4).Infof("checkDiskAttached: ManagedObjectNotFound for property collector query for nodes: %+v vms: %+v", nodes, vmList)
			// Property Collector Query failed
			// VerifyVolumePaths per VM
			for _, nodeName := range nodes {
				nodeInfo, err := vs.nodeManager.GetNodeInfo(nodeName)
				if err != nil {
					return nodesToRetry, err
				}
				devices, err := nodeInfo.vm.VirtualMachine.Device(ctx)
				if err != nil {
					if vclib.IsManagedObjectNotFoundError(err) {
						klog.V(4).Infof("checkDiskAttached: ManagedObjectNotFound for Kubernetes node: %s with vSphere Virtual Machine reference: %v", nodeName, nodeInfo.vm)
						nodesToRetry = append(nodesToRetry, nodeName)
						continue
					}
					return nodesToRetry, err
				}
				klog.V(4).Infof("Verifying Volume Paths by devices for node %s and VM %s", nodeName, nodeInfo.vm)
				vclib.VerifyVolumePathsForVMDevices(devices, nodeVolumes[nodeName], convertToString(nodeName), attached)
			}
		}
		return nodesToRetry, err
	}

	vmMoMap := make(map[string]mo.VirtualMachine)
	for _, vmMo := range vmMoList {
		if vmMo.Config == nil {
			klog.Errorf("Config is not available for VM: %q", vmMo.Name)
			continue
		}
		klog.V(9).Infof("vmMoMap vmname: %q vmuuid: %s", vmMo.Name, strings.ToLower(vmMo.Config.Uuid))
		vmMoMap[strings.ToLower(vmMo.Config.Uuid)] = vmMo
	}

	klog.V(9).Infof("vmMoMap: +%v", vmMoMap)

	for _, nodeName := range nodes {
		node, err := vs.nodeManager.GetNode(nodeName)
		if err != nil {
			return nodesToRetry, err
		}
		nodeUUID, err := GetNodeUUID(&node)
		if err != nil {
			klog.Errorf("Node Discovery failed to get node uuid for node %s with error: %v", node.Name, err)
			return nodesToRetry, err
		}
		nodeUUID = strings.ToLower(nodeUUID)
		klog.V(9).Infof("Verifying volume for node %s with nodeuuid %q: %v", nodeName, nodeUUID, vmMoMap)
		vclib.VerifyVolumePathsForVM(vmMoMap[nodeUUID], nodeVolumes[nodeName], convertToString(nodeName), attached)
	}
	return nodesToRetry, nil
}

func (vs *VSphere) GetNodeNameFromProviderID(providerID string) (string, error) {
	var nodeName string
	nodes, err := vs.nodeManager.GetNodeDetails()
	if err != nil {
		klog.Errorf("Error while obtaining Kubernetes node nodeVmDetail details. error : %+v", err)
		return "", err
	}
	for _, node := range nodes {
		// ProviderID is UUID for nodes v1.9.3+
		if node.VMUUID == GetUUIDFromProviderID(providerID) || node.NodeName == providerID {
			nodeName = node.NodeName
			break
		}
	}
	if nodeName == "" {
		msg := fmt.Sprintf("Error while obtaining Kubernetes nodename for providerID %s.", providerID)
		return "", errors.New(msg)
	}
	return nodeName, nil
}

func GetUUIDFromProviderID(providerID string) string {
	return strings.TrimPrefix(providerID, ProviderPrefix)
}

func IsUUIDSupportedNode(node *v1.Node) (bool, error) {
	newVersion, err := version.ParseSemantic("v1.9.4")
	if err != nil {
		klog.Errorf("Failed to determine whether node %+v is old with error %v", node, err)
		return false, err
	}
	nodeVersion, err := version.ParseSemantic(node.Status.NodeInfo.KubeletVersion)
	if err != nil {
		klog.Errorf("Failed to determine whether node %+v is old with error %v", node, err)
		return false, err
	}
	if nodeVersion.LessThan(newVersion) {
		return true, nil
	}
	return false, nil
}

func GetNodeUUID(node *v1.Node) (string, error) {
	oldNode, err := IsUUIDSupportedNode(node)
	if err != nil {
		klog.Errorf("Failed to get node UUID for node %+v with error %v", node, err)
		return "", err
	}
	if oldNode {
		return node.Status.NodeInfo.SystemUUID, nil
	}
	return GetUUIDFromProviderID(node.Spec.ProviderID), nil
}

func GetVMUUID() (string, error) {
	uuidFromFile, err := getRawUUID()
	if err != nil {
		return "", fmt.Errorf("error retrieving vm uuid: %s", err)
	}
	//strip leading and trailing white space and new line char
	uuid := strings.TrimSpace(uuidFromFile)
	// check the uuid starts with "VMware-"
	if !strings.HasPrefix(uuid, UUIDPrefix) {
		return "", fmt.Errorf("failed to match Prefix, UUID read from the file is %v", uuidFromFile)
	}
	// Strip the prefix and white spaces and -
	uuid = strings.Replace(uuid[len(UUIDPrefix):(len(uuid))], " ", "", -1)
	uuid = strings.Replace(uuid, "-", "", -1)
	if len(uuid) != 32 {
		return "", fmt.Errorf("length check failed, UUID read from the file is %v", uuidFromFile)
	}
	// need to add dashes, e.g. "564d395e-d807-e18a-cb25-b79f65eb2b9f"
	uuid = fmt.Sprintf("%s-%s-%s-%s-%s", uuid[0:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32])
	return uuid, nil
}

// GetWorkspaceDatacenters returns the Datacenter objects that VCP has access to.
// User can configure the list of datacenters in vsphere.conf. Otherwise all the
// Datacenters in the configured list of VCs are returned.
func (vs *VSphere) GetWorkspaceDatacenters(ctx context.Context) ([]*vclib.Datacenter, error) {
	var datacenterObjs []*vclib.Datacenter
	for vc, vsi := range vs.vsphereInstanceMap {
		// ensure connection to VC
		err := vs.nodeManager.vcConnect(ctx, vsi)
		if err != nil {
			return nil, err
		}
		if vsi.cfg.Datacenters == "" {
			vcDatacenterObjs, err := vclib.GetAllDatacenter(ctx, vsi.conn)
			if err != nil {
				klog.Errorf("Error fetching list of datacenters from VC %s: %+v", vc, err)
				return nil, err
			}
			datacenterObjs = append(datacenterObjs, vcDatacenterObjs...)
		} else {
			datacenters := strings.Split(vsi.cfg.Datacenters, ",")
			for _, dc := range datacenters {
				dc = strings.TrimSpace(dc)
				if dc == "" {
					continue
				}
				datacenterObj, err := vclib.GetDatacenter(ctx, vsi.conn, dc)
				if err != nil {
					klog.Errorf("Error fetching datacenter %s from VC %s: %+v", dc, vc, err)
					return nil, err
				}
				datacenterObjs = append(datacenterObjs, datacenterObj)
			}
		}
	}
	return datacenterObjs, nil
}

// FindDatastoreByName looks for the given datastore by name across all available datacenters.
// If more than one Datacenter has a Datastore with the given name, then returns reference to all of them.
func (vs *VSphere) FindDatastoreByName(ctx context.Context, datastoreName string) ([]*vclib.DatastoreInfo, error) {
	datacenters, err := vs.GetWorkspaceDatacenters(ctx)
	if err != nil {
		return nil, err
	}
	var datastoreInfos []*vclib.DatastoreInfo
	for _, dc := range datacenters {
		datastoreInfo, err := dc.GetDatastoreInfoByName(ctx, datastoreName)
		if err != nil {
			klog.V(9).Infof("Did not find datastore %s in datacenter %s, still looking.", datastoreName, dc.Name())
			continue
		}
		datastoreInfos = append(datastoreInfos, datastoreInfo)
	}
	if len(datastoreInfos) == 0 {
		return nil, fmt.Errorf("Datastore '%s' not found", datastoreName)
	}
	klog.V(4).Infof("Found datastore infos %v for datastore %s", datastoreInfos, datastoreName)
	return datastoreInfos, nil
}
