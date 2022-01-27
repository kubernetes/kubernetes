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

package vclib

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"k8s.io/klog/v2"
)

// Datacenter extends the govmomi Datacenter object
type Datacenter struct {
	*object.Datacenter
}

// GetDatacenter returns the DataCenter Object for the given datacenterPath
// If datacenter is located in a folder, include full path to datacenter else just provide the datacenter name
func GetDatacenter(ctx context.Context, connection *VSphereConnection, datacenterPath string) (*Datacenter, error) {
	finder := find.NewFinder(connection.Client, false)
	datacenter, err := finder.Datacenter(ctx, datacenterPath)
	if err != nil {
		klog.Errorf("Failed to find the datacenter: %s. err: %+v", datacenterPath, err)
		return nil, err
	}
	dc := Datacenter{datacenter}
	return &dc, nil
}

// GetAllDatacenter returns all the DataCenter Objects
func GetAllDatacenter(ctx context.Context, connection *VSphereConnection) ([]*Datacenter, error) {
	var dc []*Datacenter
	finder := find.NewFinder(connection.Client, false)
	datacenters, err := finder.DatacenterList(ctx, "*")
	if err != nil {
		klog.Errorf("Failed to find the datacenter. err: %+v", err)
		return nil, err
	}
	for _, datacenter := range datacenters {
		dc = append(dc, &(Datacenter{datacenter}))
	}

	return dc, nil
}

// GetVMByUUID gets the VM object from the given vmUUID
func (dc *Datacenter) GetVMByUUID(ctx context.Context, vmUUID string) (*VirtualMachine, error) {
	s := object.NewSearchIndex(dc.Client())
	vmUUID = strings.ToLower(strings.TrimSpace(vmUUID))
	svm, err := s.FindByUuid(ctx, dc.Datacenter, vmUUID, true, nil)
	if err != nil {
		klog.Errorf("Failed to find VM by UUID. VM UUID: %s, err: %+v", vmUUID, err)
		return nil, err
	}
	if svm == nil {
		klog.Errorf("Unable to find VM by UUID. VM UUID: %s", vmUUID)
		return nil, ErrNoVMFound
	}
	virtualMachine := VirtualMachine{object.NewVirtualMachine(dc.Client(), svm.Reference()), dc}
	return &virtualMachine, nil
}

// GetHostByVMUUID gets the host object from the given vmUUID
func (dc *Datacenter) GetHostByVMUUID(ctx context.Context, vmUUID string) (*types.ManagedObjectReference, error) {
	virtualMachine, err := dc.GetVMByUUID(ctx, vmUUID)
	if err != nil {
		return nil, err
	}
	var vmMo mo.VirtualMachine
	pc := property.DefaultCollector(virtualMachine.Client())
	err = pc.RetrieveOne(ctx, virtualMachine.Reference(), []string{"summary.runtime.host"}, &vmMo)
	if err != nil {
		klog.Errorf("Failed to retrieve VM runtime host, err: %v", err)
		return nil, err
	}
	host := vmMo.Summary.Runtime.Host
	klog.Infof("%s host is %s", virtualMachine.Reference(), host)
	return host, nil
}

// GetVMByPath gets the VM object from the given vmPath
// vmPath should be the full path to VM and not just the name
func (dc *Datacenter) GetVMByPath(ctx context.Context, vmPath string) (*VirtualMachine, error) {
	finder := getFinder(dc)
	vm, err := finder.VirtualMachine(ctx, vmPath)
	if err != nil {
		klog.Errorf("Failed to find VM by Path. VM Path: %s, err: %+v", vmPath, err)
		return nil, err
	}
	virtualMachine := VirtualMachine{vm, dc}
	return &virtualMachine, nil
}

// GetAllDatastores gets the datastore URL to DatastoreInfo map for all the datastores in
// the datacenter.
func (dc *Datacenter) GetAllDatastores(ctx context.Context) (map[string]*DatastoreInfo, error) {
	finder := getFinder(dc)
	datastores, err := finder.DatastoreList(ctx, "*")
	if err != nil {
		klog.Errorf("Failed to get all the datastores. err: %+v", err)
		return nil, err
	}
	var dsList []types.ManagedObjectReference
	for _, ds := range datastores {
		dsList = append(dsList, ds.Reference())
	}

	var dsMoList []mo.Datastore
	pc := property.DefaultCollector(dc.Client())
	properties := []string{DatastoreInfoProperty}
	err = pc.Retrieve(ctx, dsList, properties, &dsMoList)
	if err != nil {
		klog.Errorf("Failed to get Datastore managed objects from datastore objects."+
			" dsObjList: %+v, properties: %+v, err: %v", dsList, properties, err)
		return nil, err
	}

	dsURLInfoMap := make(map[string]*DatastoreInfo)
	for _, dsMo := range dsMoList {
		dsURLInfoMap[dsMo.Info.GetDatastoreInfo().Url] = &DatastoreInfo{
			&Datastore{object.NewDatastore(dc.Client(), dsMo.Reference()),
				dc},
			dsMo.Info.GetDatastoreInfo()}
	}
	klog.V(9).Infof("dsURLInfoMap : %+v", dsURLInfoMap)
	return dsURLInfoMap, nil
}

// GetAllHosts returns all the host objects in this datacenter of VC
func (dc *Datacenter) GetAllHosts(ctx context.Context) ([]types.ManagedObjectReference, error) {
	finder := getFinder(dc)
	hostSystems, err := finder.HostSystemList(ctx, "*")
	if err != nil {
		klog.Errorf("Failed to get all hostSystems. err: %+v", err)
		return nil, err
	}
	var hostMors []types.ManagedObjectReference
	for _, hs := range hostSystems {
		hostMors = append(hostMors, hs.Reference())
	}
	return hostMors, nil
}

// GetDatastoreByPath gets the Datastore object from the given vmDiskPath
func (dc *Datacenter) GetDatastoreByPath(ctx context.Context, vmDiskPath string) (*Datastore, error) {
	datastorePathObj := new(object.DatastorePath)
	isSuccess := datastorePathObj.FromString(vmDiskPath)
	if !isSuccess {
		klog.Errorf("Failed to parse vmDiskPath: %s", vmDiskPath)
		return nil, errors.New("Failed to parse vmDiskPath")
	}

	return dc.GetDatastoreByName(ctx, datastorePathObj.Datastore)
}

// GetDatastoreByName gets the Datastore object for the given datastore name
func (dc *Datacenter) GetDatastoreByName(ctx context.Context, name string) (*Datastore, error) {
	finder := getFinder(dc)
	ds, err := finder.Datastore(ctx, name)
	if err != nil {
		klog.Errorf("Failed while searching for datastore: %s. err: %+v", name, err)
		return nil, err
	}
	datastore := Datastore{ds, dc}
	return &datastore, nil
}

// GetDatastoreInfoByName gets the Datastore object for the given datastore name
func (dc *Datacenter) GetDatastoreInfoByName(ctx context.Context, name string) (*DatastoreInfo, error) {
	finder := getFinder(dc)
	ds, err := finder.Datastore(ctx, name)
	if err != nil {
		klog.Errorf("Failed while searching for datastore: %s. err: %+v", name, err)
		return nil, err
	}
	datastore := Datastore{ds, dc}
	var dsMo mo.Datastore
	pc := property.DefaultCollector(dc.Client())
	properties := []string{DatastoreInfoProperty}
	err = pc.RetrieveOne(ctx, ds.Reference(), properties, &dsMo)
	if err != nil {
		klog.Errorf("Failed to get Datastore managed objects from datastore reference."+
			" dsRef: %+v, err: %+v", ds.Reference(), err)
		return nil, err
	}
	klog.V(9).Infof("Result dsMo: %+v", dsMo)
	return &DatastoreInfo{Datastore: &datastore, Info: dsMo.Info.GetDatastoreInfo()}, nil
}

// GetResourcePool gets the resource pool for the given path
func (dc *Datacenter) GetResourcePool(ctx context.Context, resourcePoolPath string) (*object.ResourcePool, error) {
	finder := getFinder(dc)
	var resourcePool *object.ResourcePool
	var err error
	resourcePool, err = finder.ResourcePoolOrDefault(ctx, resourcePoolPath)
	if err != nil {
		klog.Errorf("Failed to get the ResourcePool for path '%s'. err: %+v", resourcePoolPath, err)
		return nil, err
	}
	return resourcePool, nil
}

// GetFolderByPath gets the Folder Object from the given folder path
// folderPath should be the full path to folder
func (dc *Datacenter) GetFolderByPath(ctx context.Context, folderPath string) (*Folder, error) {
	finder := getFinder(dc)
	vmFolder, err := finder.Folder(ctx, folderPath)
	if err != nil {
		klog.Errorf("Failed to get the folder reference for %s. err: %+v", folderPath, err)
		return nil, err
	}
	folder := Folder{vmFolder, dc}
	return &folder, nil
}

// GetVMMoList gets the VM Managed Objects with the given properties from the VM object
func (dc *Datacenter) GetVMMoList(ctx context.Context, vmObjList []*VirtualMachine, properties []string) ([]mo.VirtualMachine, error) {
	var vmMoList []mo.VirtualMachine
	var vmRefs []types.ManagedObjectReference
	if len(vmObjList) < 1 {
		klog.Errorf("VirtualMachine Object list is empty")
		return nil, fmt.Errorf("VirtualMachine Object list is empty")
	}

	for _, vmObj := range vmObjList {
		vmRefs = append(vmRefs, vmObj.Reference())
	}
	pc := property.DefaultCollector(dc.Client())
	err := pc.Retrieve(ctx, vmRefs, properties, &vmMoList)
	if err != nil {
		klog.Errorf("Failed to get VM managed objects from VM objects. vmObjList: %+v, properties: %+v, err: %v", vmObjList, properties, err)
		return nil, err
	}
	return vmMoList, nil
}

// GetVirtualDiskPage83Data gets the virtual disk UUID by diskPath
func (dc *Datacenter) GetVirtualDiskPage83Data(ctx context.Context, diskPath string) (string, error) {
	if len(diskPath) > 0 && filepath.Ext(diskPath) != ".vmdk" {
		diskPath += ".vmdk"
	}
	vdm := object.NewVirtualDiskManager(dc.Client())
	// Returns uuid of vmdk virtual disk
	diskUUID, err := vdm.QueryVirtualDiskUuid(ctx, diskPath, dc.Datacenter)

	if err != nil {
		klog.Warningf("QueryVirtualDiskUuid failed for diskPath: %q. err: %+v", diskPath, err)
		return "", err
	}
	diskUUID = formatVirtualDiskUUID(diskUUID)
	return diskUUID, nil
}

// GetDatastoreMoList gets the Datastore Managed Objects with the given properties from the datastore objects
func (dc *Datacenter) GetDatastoreMoList(ctx context.Context, dsObjList []*Datastore, properties []string) ([]mo.Datastore, error) {
	var dsMoList []mo.Datastore
	var dsRefs []types.ManagedObjectReference
	if len(dsObjList) < 1 {
		klog.Errorf("Datastore Object list is empty")
		return nil, fmt.Errorf("Datastore Object list is empty")
	}

	for _, dsObj := range dsObjList {
		dsRefs = append(dsRefs, dsObj.Reference())
	}
	pc := property.DefaultCollector(dc.Client())
	err := pc.Retrieve(ctx, dsRefs, properties, &dsMoList)
	if err != nil {
		klog.Errorf("Failed to get Datastore managed objects from datastore objects. dsObjList: %+v, properties: %+v, err: %v", dsObjList, properties, err)
		return nil, err
	}
	return dsMoList, nil
}

// CheckDisksAttached checks if the disk is attached to node.
// This is done by comparing the volume path with the backing.FilePath on the VM Virtual disk devices.
func (dc *Datacenter) CheckDisksAttached(ctx context.Context, nodeVolumes map[string][]string) (map[string]map[string]bool, error) {
	attached := make(map[string]map[string]bool)
	var vmList []*VirtualMachine
	for nodeName, volPaths := range nodeVolumes {
		for _, volPath := range volPaths {
			setNodeVolumeMap(attached, volPath, nodeName, false)
		}
		vm, err := dc.GetVMByPath(ctx, nodeName)
		if err != nil {
			if IsNotFound(err) {
				klog.Warningf("Node %q does not exist, vSphere CP will assume disks %v are not attached to it.", nodeName, volPaths)
			}
			continue
		}
		vmList = append(vmList, vm)
	}
	if len(vmList) == 0 {
		klog.V(2).Infof("vSphere CP will assume no disks are attached to any node.")
		return attached, nil
	}
	vmMoList, err := dc.GetVMMoList(ctx, vmList, []string{"config.hardware.device", "name"})
	if err != nil {
		// When there is an error fetching instance information
		// it is safer to return nil and let volume information not be touched.
		klog.Errorf("Failed to get VM Managed object for nodes: %+v. err: +%v", vmList, err)
		return nil, err
	}

	for _, vmMo := range vmMoList {
		if vmMo.Config == nil {
			klog.Errorf("Config is not available for VM: %q", vmMo.Name)
			continue
		}
		for nodeName, volPaths := range nodeVolumes {
			if nodeName == vmMo.Name {
				verifyVolumePathsForVM(vmMo, volPaths, attached)
			}
		}
	}
	return attached, nil
}

// VerifyVolumePathsForVM verifies if the volume paths (volPaths) are attached to VM.
func verifyVolumePathsForVM(vmMo mo.VirtualMachine, volPaths []string, nodeVolumeMap map[string]map[string]bool) {
	// Verify if the volume paths are present on the VM backing virtual disk devices
	for _, volPath := range volPaths {
		vmDevices := object.VirtualDeviceList(vmMo.Config.Hardware.Device)
		for _, device := range vmDevices {
			if vmDevices.TypeName(device) == "VirtualDisk" {
				virtualDevice := device.GetVirtualDevice()
				if backing, ok := virtualDevice.Backing.(*types.VirtualDiskFlatVer2BackingInfo); ok {
					if backing.FileName == volPath {
						setNodeVolumeMap(nodeVolumeMap, volPath, vmMo.Name, true)
					}
				}
			}
		}
	}
}

func setNodeVolumeMap(
	nodeVolumeMap map[string]map[string]bool,
	volumePath string,
	nodeName string,
	check bool) {
	volumeMap := nodeVolumeMap[nodeName]
	if volumeMap == nil {
		volumeMap = make(map[string]bool)
		nodeVolumeMap[nodeName] = volumeMap
	}
	volumeMap[volumePath] = check
}
