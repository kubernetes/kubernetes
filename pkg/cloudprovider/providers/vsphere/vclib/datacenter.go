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
	"errors"
	"fmt"
	"strings"

	"github.com/golang/glog"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

// Datacenter extends the govmomi Datacenter object
type Datacenter struct {
	*object.Datacenter
}

// GetDatacenter returns the DataCenter Object for the given datacenterPath
// If datacenter is located in a folder, include full path to datacenter else just provide the datacenter name
func GetDatacenter(ctx context.Context, connection *VSphereConnection, datacenterPath string) (*Datacenter, error) {
	finder := find.NewFinder(connection.GoVmomiClient.Client, true)
	datacenter, err := finder.Datacenter(ctx, datacenterPath)
	if err != nil {
		glog.Errorf("Failed to find the datacenter: %s. err: %+v", datacenterPath, err)
		return nil, err
	}
	dc := Datacenter{datacenter}
	return &dc, nil
}

// GetVMByUUID gets the VM object from the given vmUUID
func (dc *Datacenter) GetVMByUUID(ctx context.Context, vmUUID string) (*VirtualMachine, error) {
	s := object.NewSearchIndex(dc.Client())
	vmUUID = strings.ToLower(strings.TrimSpace(vmUUID))
	svm, err := s.FindByUuid(ctx, dc.Datacenter, vmUUID, true, nil)
	if err != nil {
		glog.Errorf("Failed to find VM by UUID. VM UUID: %s, err: %+v", vmUUID, err)
		return nil, err
	}
	if svm == nil {
		glog.Errorf("Unable to find VM by UUID. VM UUID: %s", vmUUID)
		return nil, fmt.Errorf("Failed to find VM by UUID: %s", vmUUID)
	}
	virtualMachine := VirtualMachine{object.NewVirtualMachine(dc.Client(), svm.Reference()), dc}
	return &virtualMachine, nil
}

// GetVMByPath gets the VM object from the given vmPath
// vmPath should be the full path to VM and not just the name
func (dc *Datacenter) GetVMByPath(ctx context.Context, vmPath string) (*VirtualMachine, error) {
	finder := getFinder(dc)
	vm, err := finder.VirtualMachine(ctx, vmPath)
	if err != nil {
		glog.Errorf("Failed to find VM by Path. VM Path: %s, err: %+v", vmPath, err)
		return nil, err
	}
	virtualMachine := VirtualMachine{vm, dc}
	return &virtualMachine, nil
}

// GetDatastoreByPath gets the Datastore object from the given vmDiskPath
func (dc *Datacenter) GetDatastoreByPath(ctx context.Context, vmDiskPath string) (*Datastore, error) {
	datastorePathObj := new(object.DatastorePath)
	isSuccess := datastorePathObj.FromString(vmDiskPath)
	if !isSuccess {
		glog.Errorf("Failed to parse vmDiskPath: %s", vmDiskPath)
		return nil, errors.New("Failed to parse vmDiskPath")
	}
	finder := getFinder(dc)
	ds, err := finder.Datastore(ctx, datastorePathObj.Datastore)
	if err != nil {
		glog.Errorf("Failed while searching for datastore: %s. err: %+v", datastorePathObj.Datastore, err)
		return nil, err
	}
	datastore := Datastore{ds, dc}
	return &datastore, nil
}

// GetDatastoreByName gets the Datastore object for the given datastore name
func (dc *Datacenter) GetDatastoreByName(ctx context.Context, name string) (*Datastore, error) {
	finder := getFinder(dc)
	ds, err := finder.Datastore(ctx, name)
	if err != nil {
		glog.Errorf("Failed while searching for datastore: %s. err: %+v", name, err)
		return nil, err
	}
	datastore := Datastore{ds, dc}
	return &datastore, nil
}

// GetFolderByPath gets the Folder Object from the given folder path
// folderPath should be the full path to folder
func (dc *Datacenter) GetFolderByPath(ctx context.Context, folderPath string) (*Folder, error) {
	finder := getFinder(dc)
	vmFolder, err := finder.Folder(ctx, folderPath)
	if err != nil {
		glog.Errorf("Failed to get the folder reference for %s. err: %+v", folderPath, err)
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
		glog.Errorf("VirtualMachine Object list is empty")
		return nil, fmt.Errorf("VirtualMachine Object list is empty")
	}

	for _, vmObj := range vmObjList {
		vmRefs = append(vmRefs, vmObj.Reference())
	}
	pc := property.DefaultCollector(dc.Client())
	err := pc.Retrieve(ctx, vmRefs, properties, &vmMoList)
	if err != nil {
		glog.Errorf("Failed to get VM managed objects from VM objects. vmObjList: %+v, properties: %+v, err: %v", vmObjList, properties, err)
		return nil, err
	}
	return vmMoList, nil
}

// GetDatastoreMoList gets the Datastore Managed Objects with the given properties from the datastore objects
func (dc *Datacenter) GetDatastoreMoList(ctx context.Context, dsObjList []*Datastore, properties []string) ([]mo.Datastore, error) {
	var dsMoList []mo.Datastore
	var dsRefs []types.ManagedObjectReference
	if len(dsObjList) < 1 {
		glog.Errorf("Datastore Object list is empty")
		return nil, fmt.Errorf("Datastore Object list is empty")
	}

	for _, dsObj := range dsObjList {
		dsRefs = append(dsRefs, dsObj.Reference())
	}
	pc := property.DefaultCollector(dc.Client())
	err := pc.Retrieve(ctx, dsRefs, properties, &dsMoList)
	if err != nil {
		glog.Errorf("Failed to get Datastore managed objects from datastore objects. dsObjList: %+v, properties: %+v, err: %v", dsObjList, properties, err)
		return nil, err
	}
	return dsMoList, nil
}
