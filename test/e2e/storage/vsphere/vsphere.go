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
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	volDir                    = "kubevols"
	defaultDiskCapacityKB     = 2097152
	defaultDiskFormat         = "thin"
	defaultSCSIControllerType = "lsiLogic"
	virtualMachineType        = "VirtualMachine"
)

// VSphere represents a vSphere instance where one or more kubernetes nodes are running.
type VSphere struct {
	Config *Config
	Client *govmomi.Client
}

// VolumeOptions specifies various options for a volume.
type VolumeOptions struct {
	Name               string
	CapacityKB         int
	DiskFormat         string
	SCSIControllerType string
	Datastore          string
}

// GetDatacenter returns the DataCenter Object for the given datacenterPath
func (vs *VSphere) GetDatacenter(ctx context.Context, datacenterPath string) (*object.Datacenter, error) {
	Connect(ctx, vs)
	finder := find.NewFinder(vs.Client.Client, false)
	return finder.Datacenter(ctx, datacenterPath)
}

// GetDatacenterFromObjectReference returns the DataCenter Object for the given datacenter reference
func (vs *VSphere) GetDatacenterFromObjectReference(ctx context.Context, dc object.Reference) *object.Datacenter {
	Connect(ctx, vs)
	return object.NewDatacenter(vs.Client.Client, dc.Reference())
}

// GetAllDatacenter returns all the DataCenter Objects
func (vs *VSphere) GetAllDatacenter(ctx context.Context) ([]*object.Datacenter, error) {
	Connect(ctx, vs)
	finder := find.NewFinder(vs.Client.Client, false)
	return finder.DatacenterList(ctx, "*")
}

// GetVMByUUID returns the VM object Reference from the given vmUUID
func (vs *VSphere) GetVMByUUID(ctx context.Context, vmUUID string, dc object.Reference) (object.Reference, error) {
	Connect(ctx, vs)
	datacenter := vs.GetDatacenterFromObjectReference(ctx, dc)
	s := object.NewSearchIndex(vs.Client.Client)
	vmUUID = strings.ToLower(strings.TrimSpace(vmUUID))
	return s.FindByUuid(ctx, datacenter, vmUUID, true, nil)
}

// GetHostFromVMReference returns host object reference of the host on which the specified VM resides
func (vs *VSphere) GetHostFromVMReference(ctx context.Context, vm types.ManagedObjectReference) types.ManagedObjectReference {
	Connect(ctx, vs)
	var vmMo mo.VirtualMachine
	vs.Client.RetrieveOne(ctx, vm, []string{"summary.runtime.host"}, &vmMo)
	host := *vmMo.Summary.Runtime.Host
	return host
}

// GetDatastoresMountedOnHost returns the datastore references of all the datastores mounted on the specified host
func (vs *VSphere) GetDatastoresMountedOnHost(ctx context.Context, host types.ManagedObjectReference) []types.ManagedObjectReference {
	Connect(ctx, vs)
	var hostMo mo.HostSystem
	vs.Client.RetrieveOne(ctx, host, []string{"datastore"}, &hostMo)
	return hostMo.Datastore
}

// GetDatastoreRefFromName returns the datastore reference of the specified datastore
func (vs *VSphere) GetDatastoreRefFromName(ctx context.Context, dc object.Reference, datastoreName string) (types.ManagedObjectReference, error) {
	Connect(ctx, vs)
	datacenter := object.NewDatacenter(vs.Client.Client, dc.Reference())
	finder := find.NewFinder(vs.Client.Client, false)
	finder.SetDatacenter(datacenter)
	datastore, err := finder.Datastore(ctx, datastoreName)
	return datastore.Reference(), err
}

// GetFolderByPath gets the Folder Object Reference from the given folder path
// folderPath should be the full path to folder
func (vs *VSphere) GetFolderByPath(ctx context.Context, dc object.Reference, folderPath string) (vmFolderMor types.ManagedObjectReference, err error) {
	Connect(ctx, vs)
	datacenter := object.NewDatacenter(vs.Client.Client, dc.Reference())
	finder := find.NewFinder(datacenter.Client(), false)
	finder.SetDatacenter(datacenter)
	vmFolder, err := finder.Folder(ctx, folderPath)
	if err != nil {
		framework.Logf("Failed to get the folder reference for %s. err: %+v", folderPath, err)
		return vmFolderMor, err
	}
	return vmFolder.Reference(), nil
}

// CreateVolume creates a vsphere volume using given volume parameters specified in VolumeOptions.
// If volume is created successfully the canonical disk path is returned else error is returned.
func (vs *VSphere) CreateVolume(volumeOptions *VolumeOptions, dataCenterRef types.ManagedObjectReference) (string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	Connect(ctx, vs)
	datacenter := object.NewDatacenter(vs.Client.Client, dataCenterRef)
	var (
		err                     error
		directoryAlreadyPresent = false
	)
	if datacenter == nil {
		return "", fmt.Errorf("datacenter is nil")
	}
	vs.initVolumeOptions(volumeOptions)
	finder := find.NewFinder(datacenter.Client(), false)
	finder.SetDatacenter(datacenter)
	ds, err := finder.Datastore(ctx, volumeOptions.Datastore)
	if err != nil {
		return "", fmt.Errorf("Failed while searching for datastore: %s. err: %+v", volumeOptions.Datastore, err)
	}
	directoryPath := filepath.Clean(ds.Path(volDir)) + "/"
	fileManager := object.NewFileManager(ds.Client())
	err = fileManager.MakeDirectory(ctx, directoryPath, datacenter, false)
	if err != nil {
		if soap.IsSoapFault(err) {
			soapFault := soap.ToSoapFault(err)
			if _, ok := soapFault.VimFault().(types.FileAlreadyExists); ok {
				directoryAlreadyPresent = true
				framework.Logf("Directory with the path %+q is already present", directoryPath)
			}
		}
		if !directoryAlreadyPresent {
			framework.Logf("Cannot create dir %#v. err %s", directoryPath, err)
			return "", err
		}
	}
	framework.Logf("Created dir with path as %+q", directoryPath)
	vmdkPath := directoryPath + volumeOptions.Name + ".vmdk"

	// Create a virtual disk manager
	vdm := object.NewVirtualDiskManager(ds.Client())
	// Create specification for new virtual disk
	vmDiskSpec := &types.FileBackedVirtualDiskSpec{
		VirtualDiskSpec: types.VirtualDiskSpec{
			AdapterType: volumeOptions.SCSIControllerType,
			DiskType:    volumeOptions.DiskFormat,
		},
		CapacityKb: int64(volumeOptions.CapacityKB),
	}
	// Create virtual disk
	task, err := vdm.CreateVirtualDisk(ctx, vmdkPath, datacenter, vmDiskSpec)
	if err != nil {
		framework.Logf("Failed to create virtual disk: %s. err: %+v", vmdkPath, err)
		return "", err
	}
	taskInfo, err := task.WaitForResult(ctx, nil)
	if err != nil {
		framework.Logf("Failed to complete virtual disk creation: %s. err: %+v", vmdkPath, err)
		return "", err
	}
	volumePath := taskInfo.Result.(string)
	canonicalDiskPath, err := getCanonicalVolumePath(ctx, datacenter, volumePath)
	if err != nil {
		return "", err
	}
	return canonicalDiskPath, nil
}

// DeleteVolume deletes the vmdk file specified in the volumePath.
// if an error is encountered while deleting volume, error is returned.
func (vs *VSphere) DeleteVolume(volumePath string, dataCenterRef types.ManagedObjectReference) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	Connect(ctx, vs)

	datacenter := object.NewDatacenter(vs.Client.Client, dataCenterRef)
	virtualDiskManager := object.NewVirtualDiskManager(datacenter.Client())
	diskPath := removeStorageClusterORFolderNameFromVDiskPath(volumePath)
	// Delete virtual disk
	task, err := virtualDiskManager.DeleteVirtualDisk(ctx, diskPath, datacenter)
	if err != nil {
		framework.Logf("Failed to delete virtual disk. err: %v", err)
		return err
	}
	err = task.Wait(ctx)
	if err != nil {
		framework.Logf("Failed to delete virtual disk. err: %v", err)
		return err
	}
	return nil
}

// IsVMPresent checks if VM with the name specified in the vmName argument, is present in the vCenter inventory.
// if VM is present, function returns true else false.
func (vs *VSphere) IsVMPresent(vmName string, dataCenterRef types.ManagedObjectReference) (isVMPresent bool, err error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	Connect(ctx, vs)
	folderMor, err := vs.GetFolderByPath(ctx, dataCenterRef, vs.Config.Folder)
	if err != nil {
		return
	}
	vmFolder := object.NewFolder(vs.Client.Client, folderMor)
	vmFoldersChildren, err := vmFolder.Children(ctx)
	if err != nil {
		framework.Logf("Failed to get children from Folder: %s. err: %+v", vmFolder.InventoryPath, err)
		return
	}
	for _, vmFoldersChild := range vmFoldersChildren {
		if vmFoldersChild.Reference().Type == virtualMachineType {
			if object.NewVirtualMachine(vs.Client.Client, vmFoldersChild.Reference()).Name() == vmName {
				return true, nil
			}
		}
	}
	return
}

// initVolumeOptions function sets default values for volumeOptions parameters if not set
func (vs *VSphere) initVolumeOptions(volumeOptions *VolumeOptions) {
	if volumeOptions == nil {
		volumeOptions = &VolumeOptions{}
	}
	if volumeOptions.Datastore == "" {
		volumeOptions.Datastore = vs.Config.DefaultDatastore
	}
	if volumeOptions.CapacityKB == 0 {
		volumeOptions.CapacityKB = defaultDiskCapacityKB
	}
	if volumeOptions.Name == "" {
		volumeOptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	if volumeOptions.DiskFormat == "" {
		volumeOptions.DiskFormat = defaultDiskFormat
	}
	if volumeOptions.SCSIControllerType == "" {
		volumeOptions.SCSIControllerType = defaultSCSIControllerType
	}
}
