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
	"io/ioutil"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"

	"fmt"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib/diskmanagers"
)

const (
	DatastoreProperty     = "datastore"
	DatastoreInfoProperty = "info"
	Folder                = "Folder"
	VirtualMachine        = "VirtualMachine"
)

// GetVSphere reads vSphere configuration from system environment and construct vSphere object
func GetVSphere() (*VSphere, error) {
	cfg := getVSphereConfig()
	vSphereConn := getVSphereConn(cfg)
	client, err := GetgovmomiClient(vSphereConn)
	if err != nil {
		return nil, err
	}
	vSphereConn.GoVmomiClient = client
	vs := &VSphere{
		conn:            vSphereConn,
		cfg:             cfg,
		localInstanceID: "",
	}
	runtime.SetFinalizer(vs, logout)
	return vs, nil
}

func getVSphereConfig() *VSphereConfig {
	var cfg VSphereConfig
	cfg.Global.VCenterIP = os.Getenv("VSPHERE_VCENTER")
	cfg.Global.VCenterPort = os.Getenv("VSPHERE_VCENTER_PORT")
	cfg.Global.User = os.Getenv("VSPHERE_USER")
	cfg.Global.Password = os.Getenv("VSPHERE_PASSWORD")
	cfg.Global.Datacenter = os.Getenv("VSPHERE_DATACENTER")
	cfg.Global.Datastore = os.Getenv("VSPHERE_DATASTORE")
	cfg.Global.WorkingDir = os.Getenv("VSPHERE_WORKING_DIR")
	cfg.Global.VMName = os.Getenv("VSPHERE_VM_NAME")
	cfg.Global.InsecureFlag = false
	if strings.ToLower(os.Getenv("VSPHERE_INSECURE")) == "true" {
		cfg.Global.InsecureFlag = true
	}
	return &cfg
}

func getVSphereConn(cfg *VSphereConfig) *vclib.VSphereConnection {
	vSphereConn := &vclib.VSphereConnection{
		Username:          cfg.Global.User,
		Password:          cfg.Global.Password,
		Hostname:          cfg.Global.VCenterIP,
		Insecure:          cfg.Global.InsecureFlag,
		RoundTripperCount: cfg.Global.RoundTripperCount,
		Port:              cfg.Global.VCenterPort,
	}
	return vSphereConn
}

// GetgovmomiClient gets the goVMOMI client for the vsphere connection object
func GetgovmomiClient(conn *vclib.VSphereConnection) (*govmomi.Client, error) {
	if conn == nil {
		cfg := getVSphereConfig()
		conn = getVSphereConn(cfg)
	}
	client, err := conn.NewClient(context.TODO())
	return client, err
}

// getvmUUID gets the BIOS UUID via the sys interface.  This UUID is known by vsphere
func getvmUUID() (string, error) {
	id, err := ioutil.ReadFile(UUIDPath)
	if err != nil {
		return "", fmt.Errorf("error retrieving vm uuid: %s", err)
	}
	uuidFromFile := string(id[:])
	//strip leading and trailing white space and new line char
	uuid := strings.TrimSpace(uuidFromFile)
	// check the uuid starts with "VMware-"
	if !strings.HasPrefix(uuid, UUIDPrefix) {
		return "", fmt.Errorf("Failed to match Prefix, UUID read from the file is %v", uuidFromFile)
	}
	// Strip the prefix and while spaces and -
	uuid = strings.Replace(uuid[len(UUIDPrefix):(len(uuid))], " ", "", -1)
	uuid = strings.Replace(uuid, "-", "", -1)
	if len(uuid) != 32 {
		return "", fmt.Errorf("Length check failed, UUID read from the file is %v", uuidFromFile)
	}
	// need to add dashes, e.g. "564d395e-d807-e18a-cb25-b79f65eb2b9f"
	uuid = fmt.Sprintf("%s-%s-%s-%s-%s", uuid[0:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:32])
	return uuid, nil
}

// Get all datastores accessible for the virtual machine object.
func getSharedDatastoresInK8SCluster(ctx context.Context, folder *vclib.Folder) ([]*vclib.Datastore, error) {
	vmList, err := folder.GetVirtualMachines(ctx)
	if err != nil {
		glog.Errorf("Failed to get virtual machines in the kubernetes cluster: %s, err: %+v", folder.InventoryPath, err)
		return nil, err
	}
	if vmList == nil || len(vmList) == 0 {
		glog.Errorf("No virtual machines found in the kubernetes cluster: %s", folder.InventoryPath)
		return nil, fmt.Errorf("No virtual machines found in the kubernetes cluster: %s", folder.InventoryPath)
	}
	index := 0
	var sharedDatastores []*vclib.Datastore
	for _, vm := range vmList {
		vmName, err := vm.ObjectName(ctx)
		if err != nil {
			return nil, err
		}
		if !strings.HasPrefix(vmName, DummyVMPrefixName) {
			accessibleDatastores, err := vm.GetAllAccessibleDatastores(ctx)
			if err != nil {
				return nil, err
			}
			if index == 0 {
				sharedDatastores = accessibleDatastores
			} else {
				sharedDatastores = intersect(sharedDatastores, accessibleDatastores)
				if len(sharedDatastores) == 0 {
					return nil, fmt.Errorf("No shared datastores found in the Kubernetes cluster: %s", folder.InventoryPath)
				}
			}
			index++
		}
	}
	return sharedDatastores, nil
}

func intersect(list1 []*vclib.Datastore, list2 []*vclib.Datastore) []*vclib.Datastore {
	var sharedDs []*vclib.Datastore
	for _, val1 := range list1 {
		// Check if val1 is found in list2
		for _, val2 := range list2 {
			if val1.Reference().Value == val2.Reference().Value {
				sharedDs = append(sharedDs, val1)
				break
			}
		}
	}
	return sharedDs
}

// Get the datastores accessible for the virtual machine object.
func getAllAccessibleDatastores(ctx context.Context, client *vim25.Client, vmMo mo.VirtualMachine) ([]string, error) {
	host := vmMo.Summary.Runtime.Host
	if host == nil {
		return nil, errors.New("VM doesn't have a HostSystem")
	}
	var hostSystemMo mo.HostSystem
	s := object.NewSearchIndex(client)
	err := s.Properties(ctx, host.Reference(), []string{DatastoreProperty}, &hostSystemMo)
	if err != nil {
		return nil, err
	}
	var dsRefValues []string
	for _, dsRef := range hostSystemMo.Datastore {
		dsRefValues = append(dsRefValues, dsRef.Value)
	}
	return dsRefValues, nil
}

// getMostFreeDatastore gets the best fit compatible datastore by free space.
func getMostFreeDatastoreName(ctx context.Context, client *vim25.Client, dsObjList []*vclib.Datastore) (string, error) {
	dsMoList, err := dsObjList[0].Datacenter.GetDatastoreMoList(ctx, dsObjList, []string{DatastoreInfoProperty})
	if err != nil {
		return "", err
	}
	var curMax int64
	curMax = -1
	var index int
	for i, dsMo := range dsMoList {
		dsFreeSpace := dsMo.Info.GetDatastoreInfo().FreeSpace
		if dsFreeSpace > curMax {
			curMax = dsFreeSpace
			index = i
		}
	}
	return dsMoList[index].Info.GetDatastoreInfo().Name, nil
}

func getPbmCompatibleDatastore(ctx context.Context, client *vim25.Client, storagePolicyName string, folder *vclib.Folder) (string, error) {
	pbmClient, err := vclib.NewPbmClient(ctx, client)
	if err != nil {
		return "", err
	}
	storagePolicyID, err := pbmClient.ProfileIDByName(ctx, storagePolicyName)
	if err != nil {
		glog.Errorf("Failed to get Profile ID by name: %s. err: %+v", storagePolicyName, err)
		return "", err
	}
	sharedDsList, err := getSharedDatastoresInK8SCluster(ctx, folder)
	if err != nil {
		glog.Errorf("Failed to get shared datastores from kubernetes cluster: %s. err: %+v", folder.InventoryPath, err)
		return "", err
	}
	compatibleDatastores, _, err := pbmClient.GetCompatibleDatastores(ctx, storagePolicyID, sharedDsList)
	if err != nil {
		glog.Errorf("Failed to get compatible datastores from datastores : %+v with storagePolicy: %s. err: %+v", sharedDsList, storagePolicyID, err)
		return "", err
	}
	datastore, err := getMostFreeDatastoreName(ctx, client, compatibleDatastores)
	if err != nil {
		glog.Errorf("Failed to get most free datastore from compatible datastores: %+v. err: %+v", compatibleDatastores, err)
		return "", err
	}
	return datastore, err
}

func (vs *VSphere) setVMOptions(ctx context.Context, dc *vclib.Datacenter) (*vclib.VMOptions, error) {
	var vmOptions vclib.VMOptions
	vm, err := dc.GetVMByPath(ctx, vs.cfg.Global.WorkingDir+"/"+vs.localInstanceID)
	if err != nil {
		return nil, err
	}
	resourcePool, err := vm.GetResourcePool(ctx)
	if err != nil {
		return nil, err
	}
	folder, err := dc.GetFolderByPath(ctx, vs.cfg.Global.WorkingDir)
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
		// Ensure client is logged in and session is valid
		err := vs.conn.Connect(ctx)
		if err != nil {
			glog.V(4).Infof("Failed to connect to VC with err: %+v. Retrying again...", err)
			continue
		}
		dc, err := vclib.GetDatacenter(ctx, vs.conn, vs.cfg.Global.Datacenter)
		if err != nil {
			glog.V(4).Infof("Failed to get the datacenter: %s from VC. err: %+v", vs.cfg.Global.Datacenter, err)
			continue
		}
		// Get the folder reference for global working directory where the dummy VM needs to be created.
		vmFolder, err := dc.GetFolderByPath(ctx, vs.cfg.Global.WorkingDir)
		if err != nil {
			glog.V(4).Infof("Unable to get the kubernetes folder: %q reference. err: %+v", vs.cfg.Global.WorkingDir, err)
			continue
		}
		// A write lock is acquired to make sure the cleanUp routine doesn't delete any VM's created by ongoing PVC requests.
		defer cleanUpDummyVMLock.Lock()
		err = diskmanagers.CleanUpDummyVMs(ctx, vmFolder, dc)
		if err != nil {
			glog.V(4).Infof("Unable to clean up dummy VM's in the kubernetes cluster: %q. err: %+v", vs.cfg.Global.WorkingDir, err)
		}
	}
}
