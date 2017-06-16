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
	"os"
	"runtime"
	"strings"

	"fmt"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/pbm"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"

	pbmtypes "github.com/vmware/govmomi/pbm/types"
)

const (
	DatastoreProperty     = "datastore"
	DatastoreInfoProperty = "info"
	Folder                = "Folder"
	VirtualMachine        = "VirtualMachine"
)

// Reads vSphere configuration from system environment and construct vSphere object
func GetVSphere() (*VSphere, error) {
	cfg := getVSphereConfig()
	client, err := GetgovmomiClient(cfg)
	if err != nil {
		return nil, err
	}
	vs := &VSphere{
		client:          client,
		cfg:             cfg,
		localInstanceID: "",
	}
	runtime.SetFinalizer(vs, logout)
	return vs, nil
}

func getVSphereConfig() *VSphereConfig {
	var cfg VSphereConfig
	cfg.Global.VCenterIP = os.Getenv("VSPHERE_VCENTER")
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

func GetgovmomiClient(cfg *VSphereConfig) (*govmomi.Client, error) {
	if cfg == nil {
		cfg = getVSphereConfig()
	}
	client, err := newClient(context.TODO(), cfg)
	return client, err
}

// Get placement compatibility result based on storage policy requirements.
func (vs *VSphere) GetPlacementCompatibilityResult(ctx context.Context, pbmClient *pbm.Client, storagePolicyID string) (pbm.PlacementCompatibilityResult, error) {
	datastores, err := vs.getSharedDatastoresInK8SCluster(ctx)
	if err != nil {
		return nil, err
	}
	var hubs []pbmtypes.PbmPlacementHub
	for _, ds := range datastores {
		hubs = append(hubs, pbmtypes.PbmPlacementHub{
			HubType: ds.Type,
			HubId:   ds.Value,
		})
	}
	req := []pbmtypes.BasePbmPlacementRequirement{
		&pbmtypes.PbmPlacementCapabilityProfileRequirement{
			ProfileId: pbmtypes.PbmProfileId{
				UniqueId: storagePolicyID,
			},
		},
	}
	res, err := pbmClient.CheckRequirements(ctx, hubs, nil, req)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Verify if the user specified datastore is in the list of non-compatible datastores.
// If yes, return the non compatible datastore reference.
func (vs *VSphere) IsUserSpecifiedDatastoreNonCompatible(ctx context.Context, compatibilityResult pbm.PlacementCompatibilityResult, dsName string) (bool, *types.ManagedObjectReference) {
	dsMoList := vs.GetNonCompatibleDatastoresMo(ctx, compatibilityResult)
	for _, ds := range dsMoList {
		if ds.Info.GetDatastoreInfo().Name == dsName {
			dsMoRef := ds.Reference()
			return true, &dsMoRef
		}
	}
	return false, nil
}

func GetNonCompatibleDatastoreFaultMsg(compatibilityResult pbm.PlacementCompatibilityResult, dsMoref types.ManagedObjectReference) string {
	var faultMsg string
	for _, res := range compatibilityResult {
		if res.Hub.HubId == dsMoref.Value {
			for _, err := range res.Error {
				faultMsg = faultMsg + err.LocalizedMessage
			}
		}
	}
	return faultMsg
}

// Get the best fit compatible datastore by free space.
func GetMostFreeDatastore(dsMo []mo.Datastore) mo.Datastore {
	var curMax int64
	curMax = -1
	var index int
	for i, ds := range dsMo {
		dsFreeSpace := ds.Info.GetDatastoreInfo().FreeSpace
		if dsFreeSpace > curMax {
			curMax = dsFreeSpace
			index = i
		}
	}
	return dsMo[index]
}

func (vs *VSphere) GetCompatibleDatastoresMo(ctx context.Context, compatibilityResult pbm.PlacementCompatibilityResult) ([]mo.Datastore, error) {
	compatibleHubs := compatibilityResult.CompatibleDatastores()
	// Return an error if there are no compatible datastores.
	if len(compatibleHubs) < 1 {
		return nil, fmt.Errorf("There are no compatible datastores that satisfy the storage policy requirements")
	}
	dsMoList, err := vs.getDatastoreMo(ctx, compatibleHubs)
	if err != nil {
		return nil, err
	}
	return dsMoList, nil
}

func (vs *VSphere) GetNonCompatibleDatastoresMo(ctx context.Context, compatibilityResult pbm.PlacementCompatibilityResult) []mo.Datastore {
	nonCompatibleHubs := compatibilityResult.NonCompatibleDatastores()
	// Return an error if there are no compatible datastores.
	if len(nonCompatibleHubs) < 1 {
		return nil
	}
	dsMoList, err := vs.getDatastoreMo(ctx, nonCompatibleHubs)
	if err != nil {
		return nil
	}
	return dsMoList
}

// Get the datastore managed objects for the place hubs using property collector.
func (vs *VSphere) getDatastoreMo(ctx context.Context, hubs []pbmtypes.PbmPlacementHub) ([]mo.Datastore, error) {
	var dsMoRefs []types.ManagedObjectReference
	for _, hub := range hubs {
		dsMoRefs = append(dsMoRefs, types.ManagedObjectReference{
			Type:  hub.HubType,
			Value: hub.HubId,
		})
	}

	pc := property.DefaultCollector(vs.client.Client)
	var dsMoList []mo.Datastore
	err := pc.Retrieve(ctx, dsMoRefs, []string{DatastoreInfoProperty}, &dsMoList)
	if err != nil {
		return nil, err
	}
	return dsMoList, nil
}

// Get all datastores accessible for the virtual machine object.
func (vs *VSphere) getSharedDatastoresInK8SCluster(ctx context.Context) ([]types.ManagedObjectReference, error) {
	f := find.NewFinder(vs.client.Client, true)
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	f.SetDatacenter(dc)
	vmFolder, err := f.Folder(ctx, strings.TrimSuffix(vs.cfg.Global.WorkingDir, "/"))
	if err != nil {
		return nil, err
	}
	vmMoList, err := vs.GetVMsInsideFolder(ctx, vmFolder, []string{NameProperty})
	if err != nil {
		return nil, err
	}
	index := 0
	var sharedDs []string
	for _, vmMo := range vmMoList {
		if !strings.HasPrefix(vmMo.Name, DummyVMPrefixName) {
			accessibleDatastores, err := vs.getAllAccessibleDatastores(ctx, vmMo)
			if err != nil {
				return nil, err
			}
			if index == 0 {
				sharedDs = accessibleDatastores
			} else {
				sharedDs = intersect(sharedDs, accessibleDatastores)
				if len(sharedDs) == 0 {
					return nil, fmt.Errorf("No shared datastores found in the Kubernetes cluster")
				}
			}
			index++
		}
	}
	var sharedDSMorefs []types.ManagedObjectReference
	for _, ds := range sharedDs {
		sharedDSMorefs = append(sharedDSMorefs, types.ManagedObjectReference{
			Value: ds,
			Type:  "Datastore",
		})
	}
	return sharedDSMorefs, nil
}

func intersect(list1 []string, list2 []string) []string {
	var sharedList []string
	for _, val1 := range list1 {
		// Check if val1 is found in list2
		for _, val2 := range list2 {
			if val1 == val2 {
				sharedList = append(sharedList, val1)
				break
			}
		}
	}
	return sharedList
}

// Get the VM list inside a folder.
func (vs *VSphere) GetVMsInsideFolder(ctx context.Context, vmFolder *object.Folder, properties []string) ([]mo.VirtualMachine, error) {
	vmFolders, err := vmFolder.Children(ctx)
	if err != nil {
		return nil, err
	}

	pc := property.DefaultCollector(vs.client.Client)
	var vmRefs []types.ManagedObjectReference
	var vmMoList []mo.VirtualMachine
	for _, vmFolder := range vmFolders {
		if vmFolder.Reference().Type == VirtualMachine {
			vmRefs = append(vmRefs, vmFolder.Reference())
		}
	}
	err = pc.Retrieve(ctx, vmRefs, properties, &vmMoList)
	if err != nil {
		return nil, err
	}
	return vmMoList, nil
}

// Get the datastores accessible for the virtual machine object.
func (vs *VSphere) getAllAccessibleDatastores(ctx context.Context, vmMo mo.VirtualMachine) ([]string, error) {
	f := find.NewFinder(vs.client.Client, true)
	dc, err := f.Datacenter(ctx, vs.cfg.Global.Datacenter)
	if err != nil {
		return nil, err
	}
	f.SetDatacenter(dc)
	vmRegex := vs.cfg.Global.WorkingDir + vmMo.Name
	vmObj, err := f.VirtualMachine(ctx, vmRegex)
	if err != nil {
		return nil, err
	}

	host, err := vmObj.HostSystem(ctx)
	if err != nil {
		return nil, err
	}

	var hostSystemMo mo.HostSystem
	s := object.NewSearchIndex(vs.client.Client)
	err = s.Properties(ctx, host.Reference(), []string{DatastoreProperty}, &hostSystemMo)
	if err != nil {
		return nil, err
	}

	var dsRefValues []string
	for _, dsRef := range hostSystemMo.Datastore {
		dsRefValues = append(dsRefValues, dsRef.Value)
	}
	return dsRefValues, nil
}
