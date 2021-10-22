/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package vslm

import (
	"context"
	"errors"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// ObjectManager wraps VStorageObjectManagerBase.
type ObjectManager struct {
	types.ManagedObjectReference
	c    *vim25.Client
	isVC bool
}

// NewObjectManager returns an ObjectManager referencing the VcenterVStorageObjectManager singleton when connected to vCenter or
// the HostVStorageObjectManager singleton when connected to an ESX host.  The optional ref param can be used to specify a ESX
// host instead, when connected to vCenter.
func NewObjectManager(client *vim25.Client, ref ...types.ManagedObjectReference) *ObjectManager {
	mref := *client.ServiceContent.VStorageObjectManager

	if len(ref) == 1 {
		mref = ref[0]
	}

	m := ObjectManager{
		ManagedObjectReference: mref,
		c:                      client,
		isVC:                   mref.Type == "VcenterVStorageObjectManager",
	}

	return &m
}

// PlaceDisk uses StorageResourceManager datastore placement recommendations to choose a Datastore from a Datastore cluster.
// If the given spec backing Datastore field is not that of type StoragePod, the spec is unmodifed.
// Otherwise, the backing Datastore field is replaced with a Datastore suggestion.
func (m ObjectManager) PlaceDisk(ctx context.Context, spec *types.VslmCreateSpec, pool types.ManagedObjectReference) error {
	backing := spec.BackingSpec.GetVslmCreateSpecBackingSpec()
	if backing.Datastore.Type != "StoragePod" {
		return nil
	}

	device := &types.VirtualDisk{
		VirtualDevice: types.VirtualDevice{
			Key: 0,
			Backing: &types.VirtualDiskFlatVer2BackingInfo{
				DiskMode:        string(types.VirtualDiskModePersistent),
				ThinProvisioned: types.NewBool(true),
			},
			UnitNumber: types.NewInt32(0),
		},
		CapacityInKB: spec.CapacityInMB * 1024,
	}

	storage := types.StoragePlacementSpec{
		Type:         string(types.StoragePlacementSpecPlacementTypeCreate),
		ResourcePool: &pool,
		PodSelectionSpec: types.StorageDrsPodSelectionSpec{
			StoragePod: &backing.Datastore,
			InitialVmConfig: []types.VmPodConfigForPlacement{
				{
					StoragePod: backing.Datastore,
					Disk: []types.PodDiskLocator{
						{
							DiskId:          device.Key,
							DiskBackingInfo: device.Backing,
						},
					},
				},
			},
		},
		ConfigSpec: &types.VirtualMachineConfigSpec{
			Name: spec.Name,
			DeviceChange: []types.BaseVirtualDeviceConfigSpec{
				&types.VirtualDeviceConfigSpec{
					Operation:     types.VirtualDeviceConfigSpecOperationAdd,
					FileOperation: types.VirtualDeviceConfigSpecFileOperationCreate,
					Device:        device,
				},
			},
		},
	}

	req := types.RecommendDatastores{
		This:        *m.c.ServiceContent.StorageResourceManager,
		StorageSpec: storage,
	}

	res, err := methods.RecommendDatastores(ctx, m.c, &req)
	if err != nil {
		return err
	}

	r := res.Returnval.Recommendations
	if len(r) == 0 {
		return errors.New("no storage placement recommendations")
	}

	backing.Datastore = r[0].Action[0].(*types.StoragePlacementAction).Destination

	return nil
}

func (m ObjectManager) CreateDisk(ctx context.Context, spec types.VslmCreateSpec) (*object.Task, error) {
	req := types.CreateDisk_Task{
		This: m.Reference(),
		Spec: spec,
	}

	if m.isVC {
		res, err := methods.CreateDisk_Task(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return object.NewTask(m.c, res.Returnval), nil
	}

	res, err := methods.HostCreateDisk_Task(ctx, m.c, (*types.HostCreateDisk_Task)(&req))
	if err != nil {
		return nil, err
	}

	return object.NewTask(m.c, res.Returnval), nil
}

func (m ObjectManager) Rename(ctx context.Context, ds mo.Reference, id, name string) error {
	req := types.RenameVStorageObject{
		This:      m.Reference(),
		Datastore: ds.Reference(),
		Id:        types.ID{Id: id},
		Name:      name,
	}

	if m.isVC {
		_, err := methods.RenameVStorageObject(ctx, m.c, &req)
		return err
	}

	_, err := methods.HostRenameVStorageObject(ctx, m.c, (*types.HostRenameVStorageObject)(&req))
	return err
}

func (m ObjectManager) Delete(ctx context.Context, ds mo.Reference, id string) (*object.Task, error) {
	req := types.DeleteVStorageObject_Task{
		This:      m.Reference(),
		Datastore: ds.Reference(),
		Id:        types.ID{Id: id},
	}

	if m.isVC {
		res, err := methods.DeleteVStorageObject_Task(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return object.NewTask(m.c, res.Returnval), nil
	}

	res, err := methods.HostDeleteVStorageObject_Task(ctx, m.c, (*types.HostDeleteVStorageObject_Task)(&req))
	if err != nil {
		return nil, err
	}

	return object.NewTask(m.c, res.Returnval), nil
}

func (m ObjectManager) Retrieve(ctx context.Context, ds mo.Reference, id string) (*types.VStorageObject, error) {
	req := types.RetrieveVStorageObject{
		This:      m.Reference(),
		Datastore: ds.Reference(),
		Id:        types.ID{Id: id},
	}

	if m.isVC {
		res, err := methods.RetrieveVStorageObject(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return &res.Returnval, nil
	}

	res, err := methods.HostRetrieveVStorageObject(ctx, m.c, (*types.HostRetrieveVStorageObject)(&req))
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m ObjectManager) List(ctx context.Context, ds mo.Reference) ([]types.ID, error) {
	req := types.ListVStorageObject{
		This:      m.Reference(),
		Datastore: ds.Reference(),
	}

	if m.isVC {
		res, err := methods.ListVStorageObject(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return res.Returnval, nil
	}

	res, err := methods.HostListVStorageObject(ctx, m.c, (*types.HostListVStorageObject)(&req))
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m ObjectManager) RegisterDisk(ctx context.Context, path, name string) (*types.VStorageObject, error) {
	req := types.RegisterDisk{
		This: m.Reference(),
		Path: path,
		Name: name,
	}

	if m.isVC {
		res, err := methods.RegisterDisk(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return &res.Returnval, nil
	}

	res, err := methods.HostRegisterDisk(ctx, m.c, (*types.HostRegisterDisk)(&req))
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m ObjectManager) Clone(ctx context.Context, ds mo.Reference, id string, spec types.VslmCloneSpec) (*object.Task, error) {
	req := types.CloneVStorageObject_Task{
		This:      m.Reference(),
		Datastore: ds.Reference(),
		Id:        types.ID{Id: id},
		Spec:      spec,
	}

	if m.isVC {
		res, err := methods.CloneVStorageObject_Task(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return object.NewTask(m.c, res.Returnval), nil
	}

	res, err := methods.HostCloneVStorageObject_Task(ctx, m.c, (*types.HostCloneVStorageObject_Task)(&req))
	if err != nil {
		return nil, err
	}

	return object.NewTask(m.c, res.Returnval), nil
}

func (m ObjectManager) CreateSnapshot(ctx context.Context, ds mo.Reference, id, desc string) (*object.Task, error) {
	req := types.VStorageObjectCreateSnapshot_Task{
		This:        m.Reference(),
		Id:          types.ID{Id: id},
		Description: desc,
		Datastore:   ds.Reference(),
	}

	if m.isVC {
		res, err := methods.VStorageObjectCreateSnapshot_Task(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return object.NewTask(m.c, res.Returnval), nil
	}

	res, err := methods.HostVStorageObjectCreateSnapshot_Task(ctx, m.c, (*types.HostVStorageObjectCreateSnapshot_Task)(&req))
	if err != nil {
		return nil, err
	}

	return object.NewTask(m.c, res.Returnval), nil
}

func (m ObjectManager) DeleteSnapshot(ctx context.Context, ds mo.Reference, id, sid string) (*object.Task, error) {
	req := types.DeleteSnapshot_Task{
		This:       m.Reference(),
		Datastore:  ds.Reference(),
		Id:         types.ID{Id: id},
		SnapshotId: types.ID{Id: sid},
	}

	if m.isVC {
		res, err := methods.DeleteSnapshot_Task(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return object.NewTask(m.c, res.Returnval), nil
	}

	res, err := methods.HostVStorageObjectDeleteSnapshot_Task(ctx, m.c, (*types.HostVStorageObjectDeleteSnapshot_Task)(&req))
	if err != nil {
		return nil, err
	}

	return object.NewTask(m.c, res.Returnval), nil
}

func (m ObjectManager) RetrieveSnapshotInfo(ctx context.Context, ds mo.Reference, id string) (*types.VStorageObjectSnapshotInfo, error) {
	req := types.RetrieveSnapshotInfo{
		This:      m.Reference(),
		Datastore: ds.Reference(),
		Id:        types.ID{Id: id},
	}

	if m.isVC {
		res, err := methods.RetrieveSnapshotInfo(ctx, m.c, &req)
		if err != nil {
			return nil, err
		}

		return &res.Returnval, nil
	}

	res, err := methods.HostVStorageObjectRetrieveSnapshotInfo(ctx, m.c, (*types.HostVStorageObjectRetrieveSnapshotInfo)(&req))
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m ObjectManager) AttachTag(ctx context.Context, id string, tag types.VslmTagEntry) error {
	req := &types.AttachTagToVStorageObject{
		This:     m.ManagedObjectReference,
		Id:       types.ID{Id: id},
		Category: tag.ParentCategoryName,
		Tag:      tag.TagName,
	}

	_, err := methods.AttachTagToVStorageObject(ctx, m.c, req)
	return err
}

func (m ObjectManager) DetachTag(ctx context.Context, id string, tag types.VslmTagEntry) error {
	req := &types.DetachTagFromVStorageObject{
		This:     m.ManagedObjectReference,
		Id:       types.ID{Id: id},
		Category: tag.ParentCategoryName,
		Tag:      tag.TagName,
	}

	_, err := methods.DetachTagFromVStorageObject(ctx, m.c, req)
	return err
}

func (m ObjectManager) ListAttachedObjects(ctx context.Context, category, tag string) ([]types.ID, error) {
	req := &types.ListVStorageObjectsAttachedToTag{
		This:     m.ManagedObjectReference,
		Category: category,
		Tag:      tag,
	}

	res, err := methods.ListVStorageObjectsAttachedToTag(ctx, m.c, req)
	if err != nil {
		return nil, err
	}
	return res.Returnval, nil
}

func (m ObjectManager) ListAttachedTags(ctx context.Context, id string) ([]types.VslmTagEntry, error) {
	req := &types.ListTagsAttachedToVStorageObject{
		This: m.ManagedObjectReference,
		Id:   types.ID{Id: id},
	}

	res, err := methods.ListTagsAttachedToVStorageObject(ctx, m.c, req)
	if err != nil {
		return nil, err
	}
	return res.Returnval, nil
}
