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

package simulator

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type VStorageObject struct {
	types.VStorageObject
	types.VStorageObjectSnapshotInfo
}

type VcenterVStorageObjectManager struct {
	mo.VcenterVStorageObjectManager

	objects map[types.ManagedObjectReference]map[types.ID]*VStorageObject
}

func NewVcenterVStorageObjectManager(ref types.ManagedObjectReference) object.Reference {
	m := &VcenterVStorageObjectManager{}
	m.Self = ref
	m.objects = make(map[types.ManagedObjectReference]map[types.ID]*VStorageObject)
	return m
}

func (m *VcenterVStorageObjectManager) object(ds types.ManagedObjectReference, id types.ID) *VStorageObject {
	if objects, ok := m.objects[ds]; ok {
		return objects[id]
	}
	return nil
}

func (m *VcenterVStorageObjectManager) ListVStorageObject(req *types.ListVStorageObject) soap.HasFault {
	body := &methods.ListVStorageObjectBody{
		Res: &types.ListVStorageObjectResponse{},
	}

	if objects, ok := m.objects[req.Datastore]; ok {
		for id := range objects {
			body.Res.Returnval = append(body.Res.Returnval, id)
		}
	}

	return body
}

func (m *VcenterVStorageObjectManager) RetrieveVStorageObject(req *types.RetrieveVStorageObject) soap.HasFault {
	body := new(methods.RetrieveVStorageObjectBody)

	obj := m.object(req.Datastore, req.Id)
	if obj == nil {
		body.Fault_ = Fault("", new(types.InvalidArgument))
	} else {
		body.Res = &types.RetrieveVStorageObjectResponse{
			Returnval: obj.VStorageObject,
		}
	}

	return body
}

func (m *VcenterVStorageObjectManager) RegisterDisk(ctx *Context, req *types.RegisterDisk) soap.HasFault {
	body := new(methods.RegisterDiskBody)

	invalid := func() soap.HasFault {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "path"})
		return body
	}

	u, err := url.Parse(req.Path)
	if err != nil {
		return invalid()
	}
	u.Path = strings.TrimPrefix(u.Path, folderPrefix)

	ds, err := ctx.svc.findDatastore(u.Query())
	if err != nil {
		return invalid()
	}

	st, err := os.Stat(filepath.Join(ds.Info.GetDatastoreInfo().Url, u.Path))
	if err != nil {
		return invalid()

	}
	if st.IsDir() {
		return invalid()
	}

	path := (&object.DatastorePath{Datastore: ds.Name, Path: u.Path}).String()

	for _, obj := range m.objects[ds.Self] {
		backing := obj.Config.BaseConfigInfo.Backing.(*types.BaseConfigInfoDiskFileBackingInfo)
		if backing.FilePath == path {
			return invalid()
		}
	}

	creq := &types.CreateDisk_Task{
		Spec: types.VslmCreateSpec{
			Name: req.Name,
			BackingSpec: &types.VslmCreateSpecDiskFileBackingSpec{
				VslmCreateSpecBackingSpec: types.VslmCreateSpecBackingSpec{
					Datastore: ds.Self,
					Path:      u.Path,
				},
			},
		},
	}

	obj, fault := m.createObject(creq, true)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = &types.RegisterDiskResponse{
		Returnval: *obj,
	}

	return body
}

func (m *VcenterVStorageObjectManager) createObject(req *types.CreateDisk_Task, register bool) (*types.VStorageObject, types.BaseMethodFault) {
	dir := "fcd"
	ref := req.Spec.BackingSpec.GetVslmCreateSpecBackingSpec().Datastore
	ds := Map.Get(ref).(*Datastore)
	dc := Map.getEntityDatacenter(ds)
	dm := Map.VirtualDiskManager()

	objects, ok := m.objects[ds.Self]
	if !ok {
		objects = make(map[types.ID]*VStorageObject)
		m.objects[ds.Self] = objects
		_ = os.Mkdir(filepath.Join(ds.Info.GetDatastoreInfo().Url, dir), 0755)
	}

	id := uuid.New().String()
	obj := types.VStorageObject{
		Config: types.VStorageObjectConfigInfo{
			BaseConfigInfo: types.BaseConfigInfo{
				Id: types.ID{
					Id: id,
				},
				Name:                        req.Spec.Name,
				CreateTime:                  time.Now(),
				KeepAfterDeleteVm:           req.Spec.KeepAfterDeleteVm,
				RelocationDisabled:          types.NewBool(false),
				NativeSnapshotSupported:     types.NewBool(false),
				ChangedBlockTrackingEnabled: types.NewBool(false),
				Iofilter:                    nil,
			},
			CapacityInMB:    req.Spec.CapacityInMB,
			ConsumptionType: []string{"disk"},
			ConsumerId:      nil,
		},
	}

	backing := req.Spec.BackingSpec.(*types.VslmCreateSpecDiskFileBackingSpec)
	path := object.DatastorePath{
		Datastore: ds.Name,
		Path:      backing.Path,
	}
	if path.Path == "" {
		path.Path = dir + "/" + id + ".vmdk"
	}

	if register == false {
		err := dm.createVirtualDisk(types.VirtualDeviceConfigSpecFileOperationCreate, &types.CreateVirtualDisk_Task{
			Datacenter: &dc.Self,
			Name:       path.String(),
		})
		if err != nil {
			return nil, err
		}
	}

	obj.Config.BaseConfigInfo.Backing = &types.BaseConfigInfoDiskFileBackingInfo{
		BaseConfigInfoFileBackingInfo: types.BaseConfigInfoFileBackingInfo{
			BaseConfigInfoBackingInfo: types.BaseConfigInfoBackingInfo{
				Datastore: ds.Self,
			},
			FilePath:        path.String(),
			BackingObjectId: uuid.New().String(),
			Parent:          nil,
			DeltaSizeInMB:   0,
		},
		ProvisioningType: backing.ProvisioningType,
	}

	objects[obj.Config.Id] = &VStorageObject{VStorageObject: obj}

	return &obj, nil

}

func (m *VcenterVStorageObjectManager) CreateDiskTask(req *types.CreateDisk_Task) soap.HasFault {
	task := CreateTask(m, "createDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		return m.createObject(req, false)
	})

	return &methods.CreateDisk_TaskBody{
		Res: &types.CreateDisk_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VcenterVStorageObjectManager) DeleteVStorageObjectTask(req *types.DeleteVStorageObject_Task) soap.HasFault {
	task := CreateTask(m, "deleteDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		obj := m.object(req.Datastore, req.Id)
		if obj == nil {
			return nil, &types.InvalidArgument{}
		}

		backing := obj.Config.Backing.(*types.BaseConfigInfoDiskFileBackingInfo)
		ds := Map.Get(req.Datastore).(*Datastore)
		dc := Map.getEntityDatacenter(ds)
		dm := Map.VirtualDiskManager()
		dm.DeleteVirtualDiskTask(&types.DeleteVirtualDisk_Task{
			Name:       backing.FilePath,
			Datacenter: &dc.Self,
		})

		delete(m.objects[req.Datastore], req.Id)

		return nil, nil
	})

	return &methods.DeleteVStorageObject_TaskBody{
		Res: &types.DeleteVStorageObject_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VcenterVStorageObjectManager) RetrieveSnapshotInfo(req *types.RetrieveSnapshotInfo) soap.HasFault {
	body := new(methods.RetrieveSnapshotInfoBody)

	obj := m.object(req.Datastore, req.Id)
	if obj == nil {
		body.Fault_ = Fault("", new(types.InvalidArgument))
	} else {
		body.Res = &types.RetrieveSnapshotInfoResponse{
			Returnval: obj.VStorageObjectSnapshotInfo,
		}
	}

	return body
}

func (m *VcenterVStorageObjectManager) VStorageObjectCreateSnapshotTask(req *types.VStorageObjectCreateSnapshot_Task) soap.HasFault {
	task := CreateTask(m, "createSnapshot", func(*Task) (types.AnyType, types.BaseMethodFault) {
		obj := m.object(req.Datastore, req.Id)
		if obj == nil {
			return nil, new(types.InvalidArgument)
		}

		snapshot := types.VStorageObjectSnapshotInfoVStorageObjectSnapshot{
			Id: &types.ID{
				Id: uuid.New().String(),
			},
			BackingObjectId: uuid.New().String(),
			CreateTime:      time.Now(),
			Description:     req.Description,
		}
		obj.Snapshots = append(obj.Snapshots, snapshot)

		return snapshot.Id, nil
	})

	return &methods.VStorageObjectCreateSnapshot_TaskBody{
		Res: &types.VStorageObjectCreateSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VcenterVStorageObjectManager) DeleteSnapshotTask(req *types.DeleteSnapshot_Task) soap.HasFault {
	task := CreateTask(m, "deleteSnapshot", func(*Task) (types.AnyType, types.BaseMethodFault) {
		obj := m.object(req.Datastore, req.Id)
		if obj != nil {
			for i := range obj.Snapshots {
				if *obj.Snapshots[i].Id == req.SnapshotId {
					obj.Snapshots = append(obj.Snapshots[:i], obj.Snapshots[i+1:]...)
					return nil, nil
				}
			}
		}
		return nil, new(types.InvalidArgument)
	})

	return &methods.DeleteSnapshot_TaskBody{
		Res: &types.DeleteSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VcenterVStorageObjectManager) tagID(id types.ID) types.ManagedObjectReference {
	return types.ManagedObjectReference{
		Type:  "fcd",
		Value: id.Id,
	}
}

func (m *VcenterVStorageObjectManager) AttachTagToVStorageObject(ctx *Context, req *types.AttachTagToVStorageObject) soap.HasFault {
	body := new(methods.AttachTagToVStorageObjectBody)
	ref := m.tagID(req.Id)

	err := ctx.Map.tagManager.AttachTag(ref, types.VslmTagEntry{
		ParentCategoryName: req.Category,
		TagName:            req.Tag,
	})

	if err == nil {
		body.Res = new(types.AttachTagToVStorageObjectResponse)
	} else {
		body.Fault_ = Fault("", err)
	}

	return body
}

func (m *VcenterVStorageObjectManager) DetachTagFromVStorageObject(ctx *Context, req *types.DetachTagFromVStorageObject) soap.HasFault {
	body := new(methods.DetachTagFromVStorageObjectBody)
	ref := m.tagID(req.Id)

	err := ctx.Map.tagManager.DetachTag(ref, types.VslmTagEntry{
		ParentCategoryName: req.Category,
		TagName:            req.Tag,
	})

	if err == nil {
		body.Res = new(types.DetachTagFromVStorageObjectResponse)
	} else {
		body.Fault_ = Fault("", err)
	}

	return body
}

func (m *VcenterVStorageObjectManager) ListVStorageObjectsAttachedToTag(ctx *Context, req *types.ListVStorageObjectsAttachedToTag) soap.HasFault {
	body := new(methods.ListVStorageObjectsAttachedToTagBody)

	refs, err := ctx.Map.tagManager.AttachedObjects(types.VslmTagEntry{
		ParentCategoryName: req.Category,
		TagName:            req.Tag,
	})

	if err == nil {
		body.Res = new(types.ListVStorageObjectsAttachedToTagResponse)
		for _, ref := range refs {
			body.Res.Returnval = append(body.Res.Returnval, types.ID{Id: ref.Value})
		}
	} else {
		body.Fault_ = Fault("", err)
	}

	return body
}

func (m *VcenterVStorageObjectManager) ListTagsAttachedToVStorageObject(ctx *Context, req *types.ListTagsAttachedToVStorageObject) soap.HasFault {
	body := new(methods.ListTagsAttachedToVStorageObjectBody)
	ref := m.tagID(req.Id)

	tags, err := ctx.Map.tagManager.AttachedTags(ref)

	if err == nil {
		body.Res = &types.ListTagsAttachedToVStorageObjectResponse{
			Returnval: tags,
		}
	} else {
		body.Fault_ = Fault("", err)
	}

	return body
}
