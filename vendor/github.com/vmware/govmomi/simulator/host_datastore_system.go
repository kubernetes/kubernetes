/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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
	"os"
	"path"

	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HostDatastoreSystem struct {
	mo.HostDatastoreSystem

	Host *mo.HostSystem
}

func (dss *HostDatastoreSystem) add(ctx *Context, ds *Datastore) *soap.Fault {
	info := ds.Info.GetDatastoreInfo()

	info.Name = ds.Name

	if e := ctx.Map.FindByName(ds.Name, dss.Datastore); e != nil {
		return Fault(e.Reference().Value, &types.DuplicateName{
			Name:   ds.Name,
			Object: e.Reference(),
		})
	}

	fi, err := os.Stat(info.Url)
	if err == nil && !fi.IsDir() {
		err = os.ErrInvalid
	}

	if err != nil {
		switch {
		case os.IsNotExist(err):
			return Fault(err.Error(), &types.NotFound{})
		default:
			return Fault(err.Error(), &types.HostConfigFault{})
		}
	}

	folder := ctx.Map.getEntityFolder(dss.Host, "datastore")

	found := false
	if e := ctx.Map.FindByName(ds.Name, folder.ChildEntity); e != nil {
		if e.Reference().Type != "Datastore" {
			return Fault(e.Reference().Value, &types.DuplicateName{
				Name:   ds.Name,
				Object: e.Reference(),
			})
		}

		// if datastore already exists, use current reference
		found = true
		ds.Self = e.Reference()
	} else {
		// put datastore to folder and generate reference
		folderPutChild(ctx, folder, ds)
	}

	ds.Summary.Datastore = &ds.Self
	ds.Summary.Name = ds.Name
	ds.Summary.Url = info.Url
	ds.Capability = types.DatastoreCapability{
		DirectoryHierarchySupported:      true,
		RawDiskMappingsSupported:         false,
		PerFileThinProvisioningSupported: true,
		StorageIORMSupported:             types.NewBool(true),
		NativeSnapshotSupported:          types.NewBool(false),
		TopLevelDirectoryCreateSupported: types.NewBool(true),
		SeSparseSupported:                types.NewBool(true),
	}

	dss.Datastore = append(dss.Datastore, ds.Self)
	dss.Host.Datastore = dss.Datastore
	parent := hostParent(dss.Host)
	ctx.Map.AddReference(ctx, parent, &parent.Datastore, ds.Self)

	// NOTE: browser must be created after ds is appended to dss.Datastore
	if !found {
		browser := &HostDatastoreBrowser{}
		browser.Datastore = dss.Datastore
		ds.Browser = ctx.Map.Put(browser).Reference()

		ds.Summary.Capacity = int64(units.TB * 10)
		ds.Summary.FreeSpace = ds.Summary.Capacity

		info.FreeSpace = ds.Summary.FreeSpace
		info.MaxMemoryFileSize = ds.Summary.Capacity
		info.MaxFileSize = ds.Summary.Capacity
	}

	return nil
}

func (dss *HostDatastoreSystem) CreateLocalDatastore(ctx *Context, c *types.CreateLocalDatastore) soap.HasFault {
	r := &methods.CreateLocalDatastoreBody{}

	ds := &Datastore{}
	ds.Name = c.Name

	ds.Info = &types.LocalDatastoreInfo{
		DatastoreInfo: types.DatastoreInfo{
			Name: c.Name,
			Url:  c.Path,
		},
		Path: c.Path,
	}

	ds.Summary.Type = string(types.HostFileSystemVolumeFileSystemTypeOTHER)
	ds.Summary.MaintenanceMode = string(types.DatastoreSummaryMaintenanceModeStateNormal)
	ds.Summary.Accessible = true

	if err := dss.add(ctx, ds); err != nil {
		r.Fault_ = err
		return r
	}

	ds.Host = append(ds.Host, types.DatastoreHostMount{
		Key: dss.Host.Reference(),
		MountInfo: types.HostMountInfo{
			AccessMode: string(types.HostMountModeReadWrite),
			Mounted:    types.NewBool(true),
			Accessible: types.NewBool(true),
		},
	})

	_ = ds.RefreshDatastore(&types.RefreshDatastore{This: ds.Self})

	r.Res = &types.CreateLocalDatastoreResponse{
		Returnval: ds.Self,
	}

	return r
}

func (dss *HostDatastoreSystem) CreateNasDatastore(ctx *Context, c *types.CreateNasDatastore) soap.HasFault {
	r := &methods.CreateNasDatastoreBody{}

	// validate RemoteHost and RemotePath are specified
	if c.Spec.RemoteHost == "" {
		r.Fault_ = Fault(
			"A specified parameter was not correct: Spec.RemoteHost",
			&types.InvalidArgument{InvalidProperty: "RemoteHost"},
		)
		return r
	}
	if c.Spec.RemotePath == "" {
		r.Fault_ = Fault(
			"A specified parameter was not correct: Spec.RemotePath",
			&types.InvalidArgument{InvalidProperty: "RemotePath"},
		)
		return r
	}

	ds := &Datastore{}
	ds.Name = path.Base(c.Spec.LocalPath)

	ds.Info = &types.NasDatastoreInfo{
		DatastoreInfo: types.DatastoreInfo{
			Url: c.Spec.LocalPath,
		},
		Nas: &types.HostNasVolume{
			HostFileSystemVolume: types.HostFileSystemVolume{
				Name: c.Spec.LocalPath,
				Type: c.Spec.Type,
			},
			RemoteHost: c.Spec.RemoteHost,
			RemotePath: c.Spec.RemotePath,
		},
	}

	ds.Summary.Type = c.Spec.Type
	ds.Summary.MaintenanceMode = string(types.DatastoreSummaryMaintenanceModeStateNormal)
	ds.Summary.Accessible = true

	if err := dss.add(ctx, ds); err != nil {
		r.Fault_ = err
		return r
	}

	_ = ds.RefreshDatastore(&types.RefreshDatastore{This: ds.Self})

	r.Res = &types.CreateNasDatastoreResponse{
		Returnval: ds.Self,
	}

	return r
}
