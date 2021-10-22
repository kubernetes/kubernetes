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
	"io"
	"os"
	"path"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type FileManager struct {
	mo.FileManager
}

func NewFileManager(ref types.ManagedObjectReference) object.Reference {
	m := &FileManager{}
	m.Self = ref
	return m
}

func (f *FileManager) findDatastore(ref mo.Reference, name string) (*Datastore, types.BaseMethodFault) {
	var refs []types.ManagedObjectReference

	switch obj := ref.(type) {
	case *Folder:
		refs = obj.ChildEntity
	case *StoragePod:
		refs = obj.ChildEntity
	}

	for _, ref := range refs {
		switch obj := Map.Get(ref).(type) {
		case *Datastore:
			if obj.Name == name {
				return obj, nil
			}
		case *Folder, *StoragePod:
			ds, _ := f.findDatastore(obj, name)
			if ds != nil {
				return ds, nil
			}
		}
	}

	return nil, &types.InvalidDatastore{Name: name}
}

func (f *FileManager) resolve(dc *types.ManagedObjectReference, name string) (string, types.BaseMethodFault) {
	p, fault := parseDatastorePath(name)
	if fault != nil {
		return "", fault
	}

	if dc == nil {
		if Map.IsESX() {
			dc = &esx.Datacenter.Self
		} else {
			return "", &types.InvalidArgument{InvalidProperty: "dc"}
		}
	}

	folder := Map.Get(*dc).(*Datacenter).DatastoreFolder

	ds, fault := f.findDatastore(Map.Get(folder), p.Datastore)
	if fault != nil {
		return "", fault
	}

	dir := ds.Info.GetDatastoreInfo().Url

	return path.Join(dir, p.Path), nil
}

func (f *FileManager) fault(name string, err error, fault types.BaseFileFault) types.BaseMethodFault {
	switch {
	case os.IsNotExist(err):
		fault = new(types.FileNotFound)
	case os.IsExist(err):
		fault = new(types.FileAlreadyExists)
	}

	fault.GetFileFault().File = name

	return fault.(types.BaseMethodFault)
}

func (f *FileManager) deleteDatastoreFile(req *types.DeleteDatastoreFile_Task) types.BaseMethodFault {
	file, fault := f.resolve(req.Datacenter, req.Name)
	if fault != nil {
		return fault
	}

	_, err := os.Stat(file)
	if err != nil {
		if os.IsNotExist(err) {
			return f.fault(file, err, new(types.CannotDeleteFile))
		}
	}

	err = os.RemoveAll(file)
	if err != nil {
		return f.fault(file, err, new(types.CannotDeleteFile))
	}

	return nil
}

func (f *FileManager) DeleteDatastoreFileTask(req *types.DeleteDatastoreFile_Task) soap.HasFault {
	task := CreateTask(f, "deleteDatastoreFile", func(*Task) (types.AnyType, types.BaseMethodFault) {
		return nil, f.deleteDatastoreFile(req)
	})

	return &methods.DeleteDatastoreFile_TaskBody{
		Res: &types.DeleteDatastoreFile_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (f *FileManager) MakeDirectory(req *types.MakeDirectory) soap.HasFault {
	body := &methods.MakeDirectoryBody{}

	name, fault := f.resolve(req.Datacenter, req.Name)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	mkdir := os.Mkdir

	if isTrue(req.CreateParentDirectories) {
		mkdir = os.MkdirAll
	}

	err := mkdir(name, 0700)
	if err != nil {
		fault = f.fault(req.Name, err, new(types.CannotCreateFile))
		body.Fault_ = Fault(err.Error(), fault)
		return body
	}

	body.Res = new(types.MakeDirectoryResponse)
	return body
}

func (f *FileManager) moveDatastoreFile(req *types.MoveDatastoreFile_Task) types.BaseMethodFault {
	src, fault := f.resolve(req.SourceDatacenter, req.SourceName)
	if fault != nil {
		return fault
	}

	dst, fault := f.resolve(req.DestinationDatacenter, req.DestinationName)
	if fault != nil {
		return fault
	}

	if !isTrue(req.Force) {
		_, err := os.Stat(dst)
		if err == nil {
			return f.fault(dst, nil, new(types.FileAlreadyExists))
		}
	}

	err := os.Rename(src, dst)
	if err != nil {
		return f.fault(src, err, new(types.CannotAccessFile))
	}

	return nil
}

func (f *FileManager) MoveDatastoreFileTask(req *types.MoveDatastoreFile_Task) soap.HasFault {
	task := CreateTask(f, "moveDatastoreFile", func(*Task) (types.AnyType, types.BaseMethodFault) {
		return nil, f.moveDatastoreFile(req)
	})

	return &methods.MoveDatastoreFile_TaskBody{
		Res: &types.MoveDatastoreFile_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (f *FileManager) copyDatastoreFile(req *types.CopyDatastoreFile_Task) types.BaseMethodFault {
	src, fault := f.resolve(req.SourceDatacenter, req.SourceName)
	if fault != nil {
		return fault
	}

	dst, fault := f.resolve(req.DestinationDatacenter, req.DestinationName)
	if fault != nil {
		return fault
	}

	if !isTrue(req.Force) {
		_, err := os.Stat(dst)
		if err == nil {
			return f.fault(dst, nil, new(types.FileAlreadyExists))
		}
	}

	r, err := os.Open(src)
	if err != nil {
		return f.fault(dst, err, new(types.CannotAccessFile))
	}
	defer r.Close()

	w, err := os.Create(dst)
	if err != nil {
		return f.fault(dst, err, new(types.CannotCreateFile))
	}
	defer w.Close()

	if _, err = io.Copy(w, r); err != nil {
		return f.fault(dst, err, new(types.CannotCreateFile))
	}

	return nil
}

func (f *FileManager) CopyDatastoreFileTask(req *types.CopyDatastoreFile_Task) soap.HasFault {
	task := CreateTask(f, "copyDatastoreFile", func(*Task) (types.AnyType, types.BaseMethodFault) {
		return nil, f.copyDatastoreFile(req)
	})

	return &methods.CopyDatastoreFile_TaskBody{
		Res: &types.CopyDatastoreFile_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
