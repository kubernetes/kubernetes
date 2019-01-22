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
	"strings"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type VirtualDiskManager struct {
	mo.VirtualDiskManager
}

func NewVirtualDiskManager(ref types.ManagedObjectReference) object.Reference {
	m := &VirtualDiskManager{}
	m.Self = ref
	return m
}

func (m *VirtualDiskManager) names(name string) []string {
	return []string{
		strings.Replace(name, ".vmdk", "-flat.vmdk", 1),
		name,
	}
}

func (m *VirtualDiskManager) createVirtualDisk(op types.VirtualDeviceConfigSpecFileOperation, req *types.CreateVirtualDisk_Task) types.BaseMethodFault {
	fm := Map.FileManager()

	file, fault := fm.resolve(req.Datacenter, req.Name)
	if fault != nil {
		return fault
	}

	shouldReplace := op == types.VirtualDeviceConfigSpecFileOperationReplace
	shouldExist := op == ""
	for _, name := range m.names(file) {
		_, err := os.Stat(name)
		if err == nil {
			if shouldExist {
				return nil
			}
			if shouldReplace {
				if err = os.Truncate(file, 0); err != nil {
					return fm.fault(name, err, new(types.CannotCreateFile))
				}
				return nil
			}
			return fm.fault(name, nil, new(types.FileAlreadyExists))
		} else if shouldExist {
			return fm.fault(name, nil, new(types.FileNotFound))
		}

		f, err := os.Create(name)
		if err != nil {
			return fm.fault(name, err, new(types.CannotCreateFile))
		}

		_ = f.Close()
	}

	return nil
}

func (m *VirtualDiskManager) CreateVirtualDiskTask(req *types.CreateVirtualDisk_Task) soap.HasFault {
	task := CreateTask(m, "createVirtualDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		return nil, m.createVirtualDisk(types.VirtualDeviceConfigSpecFileOperationCreate, req)
	})

	return &methods.CreateVirtualDisk_TaskBody{
		Res: &types.CreateVirtualDisk_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VirtualDiskManager) DeleteVirtualDiskTask(req *types.DeleteVirtualDisk_Task) soap.HasFault {
	task := CreateTask(m, "deleteVirtualDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		fm := Map.FileManager()

		for _, name := range m.names(req.Name) {
			err := fm.deleteDatastoreFile(&types.DeleteDatastoreFile_Task{
				Name:       name,
				Datacenter: req.Datacenter,
			})

			if err != nil {
				return nil, err
			}
		}

		return nil, nil
	})

	return &methods.DeleteVirtualDisk_TaskBody{
		Res: &types.DeleteVirtualDisk_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VirtualDiskManager) MoveVirtualDiskTask(req *types.MoveVirtualDisk_Task) soap.HasFault {
	task := CreateTask(m, "moveVirtualDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		fm := Map.FileManager()

		dest := m.names(req.DestName)

		for i, name := range m.names(req.SourceName) {
			err := fm.moveDatastoreFile(&types.MoveDatastoreFile_Task{
				SourceName:            name,
				SourceDatacenter:      req.SourceDatacenter,
				DestinationName:       dest[i],
				DestinationDatacenter: req.DestDatacenter,
				Force: req.Force,
			})

			if err != nil {
				return nil, err
			}
		}

		return nil, nil
	})

	return &methods.MoveVirtualDisk_TaskBody{
		Res: &types.MoveVirtualDisk_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VirtualDiskManager) CopyVirtualDiskTask(req *types.CopyVirtualDisk_Task) soap.HasFault {
	task := CreateTask(m, "copyVirtualDisk", func(*Task) (types.AnyType, types.BaseMethodFault) {
		if req.DestSpec != nil {
			if Map.IsVPX() {
				return nil, new(types.NotImplemented)
			}
		}

		fm := Map.FileManager()

		dest := m.names(req.DestName)

		for i, name := range m.names(req.SourceName) {
			err := fm.copyDatastoreFile(&types.CopyDatastoreFile_Task{
				SourceName:            name,
				SourceDatacenter:      req.SourceDatacenter,
				DestinationName:       dest[i],
				DestinationDatacenter: req.DestDatacenter,
				Force: req.Force,
			})

			if err != nil {
				return nil, err
			}
		}

		return nil, nil
	})

	return &methods.CopyVirtualDisk_TaskBody{
		Res: &types.CopyVirtualDisk_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (m *VirtualDiskManager) QueryVirtualDiskUuid(req *types.QueryVirtualDiskUuid) soap.HasFault {
	body := new(methods.QueryVirtualDiskUuidBody)

	fm := Map.FileManager()

	file, fault := fm.resolve(req.Datacenter, req.Name)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	_, err := os.Stat(file)
	if err != nil {
		fault = fm.fault(file, err, new(types.CannotAccessFile))
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = &types.QueryVirtualDiskUuidResponse{
		Returnval: uuid.NewSHA1(uuid.NameSpaceOID, []byte(file)).String(),
	}

	return body
}

func (m *VirtualDiskManager) SetVirtualDiskUuid(req *types.SetVirtualDiskUuid) soap.HasFault {
	body := new(methods.SetVirtualDiskUuidBody)
	// TODO: validate uuid format and persist
	body.Res = new(types.SetVirtualDiskUuidResponse)
	return body
}
