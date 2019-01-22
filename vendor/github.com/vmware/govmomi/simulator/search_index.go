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
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type SearchIndex struct {
	mo.SearchIndex
}

func NewSearchIndex(ref types.ManagedObjectReference) object.Reference {
	m := &SearchIndex{}
	m.Self = ref
	return m
}

func (s *SearchIndex) FindByDatastorePath(r *types.FindByDatastorePath) soap.HasFault {
	res := &methods.FindByDatastorePathBody{Res: new(types.FindByDatastorePathResponse)}

	for ref, obj := range Map.objects {
		vm, ok := obj.(*VirtualMachine)
		if !ok {
			continue
		}

		if vm.Config.Files.VmPathName == r.Path {
			res.Res.Returnval = &ref
			break
		}
	}

	return res
}

func (s *SearchIndex) FindByInventoryPath(req *types.FindByInventoryPath) soap.HasFault {
	body := &methods.FindByInventoryPathBody{Res: new(types.FindByInventoryPathResponse)}

	split := func(c rune) bool {
		return c == '/'
	}
	path := strings.FieldsFunc(req.InventoryPath, split)
	if len(path) < 1 {
		return body
	}

	root := Map.content().RootFolder
	o := &root

	for _, name := range path {
		f := s.FindChild(&types.FindChild{Entity: *o, Name: name})

		o = f.(*methods.FindChildBody).Res.Returnval
		if o == nil {
			break
		}
	}

	body.Res.Returnval = o

	return body
}

func (s *SearchIndex) FindChild(req *types.FindChild) soap.HasFault {
	body := &methods.FindChildBody{}

	obj := Map.Get(req.Entity)

	if obj == nil {
		body.Fault_ = Fault("", &types.ManagedObjectNotFound{Obj: req.Entity})
		return body
	}

	body.Res = new(types.FindChildResponse)

	var children []types.ManagedObjectReference

	switch e := obj.(type) {
	case *Datacenter:
		children = []types.ManagedObjectReference{e.VmFolder, e.HostFolder, e.DatastoreFolder, e.NetworkFolder}
	case *Folder:
		children = e.ChildEntity
	case *mo.ComputeResource:
		children = e.Host
		children = append(children, *e.ResourcePool)
	case *ClusterComputeResource:
		children = e.Host
		children = append(children, *e.ResourcePool)
	case *ResourcePool:
		children = e.ResourcePool.ResourcePool
		children = append(children, e.Vm...)
	case *VirtualApp:
		children = e.ResourcePool.ResourcePool
		children = append(children, e.Vm...)
	}

	match := Map.FindByName(req.Name, children)

	if match != nil {
		ref := match.Reference()
		body.Res.Returnval = &ref
	}

	return body
}

func (s *SearchIndex) FindByUuid(req *types.FindByUuid) soap.HasFault {
	body := &methods.FindByUuidBody{Res: new(types.FindByUuidResponse)}

	if req.VmSearch {
		// Find Virtual Machine using UUID
		for ref, obj := range Map.objects {
			vm, ok := obj.(*VirtualMachine)
			if !ok {
				continue
			}
			if req.InstanceUuid != nil && *req.InstanceUuid {
				if vm.Config.InstanceUuid == req.Uuid {
					body.Res.Returnval = &ref
					break
				}
			} else {
				if vm.Config.Uuid == req.Uuid {
					body.Res.Returnval = &ref
					break
				}
			}
		}
	} else {
		// Find Host System using UUID
		for ref, obj := range Map.objects {
			host, ok := obj.(*HostSystem)
			if !ok {
				continue
			}
			if host.Summary.Hardware.Uuid == req.Uuid {
				body.Res.Returnval = &ref
				break
			}
		}
	}

	return body
}
