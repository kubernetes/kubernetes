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
	"fmt"
	"strings"

	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ResourcePool struct {
	mo.ResourcePool
}

func NewResourcePool() *ResourcePool {
	pool := &ResourcePool{
		ResourcePool: esx.ResourcePool,
	}

	if Map.IsVPX() {
		pool.DisabledMethod = nil // Enable VApp methods for VC
	}

	return pool
}

func allResourceFieldsSet(info *types.ResourceAllocationInfo) bool {
	return info.Reservation != nil &&
		info.Limit != nil &&
		info.ExpandableReservation != nil &&
		info.Shares != nil
}

func allResourceFieldsValid(info *types.ResourceAllocationInfo) bool {
	if info.Reservation != nil {
		if *info.Reservation < 0 {
			return false
		}
	}

	if info.Limit != nil {
		if *info.Limit < -1 {
			return false
		}
	}

	if info.Shares != nil {
		if info.Shares.Level == types.SharesLevelCustom {
			if info.Shares.Shares < 0 {
				return false
			}
		}
	}

	if info.OverheadLimit != nil {
		return false
	}

	return true
}

func (p *ResourcePool) createChild(name string, spec types.ResourceConfigSpec) (*ResourcePool, *soap.Fault) {
	if e := Map.FindByName(name, p.ResourcePool.ResourcePool); e != nil {
		return nil, Fault("", &types.DuplicateName{
			Name:   e.Entity().Name,
			Object: e.Reference(),
		})
	}

	if !(allResourceFieldsSet(&spec.CpuAllocation) && allResourceFieldsValid(&spec.CpuAllocation)) {
		return nil, Fault("", &types.InvalidArgument{
			InvalidProperty: "spec.cpuAllocation",
		})
	}

	if !(allResourceFieldsSet(&spec.MemoryAllocation) && allResourceFieldsValid(&spec.MemoryAllocation)) {
		return nil, Fault("", &types.InvalidArgument{
			InvalidProperty: "spec.memoryAllocation",
		})
	}

	child := NewResourcePool()

	child.Name = name
	child.Owner = p.Owner
	child.Summary.GetResourcePoolSummary().Name = name
	child.Config.CpuAllocation = spec.CpuAllocation
	child.Config.MemoryAllocation = spec.MemoryAllocation
	child.Config.Entity = spec.Entity

	return child, nil
}

func (p *ResourcePool) CreateResourcePool(c *types.CreateResourcePool) soap.HasFault {
	body := &methods.CreateResourcePoolBody{}

	child, err := p.createChild(c.Name, c.Spec)
	if err != nil {
		body.Fault_ = err
		return body
	}

	Map.PutEntity(p, Map.NewEntity(child))

	p.ResourcePool.ResourcePool = append(p.ResourcePool.ResourcePool, child.Reference())

	body.Res = &types.CreateResourcePoolResponse{
		Returnval: child.Reference(),
	}

	return body
}

func updateResourceAllocation(kind string, src, dst *types.ResourceAllocationInfo) types.BaseMethodFault {
	if !allResourceFieldsValid(src) {
		return &types.InvalidArgument{
			InvalidProperty: fmt.Sprintf("spec.%sAllocation", kind),
		}
	}

	if src.Reservation != nil {
		dst.Reservation = src.Reservation
	}

	if src.Limit != nil {
		dst.Limit = src.Limit
	}

	if src.Shares != nil {
		dst.Shares = src.Shares
	}

	return nil
}

func (p *ResourcePool) UpdateConfig(c *types.UpdateConfig) soap.HasFault {
	body := &methods.UpdateConfigBody{}

	if c.Name != "" {
		if e := Map.FindByName(c.Name, p.ResourcePool.ResourcePool); e != nil {
			body.Fault_ = Fault("", &types.DuplicateName{
				Name:   e.Entity().Name,
				Object: e.Reference(),
			})
			return body
		}

		p.Name = c.Name
	}

	spec := c.Config

	if spec != nil {
		if err := updateResourceAllocation("memory", &spec.MemoryAllocation, &p.Config.MemoryAllocation); err != nil {
			body.Fault_ = Fault("", err)
			return body
		}

		if err := updateResourceAllocation("cpu", &spec.CpuAllocation, &p.Config.CpuAllocation); err != nil {
			body.Fault_ = Fault("", err)
			return body
		}
	}

	body.Res = &types.UpdateConfigResponse{}

	return body
}

type VirtualApp struct {
	mo.VirtualApp
}

func NewVAppConfigSpec() types.VAppConfigSpec {
	spec := types.VAppConfigSpec{
		Annotation: "vcsim",
		VmConfigSpec: types.VmConfigSpec{
			Product: []types.VAppProductSpec{
				{
					Info: &types.VAppProductInfo{
						Name:      "vcsim",
						Vendor:    "VMware",
						VendorUrl: "http://www.vmware.com/",
						Version:   "0.1",
					},
					ArrayUpdateSpec: types.ArrayUpdateSpec{
						Operation: types.ArrayUpdateOperationAdd,
					},
				},
			},
		},
	}

	return spec
}

func (p *ResourcePool) CreateVApp(req *types.CreateVApp) soap.HasFault {
	body := &methods.CreateVAppBody{}

	pool, err := p.createChild(req.Name, req.ResSpec)
	if err != nil {
		body.Fault_ = err
		return body
	}

	child := &VirtualApp{}
	child.ResourcePool = pool.ResourcePool
	child.Self.Type = "VirtualApp"
	child.ParentFolder = req.VmFolder

	if child.ParentFolder == nil {
		folder := Map.getEntityDatacenter(p).VmFolder
		child.ParentFolder = &folder
	}

	child.VAppConfig = &types.VAppConfigInfo{
		VmConfigInfo: types.VmConfigInfo{},
		Annotation:   req.ConfigSpec.Annotation,
	}

	for _, product := range req.ConfigSpec.Product {
		child.VAppConfig.Product = append(child.VAppConfig.Product, *product.Info)
	}

	Map.PutEntity(p, Map.NewEntity(child))

	p.ResourcePool.ResourcePool = append(p.ResourcePool.ResourcePool, child.Reference())

	body.Res = &types.CreateVAppResponse{
		Returnval: child.Reference(),
	}

	return body
}

func (a *VirtualApp) CreateChildVMTask(ctx *Context, req *types.CreateChildVM_Task) soap.HasFault {
	ctx.Caller = &a.Self
	body := &methods.CreateChildVM_TaskBody{}

	folder := Map.Get(*a.ParentFolder).(*Folder)

	res := folder.CreateVMTask(ctx, &types.CreateVM_Task{
		This:   folder.Self,
		Config: req.Config,
		Host:   req.Host,
		Pool:   req.This,
	})

	body.Res = &types.CreateChildVM_TaskResponse{
		Returnval: res.(*methods.CreateVM_TaskBody).Res.Returnval,
	}

	return body
}

func (a *VirtualApp) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	return (&ResourcePool{ResourcePool: a.ResourcePool}).DestroyTask(req)
}

func (p *ResourcePool) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(p, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if strings.HasSuffix(p.Parent.Type, "ComputeResource") {
			// Can't destroy the root pool
			return nil, &types.InvalidArgument{}
		}

		pp := Map.Get(*p.Parent).(*ResourcePool)

		parent := &pp.ResourcePool
		// Remove child reference from rp
		Map.RemoveReference(parent, &parent.ResourcePool, req.This)

		// The grandchildren become children of the parent (rp)
		Map.AppendReference(parent, &parent.ResourcePool, p.ResourcePool.ResourcePool...)

		// And VMs move to the parent
		vms := p.ResourcePool.Vm
		for _, ref := range vms {
			vm := Map.Get(ref).(*VirtualMachine)
			Map.WithLock(vm, func() { vm.ResourcePool = &parent.Self })
		}

		Map.AppendReference(parent, &parent.Vm, vms...)

		Map.Remove(req.This)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
