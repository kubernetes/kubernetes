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
	"net/url"
	"path"
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ResourcePool struct {
	mo.ResourcePool
}

func asResourcePoolMO(obj mo.Reference) (*mo.ResourcePool, bool) {
	rp, ok := getManagedObject(obj).Addr().Interface().(*mo.ResourcePool)
	return rp, ok
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

func (a *VirtualApp) ImportVApp(ctx *Context, req *types.ImportVApp) soap.HasFault {
	return (&ResourcePool{ResourcePool: a.ResourcePool}).ImportVApp(ctx, req)
}

func (p *ResourcePool) ImportVApp(ctx *Context, req *types.ImportVApp) soap.HasFault {
	body := new(methods.ImportVAppBody)

	spec, ok := req.Spec.(*types.VirtualMachineImportSpec)
	if !ok {
		body.Fault_ = Fault(fmt.Sprintf("%T: type not supported", spec), &types.InvalidArgument{InvalidProperty: "spec"})
		return body
	}

	dc := ctx.Map.getEntityDatacenter(p)
	folder := ctx.Map.Get(dc.VmFolder).(*Folder)
	if req.Folder != nil {
		if p.Self.Type == "VirtualApp" {
			body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "pool"})
			return body
		}
		folder = ctx.Map.Get(*req.Folder).(*Folder)
	}

	res := folder.CreateVMTask(ctx, &types.CreateVM_Task{
		This:   folder.Self,
		Config: spec.ConfigSpec,
		Pool:   p.Self,
		Host:   req.Host,
	})

	ctask := ctx.Map.Get(res.(*methods.CreateVM_TaskBody).Res.Returnval).(*Task)
	ctask.Wait()

	if ctask.Info.Error != nil {
		body.Fault_ = Fault("", ctask.Info.Error.Fault)
		return body
	}

	lease := NewHttpNfcLease(ctx, ctask.Info.Result.(types.ManagedObjectReference))
	ref := lease.Reference()
	lease.Info.Lease = ref

	vm := ctx.Map.Get(lease.Info.Entity).(*VirtualMachine)
	device := object.VirtualDeviceList(vm.Config.Hardware.Device)
	ndevice := make(map[string]int)
	for _, d := range device {
		info, ok := d.GetVirtualDevice().Backing.(types.BaseVirtualDeviceFileBackingInfo)
		if !ok {
			continue
		}
		var file object.DatastorePath
		file.FromString(info.GetVirtualDeviceFileBackingInfo().FileName)
		name := path.Base(file.Path)
		ds := vm.findDatastore(file.Datastore)
		lease.files[name] = path.Join(ds.Info.GetDatastoreInfo().Url, file.Path)

		_, disk := d.(*types.VirtualDisk)
		kind := device.Type(d)
		n := ndevice[kind]
		ndevice[kind]++

		lease.Info.DeviceUrl = append(lease.Info.DeviceUrl, types.HttpNfcLeaseDeviceUrl{
			Key:       fmt.Sprintf("/%s/%s:%d", vm.Self.Value, kind, n),
			ImportKey: fmt.Sprintf("/%s/%s:%d", vm.Name, kind, n),
			Url: (&url.URL{
				Scheme: "https",
				Host:   "*",
				Path:   nfcPrefix + path.Join(ref.Value, name),
			}).String(),
			SslThumbprint: "",
			Disk:          types.NewBool(disk),
			TargetId:      name,
			DatastoreKey:  "",
			FileSize:      0,
		})
	}

	body.Res = &types.ImportVAppResponse{
		Returnval: ref,
	}

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
	body := &methods.CreateChildVM_TaskBody{}

	folder := ctx.Map.Get(*a.ParentFolder).(*Folder)

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

func (a *VirtualApp) CloneVAppTask(ctx *Context, req *types.CloneVApp_Task) soap.HasFault {
	task := CreateTask(a, "cloneVapp", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		folder := req.Spec.VmFolder
		if folder == nil {
			folder = a.ParentFolder
		}

		rspec := req.Spec.ResourceSpec
		if rspec == nil {
			s := types.DefaultResourceConfigSpec()
			rspec = &s
		}

		res := a.CreateVApp(&types.CreateVApp{
			This:       a.Self,
			Name:       req.Name,
			ResSpec:    *rspec,
			ConfigSpec: types.VAppConfigSpec{},
			VmFolder:   folder,
		})

		if res.Fault() != nil {
			return nil, res.Fault().VimFault().(types.BaseMethodFault)
		}

		target := res.(*methods.CreateVAppBody).Res.Returnval

		for _, ref := range a.Vm {
			vm := ctx.Map.Get(ref).(*VirtualMachine)

			res := vm.CloneVMTask(ctx, &types.CloneVM_Task{
				This:   ref,
				Folder: *folder,
				Name:   req.Name,
				Spec: types.VirtualMachineCloneSpec{
					Location: types.VirtualMachineRelocateSpec{
						Pool: &target,
						Host: req.Spec.Host,
					},
				},
			})

			ctask := ctx.Map.Get(res.(*methods.CloneVM_TaskBody).Res.Returnval).(*Task)
			ctask.Wait()
			if ctask.Info.Error != nil {
				return nil, ctask.Info.Error.Fault
			}
		}

		return target, nil
	})

	return &methods.CloneVApp_TaskBody{
		Res: &types.CloneVApp_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (a *VirtualApp) CreateVApp(req *types.CreateVApp) soap.HasFault {
	return (&ResourcePool{ResourcePool: a.ResourcePool}).CreateVApp(req)
}

func (a *VirtualApp) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	return (&ResourcePool{ResourcePool: a.ResourcePool}).DestroyTask(ctx, req)
}

func (p *ResourcePool) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(p, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if strings.HasSuffix(p.Parent.Type, "ComputeResource") {
			// Can't destroy the root pool
			return nil, &types.InvalidArgument{}
		}

		parent, _ := asResourcePoolMO(ctx.Map.Get(*p.Parent))

		// Remove child reference from rp
		ctx.WithLock(parent, func() {
			RemoveReference(&parent.ResourcePool, req.This)

			// The grandchildren become children of the parent (rp)
			parent.ResourcePool = append(parent.ResourcePool, p.ResourcePool.ResourcePool...)
		})

		// And VMs move to the parent
		vms := p.ResourcePool.Vm
		for _, ref := range vms {
			vm := ctx.Map.Get(ref).(*VirtualMachine)
			ctx.WithLock(vm, func() { vm.ResourcePool = &parent.Self })
		}

		ctx.WithLock(parent, func() {
			parent.Vm = append(parent.Vm, vms...)
		})

		ctx.Map.Remove(ctx, req.This)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (p *ResourcePool) DestroyChildren(ctx *Context, req *types.DestroyChildren) soap.HasFault {
	walk(p, func(child types.ManagedObjectReference) {
		if child.Type != "ResourcePool" {
			return
		}
		ctx.Map.Get(child).(*ResourcePool).DestroyTask(ctx, &types.Destroy_Task{This: child})
	})

	return &methods.DestroyChildrenBody{Res: new(types.DestroyChildrenResponse)}
}
