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
	"math/rand"
	"path"
	"strings"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type Folder struct {
	mo.Folder
}

func (f *Folder) eventArgument() types.FolderEventArgument {
	return types.FolderEventArgument{
		Folder:              f.Self,
		EntityEventArgument: types.EntityEventArgument{Name: f.Name},
	}
}

// update references when objects are added/removed from a Folder
func (f *Folder) update(o mo.Reference, u func(mo.Reference, *[]types.ManagedObjectReference, types.ManagedObjectReference)) {
	ref := o.Reference()

	if f.Parent == nil {
		return // this is the root folder
	}

	switch ref.Type {
	case "Datacenter", "Folder":
		return // nothing to update
	}

	dc := Map.getEntityDatacenter(f)

	switch ref.Type {
	case "Network", "DistributedVirtualSwitch", "DistributedVirtualPortgroup":
		u(dc, &dc.Network, ref)
	case "Datastore":
		u(dc, &dc.Datastore, ref)
	}
}

func networkSummary(n *mo.Network) *types.NetworkSummary {
	return &types.NetworkSummary{
		Network:    &n.Self,
		Name:       n.Name,
		Accessible: true,
	}
}

func (f *Folder) putChild(o mo.Entity) {
	Map.PutEntity(f, o)

	f.ChildEntity = append(f.ChildEntity, o.Reference())

	f.update(o, Map.AddReference)

	switch e := o.(type) {
	case *mo.Network:
		e.Summary = networkSummary(e)
	case *mo.OpaqueNetwork:
		e.Summary = networkSummary(&e.Network)
	case *DistributedVirtualPortgroup:
		e.Summary = networkSummary(&e.Network)
	}
}

func (f *Folder) removeChild(o mo.Reference) {
	Map.Remove(o.Reference())

	RemoveReference(&f.ChildEntity, o.Reference())

	f.update(o, Map.RemoveReference)
}

func (f *Folder) hasChildType(kind string) bool {
	for _, t := range f.ChildType {
		if t == kind {
			return true
		}
	}
	return false
}

func (f *Folder) typeNotSupported() *soap.Fault {
	return Fault(fmt.Sprintf("%s supports types: %#v", f.Self, f.ChildType), &types.NotSupported{})
}

type addStandaloneHost struct {
	*Folder

	req *types.AddStandaloneHost_Task
}

func (add *addStandaloneHost) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	host, err := CreateStandaloneHost(add.Folder, add.req.Spec)
	if err != nil {
		return nil, err
	}

	if add.req.AddConnected {
		host.Runtime.ConnectionState = types.HostSystemConnectionStateConnected
	}

	return host.Reference(), nil
}

func (f *Folder) AddStandaloneHostTask(a *types.AddStandaloneHost_Task) soap.HasFault {
	r := &methods.AddStandaloneHost_TaskBody{}

	if f.hasChildType("ComputeResource") && f.hasChildType("Folder") {
		r.Res = &types.AddStandaloneHost_TaskResponse{
			Returnval: NewTask(&addStandaloneHost{f, a}).Run(),
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (f *Folder) CreateFolder(c *types.CreateFolder) soap.HasFault {
	r := &methods.CreateFolderBody{}

	if f.hasChildType("Folder") {
		folder := &Folder{}

		folder.Name = c.Name
		folder.ChildType = f.ChildType

		f.putChild(folder)

		r.Res = &types.CreateFolderResponse{
			Returnval: folder.Self,
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

// StoragePod aka "Datastore Cluster"
type StoragePod struct {
	mo.StoragePod
}

func (f *Folder) CreateStoragePod(c *types.CreateStoragePod) soap.HasFault {
	r := &methods.CreateStoragePodBody{}

	if f.hasChildType("StoragePod") {
		pod := &StoragePod{}

		pod.Name = c.Name
		pod.ChildType = []string{"Datastore"}

		f.putChild(pod)

		r.Res = &types.CreateStoragePodResponse{
			Returnval: pod.Self,
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (p *StoragePod) MoveIntoFolderTask(c *types.MoveIntoFolder_Task) soap.HasFault {
	return (&Folder{Folder: p.Folder}).MoveIntoFolderTask(c)
}

func (f *Folder) CreateDatacenter(ctx *Context, c *types.CreateDatacenter) soap.HasFault {
	r := &methods.CreateDatacenterBody{}

	if f.hasChildType("Datacenter") && f.hasChildType("Folder") {
		dc := NewDatacenter(f)

		dc.Name = c.Name

		r.Res = &types.CreateDatacenterResponse{
			Returnval: dc.Self,
		}

		ctx.postEvent(&types.DatacenterCreatedEvent{
			DatacenterEvent: types.DatacenterEvent{
				Event: types.Event{
					Datacenter: datacenterEventArgument(dc),
				},
			},
			Parent: f.eventArgument(),
		})
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (f *Folder) CreateClusterEx(c *types.CreateClusterEx) soap.HasFault {
	r := &methods.CreateClusterExBody{}

	if f.hasChildType("ComputeResource") && f.hasChildType("Folder") {
		cluster, err := CreateClusterComputeResource(f, c.Name, c.Spec)
		if err != nil {
			r.Fault_ = Fault("", err)
			return r
		}

		r.Res = &types.CreateClusterExResponse{
			Returnval: cluster.Self,
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

type createVM struct {
	*Folder

	ctx *Context
	req *types.CreateVM_Task

	register bool
}

func (c *createVM) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	vm, err := NewVirtualMachine(c.Folder.Self, &c.req.Config)
	if err != nil {
		return nil, err
	}

	vm.ResourcePool = &c.req.Pool

	if c.req.Host == nil {
		var hosts []types.ManagedObjectReference

		pool := Map.Get(c.req.Pool).(mo.Entity)

		switch cr := Map.getEntityComputeResource(pool).(type) {
		case *mo.ComputeResource:
			hosts = cr.Host
		case *ClusterComputeResource:
			hosts = cr.Host
		}

		// Assuming for now that all hosts have access to the datastore
		host := hosts[rand.Intn(len(hosts))]
		vm.Runtime.Host = &host
	} else {
		vm.Runtime.Host = c.req.Host
	}

	vm.Guest = &types.GuestInfo{
		ToolsStatus:  types.VirtualMachineToolsStatusToolsNotInstalled,
		ToolsVersion: "0",
	}

	vm.Summary.Guest = &types.VirtualMachineGuestSummary{
		ToolsStatus: vm.Guest.ToolsStatus,
	}
	vm.Summary.Config.VmPathName = vm.Config.Files.VmPathName
	vm.Summary.Runtime.Host = vm.Runtime.Host

	err = vm.create(&c.req.Config, c.register)
	if err != nil {
		return nil, err
	}

	c.Folder.putChild(vm)

	host := Map.Get(*vm.Runtime.Host).(*HostSystem)
	Map.AppendReference(host, &host.Vm, vm.Self)

	for i := range vm.Datastore {
		ds := Map.Get(vm.Datastore[i]).(*Datastore)
		Map.AppendReference(ds, &ds.Vm, vm.Self)
	}

	pool := Map.Get(*vm.ResourcePool)
	// This can be an internal call from VirtualApp.CreateChildVMTask, where pool is already locked.
	c.ctx.WithLock(pool, func() {
		switch rp := pool.(type) {
		case *ResourcePool:
			rp.Vm = append(rp.Vm, vm.Self)
		case *VirtualApp:
			rp.Vm = append(rp.Vm, vm.Self)
		}
	})

	event := vm.event()
	c.ctx.postEvent(
		&types.VmBeingCreatedEvent{
			VmEvent:    event,
			ConfigSpec: &c.req.Config,
		},
		&types.VmInstanceUuidAssignedEvent{
			VmEvent:      event,
			InstanceUuid: vm.Config.InstanceUuid,
		},
		&types.VmUuidAssignedEvent{
			VmEvent: event,
			Uuid:    vm.Config.Uuid,
		},
		&types.VmCreatedEvent{
			VmEvent: event,
		},
	)

	return vm.Reference(), nil
}

func (f *Folder) CreateVMTask(ctx *Context, c *types.CreateVM_Task) soap.HasFault {
	return &methods.CreateVM_TaskBody{
		Res: &types.CreateVM_TaskResponse{
			Returnval: NewTask(&createVM{f, ctx, c, false}).Run(),
		},
	}
}

type registerVM struct {
	*Folder

	ctx *Context
	req *types.RegisterVM_Task
}

func (c *registerVM) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	host := c.req.Host
	pool := c.req.Pool

	if c.req.AsTemplate {
		if host == nil {
			return nil, &types.InvalidArgument{InvalidProperty: "host"}
		} else if pool != nil {
			return nil, &types.InvalidArgument{InvalidProperty: "pool"}
		}

		pool = hostParent(&Map.Get(*host).(*HostSystem).HostSystem).ResourcePool
	} else {
		if pool == nil {
			return nil, &types.InvalidArgument{InvalidProperty: "pool"}
		}
	}

	if c.req.Path == "" {
		return nil, &types.InvalidArgument{InvalidProperty: "path"}
	}

	s := Map.SearchIndex()
	r := s.FindByDatastorePath(&types.FindByDatastorePath{
		This:       s.Reference(),
		Path:       c.req.Path,
		Datacenter: Map.getEntityDatacenter(c.Folder).Reference(),
	})

	if ref := r.(*methods.FindByDatastorePathBody).Res.Returnval; ref != nil {
		return nil, &types.AlreadyExists{Name: ref.Value}
	}

	if c.req.Name == "" {
		p, err := parseDatastorePath(c.req.Path)
		if err != nil {
			return nil, err
		}

		c.req.Name = path.Dir(p.Path)
	}

	create := NewTask(&createVM{
		Folder:   c.Folder,
		register: true,
		ctx:      c.ctx,
		req: &types.CreateVM_Task{
			This: c.Folder.Reference(),
			Config: types.VirtualMachineConfigSpec{
				Name: c.req.Name,
				Files: &types.VirtualMachineFileInfo{
					VmPathName: c.req.Path,
				},
			},
			Pool: *pool,
			Host: host,
		},
	})

	create.Run()

	if create.Info.Error != nil {
		return nil, create.Info.Error.Fault
	}

	return create.Info.Result, nil
}

func (f *Folder) RegisterVMTask(ctx *Context, c *types.RegisterVM_Task) soap.HasFault {
	ctx.Caller = &f.Self

	return &methods.RegisterVM_TaskBody{
		Res: &types.RegisterVM_TaskResponse{
			Returnval: NewTask(&registerVM{f, ctx, c}).Run(),
		},
	}
}

func (f *Folder) MoveIntoFolderTask(c *types.MoveIntoFolder_Task) soap.HasFault {
	task := CreateTask(f, "moveIntoFolder", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		for _, ref := range c.List {
			obj := Map.Get(ref).(mo.Entity)

			parent, ok := Map.Get(*(obj.Entity()).Parent).(*Folder)

			if !ok || !f.hasChildType(ref.Type) {
				return nil, &types.NotSupported{}
			}

			parent.removeChild(ref)
			f.putChild(obj)
		}

		return nil, nil
	})

	return &methods.MoveIntoFolder_TaskBody{
		Res: &types.MoveIntoFolder_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (f *Folder) CreateDVSTask(req *types.CreateDVS_Task) soap.HasFault {
	task := CreateTask(f, "createDVS", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		spec := req.Spec.ConfigSpec.GetDVSConfigSpec()
		dvs := &DistributedVirtualSwitch{}
		dvs.Name = spec.Name
		dvs.Entity().Name = dvs.Name

		if Map.FindByName(dvs.Name, f.ChildEntity) != nil {
			return nil, &types.InvalidArgument{InvalidProperty: "name"}
		}

		dvs.Uuid = uuid.New().String()

		f.putChild(dvs)

		dvs.Summary = types.DVSSummary{
			Name:        dvs.Name,
			Uuid:        dvs.Uuid,
			NumPorts:    spec.NumStandalonePorts,
			ProductInfo: req.Spec.ProductInfo,
			Description: spec.Description,
		}

		configInfo := &types.VMwareDVSConfigInfo{
			DVSConfigInfo: types.DVSConfigInfo{
				Uuid:                                dvs.Uuid,
				Name:                                spec.Name,
				ConfigVersion:                       spec.ConfigVersion,
				NumStandalonePorts:                  spec.NumStandalonePorts,
				MaxPorts:                            spec.MaxPorts,
				UplinkPortPolicy:                    spec.UplinkPortPolicy,
				UplinkPortgroup:                     spec.UplinkPortgroup,
				DefaultPortConfig:                   spec.DefaultPortConfig,
				ExtensionKey:                        spec.ExtensionKey,
				Description:                         spec.Description,
				Policy:                              spec.Policy,
				VendorSpecificConfig:                spec.VendorSpecificConfig,
				SwitchIpAddress:                     spec.SwitchIpAddress,
				DefaultProxySwitchMaxNumPorts:       spec.DefaultProxySwitchMaxNumPorts,
				InfrastructureTrafficResourceConfig: spec.InfrastructureTrafficResourceConfig,
				NetworkResourceControlVersion:       spec.NetworkResourceControlVersion,
			},
		}

		if spec.Contact != nil {
			configInfo.Contact = *spec.Contact
		}

		dvs.Config = configInfo

		if dvs.Summary.ProductInfo == nil {
			product := Map.content().About
			dvs.Summary.ProductInfo = &types.DistributedVirtualSwitchProductSpec{
				Name:            "DVS",
				Vendor:          product.Vendor,
				Version:         product.Version,
				Build:           product.Build,
				ForwardingClass: "etherswitch",
			}
		}

		dvs.AddDVPortgroupTask(&types.AddDVPortgroup_Task{
			Spec: []types.DVPortgroupConfigSpec{{
				Name: dvs.Name + "-DVUplinks" + strings.TrimPrefix(dvs.Self.Value, "dvs"),
				DefaultPortConfig: &types.VMwareDVSPortSetting{
					Vlan: &types.VmwareDistributedVirtualSwitchTrunkVlanSpec{
						VlanId: []types.NumericRange{{Start: 0, End: 4094}},
					},
				},
			}},
		})

		return dvs.Reference(), nil
	})

	return &methods.CreateDVS_TaskBody{
		Res: &types.CreateDVS_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (f *Folder) RenameTask(r *types.Rename_Task) soap.HasFault {
	return RenameTask(f, r)
}

func (f *Folder) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	type destroyer interface {
		mo.Reference
		DestroyTask(*types.Destroy_Task) soap.HasFault
	}

	task := CreateTask(f, "destroy", func(*Task) (types.AnyType, types.BaseMethodFault) {
		// Attempt to destroy all children
		for _, c := range f.ChildEntity {
			obj, ok := Map.Get(c).(destroyer)
			if !ok {
				continue
			}

			var fault types.BaseMethodFault
			Map.WithLock(obj, func() {
				id := obj.DestroyTask(&types.Destroy_Task{
					This: c,
				}).(*methods.Destroy_TaskBody).Res.Returnval

				t := Map.Get(id).(*Task)
				if t.Info.Error != nil {
					fault = t.Info.Error.Fault // For example, can't destroy a powered on VM
				}
			})
			if fault != nil {
				return nil, fault
			}
		}

		// Remove the folder itself
		Map.Get(*f.Parent).(*Folder).removeChild(f.Self)
		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
