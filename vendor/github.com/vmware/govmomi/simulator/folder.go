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
	"errors"
	"fmt"
	"math/rand"
	"path"
	"strings"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type Folder struct {
	mo.Folder
}

func asFolderMO(obj mo.Reference) (*mo.Folder, bool) {
	if obj == nil {
		return nil, false
	}
	f, ok := getManagedObject(obj).Addr().Interface().(*mo.Folder)
	return f, ok
}

func folderEventArgument(f *mo.Folder) types.FolderEventArgument {
	return types.FolderEventArgument{
		Folder:              f.Self,
		EntityEventArgument: types.EntityEventArgument{Name: f.Name},
	}
}

// update references when objects are added/removed from a Folder
func folderUpdate(ctx *Context, f *mo.Folder, o mo.Reference, u func(*Context, mo.Reference, *[]types.ManagedObjectReference, types.ManagedObjectReference)) {
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
		u(ctx, dc, &dc.Network, ref)
	case "Datastore":
		u(ctx, dc, &dc.Datastore, ref)
	}
}

func networkSummary(n *mo.Network) types.BaseNetworkSummary {
	if n.Summary != nil {
		return n.Summary
	}
	return &types.NetworkSummary{
		Network:    &n.Self,
		Name:       n.Name,
		Accessible: true,
	}
}

func folderPutChild(ctx *Context, f *mo.Folder, o mo.Entity) {
	ctx.WithLock(f, func() {
		// Need to update ChildEntity before Map.Put for ContainerView updates to work properly
		f.ChildEntity = append(f.ChildEntity, Map.reference(o))
		Map.PutEntity(f, o)

		folderUpdate(ctx, f, o, Map.AddReference)

		ctx.WithLock(o, func() {
			switch e := o.(type) {
			case *mo.Network:
				e.Summary = networkSummary(e)
			case *mo.OpaqueNetwork:
				e.Summary = networkSummary(&e.Network)
			case *DistributedVirtualPortgroup:
				e.Summary = networkSummary(&e.Network)
			}
		})
	})
}

func folderRemoveChild(ctx *Context, f *mo.Folder, o mo.Reference) {
	Map.Remove(ctx, o.Reference())

	ctx.WithLock(f, func() {
		RemoveReference(&f.ChildEntity, o.Reference())

		folderUpdate(ctx, f, o, Map.RemoveReference)
	})
}

func folderHasChildType(f *mo.Folder, kind string) bool {
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

// AddOpaqueNetwork adds an OpaqueNetwork type to the inventory, with default backing to that of an nsx.LogicalSwitch.
// The vSphere API does not have a method to add this directly, so it must either be called directly or via Model.OpaqueNetwork setting.
func (f *Folder) AddOpaqueNetwork(ctx *Context, summary types.OpaqueNetworkSummary) error {
	if !folderHasChildType(&f.Folder, "Network") {
		return errors.New("not a network folder")
	}

	if summary.OpaqueNetworkId == "" {
		summary.OpaqueNetworkId = uuid.New().String()
	}
	if summary.OpaqueNetworkType == "" {
		summary.OpaqueNetworkType = "nsx.LogicalSwitch"
	}
	if summary.Name == "" {
		summary.Name = summary.OpaqueNetworkType + "-" + summary.OpaqueNetworkId
	}

	net := new(mo.OpaqueNetwork)
	if summary.Network == nil {
		summary.Network = &net.Self
	} else {
		net.Self = *summary.Network
	}
	summary.Accessible = true
	net.Network.Name = summary.Name
	net.Summary = &summary

	folderPutChild(ctx, &f.Folder, net)

	return nil
}

type addStandaloneHost struct {
	*Folder
	ctx *Context
	req *types.AddStandaloneHost_Task
}

func (add *addStandaloneHost) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	host, err := CreateStandaloneHost(add.ctx, add.Folder, add.req.Spec)
	if err != nil {
		return nil, err
	}

	if add.req.AddConnected {
		host.Runtime.ConnectionState = types.HostSystemConnectionStateConnected
	}

	return host.Reference(), nil
}

func (f *Folder) AddStandaloneHostTask(ctx *Context, a *types.AddStandaloneHost_Task) soap.HasFault {
	r := &methods.AddStandaloneHost_TaskBody{}

	if folderHasChildType(&f.Folder, "ComputeResource") && folderHasChildType(&f.Folder, "Folder") {
		r.Res = &types.AddStandaloneHost_TaskResponse{
			Returnval: NewTask(&addStandaloneHost{f, ctx, a}).Run(ctx),
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (f *Folder) CreateFolder(ctx *Context, c *types.CreateFolder) soap.HasFault {
	r := &methods.CreateFolderBody{}

	if folderHasChildType(&f.Folder, "Folder") {
		if obj := Map.FindByName(c.Name, f.ChildEntity); obj != nil {
			r.Fault_ = Fault("", &types.DuplicateName{
				Name:   c.Name,
				Object: f.Self,
			})

			return r
		}

		folder := &Folder{}

		folder.Name = c.Name
		folder.ChildType = f.ChildType

		folderPutChild(ctx, &f.Folder, folder)

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

func (f *Folder) CreateStoragePod(ctx *Context, c *types.CreateStoragePod) soap.HasFault {
	r := &methods.CreateStoragePodBody{}

	if folderHasChildType(&f.Folder, "StoragePod") {
		if obj := Map.FindByName(c.Name, f.ChildEntity); obj != nil {
			r.Fault_ = Fault("", &types.DuplicateName{
				Name:   c.Name,
				Object: f.Self,
			})

			return r
		}

		pod := &StoragePod{}

		pod.Name = c.Name
		pod.ChildType = []string{"Datastore"}
		pod.Summary = new(types.StoragePodSummary)
		pod.PodStorageDrsEntry = new(types.PodStorageDrsEntry)
		pod.PodStorageDrsEntry.StorageDrsConfig.PodConfig.Enabled = true

		folderPutChild(ctx, &f.Folder, pod)

		r.Res = &types.CreateStoragePodResponse{
			Returnval: pod.Self,
		}
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (p *StoragePod) MoveIntoFolderTask(ctx *Context, c *types.MoveIntoFolder_Task) soap.HasFault {
	task := CreateTask(p, "moveIntoFolder", func(*Task) (types.AnyType, types.BaseMethodFault) {
		f := &Folder{Folder: p.Folder}
		id := f.MoveIntoFolderTask(ctx, c).(*methods.MoveIntoFolder_TaskBody).Res.Returnval
		ftask := Map.Get(id).(*Task)
		ftask.Wait()
		if ftask.Info.Error != nil {
			return nil, ftask.Info.Error.Fault
		}
		p.ChildEntity = append(p.ChildEntity, f.ChildEntity...)
		return nil, nil
	})
	return &methods.MoveIntoFolder_TaskBody{
		Res: &types.MoveIntoFolder_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (f *Folder) CreateDatacenter(ctx *Context, c *types.CreateDatacenter) soap.HasFault {
	r := &methods.CreateDatacenterBody{}

	if folderHasChildType(&f.Folder, "Datacenter") && folderHasChildType(&f.Folder, "Folder") {
		dc := NewDatacenter(ctx, &f.Folder)

		Map.Update(dc, []types.PropertyChange{
			{Name: "name", Val: c.Name},
		})

		r.Res = &types.CreateDatacenterResponse{
			Returnval: dc.Self,
		}

		ctx.postEvent(&types.DatacenterCreatedEvent{
			DatacenterEvent: types.DatacenterEvent{
				Event: types.Event{
					Datacenter: datacenterEventArgument(dc),
				},
			},
			Parent: folderEventArgument(&f.Folder),
		})
	} else {
		r.Fault_ = f.typeNotSupported()
	}

	return r
}

func (f *Folder) CreateClusterEx(ctx *Context, c *types.CreateClusterEx) soap.HasFault {
	r := &methods.CreateClusterExBody{}

	if folderHasChildType(&f.Folder, "ComputeResource") && folderHasChildType(&f.Folder, "Folder") {
		cluster, err := CreateClusterComputeResource(ctx, f, c.Name, c.Spec)
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

// hostsWithDatastore returns hosts that have access to the given datastore path
func hostsWithDatastore(hosts []types.ManagedObjectReference, path string) []types.ManagedObjectReference {
	attached := hosts[:0]
	var p object.DatastorePath
	p.FromString(path)

	for _, host := range hosts {
		h := Map.Get(host).(*HostSystem)
		if Map.FindByName(p.Datastore, h.Datastore) != nil {
			attached = append(attached, host)
		}
	}

	return attached
}

func (c *createVM) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	vm, err := NewVirtualMachine(c.ctx, c.Folder.Self, &c.req.Config)
	if err != nil {
		folderRemoveChild(c.ctx, &c.Folder.Folder, vm)
		return nil, err
	}

	vm.ResourcePool = &c.req.Pool

	if c.req.Host == nil {
		pool := Map.Get(c.req.Pool).(mo.Entity)
		cr := Map.getEntityComputeResource(pool)

		c.ctx.WithLock(cr, func() {
			var hosts []types.ManagedObjectReference
			switch cr := cr.(type) {
			case *mo.ComputeResource:
				hosts = cr.Host
			case *ClusterComputeResource:
				hosts = cr.Host
			}

			hosts = hostsWithDatastore(hosts, c.req.Config.Files.VmPathName)
			host := hosts[rand.Intn(len(hosts))]
			vm.Runtime.Host = &host
		})
	} else {
		vm.Runtime.Host = c.req.Host
	}

	vm.Guest = &types.GuestInfo{
		ToolsStatus:        types.VirtualMachineToolsStatusToolsNotInstalled,
		ToolsVersion:       "0",
		ToolsRunningStatus: string(types.VirtualMachineToolsRunningStatusGuestToolsNotRunning),
	}

	vm.Summary.Guest = &types.VirtualMachineGuestSummary{
		ToolsStatus: vm.Guest.ToolsStatus,
	}
	vm.Summary.Config.VmPathName = vm.Config.Files.VmPathName
	vm.Summary.Runtime.Host = vm.Runtime.Host

	err = vm.create(c.ctx, &c.req.Config, c.register)
	if err != nil {
		folderRemoveChild(c.ctx, &c.Folder.Folder, vm)
		return nil, err
	}

	host := Map.Get(*vm.Runtime.Host).(*HostSystem)
	Map.AppendReference(c.ctx, host, &host.Vm, vm.Self)
	vm.EnvironmentBrowser = *hostParent(&host.HostSystem).EnvironmentBrowser

	for i := range vm.Datastore {
		ds := Map.Get(vm.Datastore[i]).(*Datastore)
		Map.AppendReference(c.ctx, ds, &ds.Vm, vm.Self)
	}

	pool := Map.Get(*vm.ResourcePool)
	// This can be an internal call from VirtualApp.CreateChildVMTask, where pool is already locked.
	c.ctx.WithLock(pool, func() {
		if rp, ok := asResourcePoolMO(pool); ok {
			rp.Vm = append(rp.Vm, vm.Self)
		}
		if vapp, ok := pool.(*VirtualApp); ok {
			vapp.Vm = append(vapp.Vm, vm.Self)
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

	vm.RefreshStorageInfo(c.ctx, nil)

	Map.Update(vm, []types.PropertyChange{
		{Name: "name", Val: c.req.Config.Name},
	})

	return vm.Reference(), nil
}

func (f *Folder) CreateVMTask(ctx *Context, c *types.CreateVM_Task) soap.HasFault {
	return &methods.CreateVM_TaskBody{
		Res: &types.CreateVM_TaskResponse{
			Returnval: NewTask(&createVM{f, ctx, c, false}).Run(ctx),
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

	create.RunBlocking(c.ctx)

	if create.Info.Error != nil {
		return nil, create.Info.Error.Fault
	}

	return create.Info.Result, nil
}

func (f *Folder) RegisterVMTask(ctx *Context, c *types.RegisterVM_Task) soap.HasFault {
	return &methods.RegisterVM_TaskBody{
		Res: &types.RegisterVM_TaskResponse{
			Returnval: NewTask(&registerVM{f, ctx, c}).Run(ctx),
		},
	}
}

func (f *Folder) MoveIntoFolderTask(ctx *Context, c *types.MoveIntoFolder_Task) soap.HasFault {
	task := CreateTask(f, "moveIntoFolder", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		for _, ref := range c.List {
			obj := Map.Get(ref).(mo.Entity)

			parent, ok := Map.Get(*(obj.Entity()).Parent).(*Folder)

			if !ok || !folderHasChildType(&f.Folder, ref.Type) {
				return nil, &types.NotSupported{}
			}

			folderRemoveChild(ctx, &parent.Folder, ref)
			folderPutChild(ctx, &f.Folder, obj)
		}

		return nil, nil
	})

	return &methods.MoveIntoFolder_TaskBody{
		Res: &types.MoveIntoFolder_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (f *Folder) CreateDVSTask(ctx *Context, req *types.CreateDVS_Task) soap.HasFault {
	task := CreateTask(f, "createDVS", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		spec := req.Spec.ConfigSpec.GetDVSConfigSpec()
		dvs := &DistributedVirtualSwitch{}
		dvs.Name = spec.Name
		dvs.Entity().Name = dvs.Name

		if Map.FindByName(dvs.Name, f.ChildEntity) != nil {
			return nil, &types.InvalidArgument{InvalidProperty: "name"}
		}

		dvs.Uuid = newUUID(dvs.Name)

		folderPutChild(ctx, &f.Folder, dvs)

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

		dvs.AddDVPortgroupTask(ctx, &types.AddDVPortgroup_Task{
			Spec: []types.DVPortgroupConfigSpec{{
				Name: dvs.Name + "-DVUplinks" + strings.TrimPrefix(dvs.Self.Value, "dvs"),
				Type: string(types.DistributedVirtualPortgroupPortgroupTypeEarlyBinding),
				DefaultPortConfig: &types.VMwareDVSPortSetting{
					Vlan: &types.VmwareDistributedVirtualSwitchTrunkVlanSpec{
						VlanId: []types.NumericRange{{Start: 0, End: 4094}},
					},
					UplinkTeamingPolicy: &types.VmwareUplinkPortTeamingPolicy{
						Policy: &types.StringPolicy{
							Value: "loadbalance_srcid",
						},
						ReversePolicy: &types.BoolPolicy{
							Value: types.NewBool(true),
						},
						NotifySwitches: &types.BoolPolicy{
							Value: types.NewBool(true),
						},
						RollingOrder: &types.BoolPolicy{
							Value: types.NewBool(true),
						},
					},
				},
			}},
		})

		return dvs.Reference(), nil
	})

	return &methods.CreateDVS_TaskBody{
		Res: &types.CreateDVS_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (f *Folder) RenameTask(ctx *Context, r *types.Rename_Task) soap.HasFault {
	return RenameTask(ctx, f, r)
}

func (f *Folder) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
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
			ctx.WithLock(obj, func() {
				id := obj.DestroyTask(&types.Destroy_Task{
					This: c,
				}).(*methods.Destroy_TaskBody).Res.Returnval

				t := Map.Get(id).(*Task)
				t.Wait()
				if t.Info.Error != nil {
					fault = t.Info.Error.Fault // For example, can't destroy a powered on VM
				}
			})
			if fault != nil {
				return nil, fault
			}
		}

		// Remove the folder itself
		folderRemoveChild(ctx, &Map.Get(*f.Parent).(*Folder).Folder, f.Self)
		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}
