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
	"net"
	"os"
	"time"

	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

var (
	hostPortUnique = os.Getenv("VCSIM_HOST_PORT_UNIQUE") == "true"
)

type HostSystem struct {
	mo.HostSystem
}

func asHostSystemMO(obj mo.Reference) (*mo.HostSystem, bool) {
	h, ok := getManagedObject(obj).Addr().Interface().(*mo.HostSystem)
	return h, ok
}

func NewHostSystem(host mo.HostSystem) *HostSystem {
	if hostPortUnique { // configure unique port for each host
		port := &esx.HostSystem.Summary.Config.Port
		*port++
		host.Summary.Config.Port = *port
	}

	now := time.Now()

	hs := &HostSystem{
		HostSystem: host,
	}

	hs.Name = hs.Summary.Config.Name
	hs.Summary.Runtime = &hs.Runtime
	hs.Summary.Runtime.BootTime = &now

	// shallow copy Summary.Hardware, as each host will be assigned its own .Uuid
	hardware := *host.Summary.Hardware
	hs.Summary.Hardware = &hardware

	if hs.Hardware == nil {
		// shallow copy Hardware, as each host will be assigned its own .Uuid
		info := *esx.HostHardwareInfo
		hs.Hardware = &info
	}

	cfg := new(types.HostConfigInfo)
	deepCopy(hs.Config, cfg)
	hs.Config = cfg

	config := []struct {
		ref **types.ManagedObjectReference
		obj mo.Reference
	}{
		{&hs.ConfigManager.DatastoreSystem, &HostDatastoreSystem{Host: &hs.HostSystem}},
		{&hs.ConfigManager.NetworkSystem, NewHostNetworkSystem(&hs.HostSystem)},
		{&hs.ConfigManager.AdvancedOption, NewOptionManager(nil, esx.Setting)},
		{&hs.ConfigManager.FirewallSystem, NewHostFirewallSystem(&hs.HostSystem)},
		{&hs.ConfigManager.StorageSystem, NewHostStorageSystem(&hs.HostSystem)},
	}

	for _, c := range config {
		ref := Map.Put(c.obj).Reference()

		*c.ref = &ref
	}

	return hs
}

func (h *HostSystem) configure(spec types.HostConnectSpec, connected bool) {
	h.Runtime.ConnectionState = types.HostSystemConnectionStateDisconnected
	if connected {
		h.Runtime.ConnectionState = types.HostSystemConnectionStateConnected
	}
	if net.ParseIP(spec.HostName) != nil {
		h.Config.Network.Vnic[0].Spec.Ip.IpAddress = spec.HostName
	}

	h.Summary.Config.Name = spec.HostName
	h.Name = h.Summary.Config.Name
	id := newUUID(h.Name)
	h.Summary.Hardware.Uuid = id
	h.Hardware.SystemInfo.Uuid = id
}

func (h *HostSystem) event() types.HostEvent {
	return types.HostEvent{
		Event: types.Event{
			Datacenter:      datacenterEventArgument(h),
			ComputeResource: h.eventArgumentParent(),
			Host:            h.eventArgument(),
		},
	}
}

func (h *HostSystem) eventArgument() *types.HostEventArgument {
	return &types.HostEventArgument{
		Host:                h.Self,
		EntityEventArgument: types.EntityEventArgument{Name: h.Name},
	}
}

func (h *HostSystem) eventArgumentParent() *types.ComputeResourceEventArgument {
	parent := hostParent(&h.HostSystem)

	return &types.ComputeResourceEventArgument{
		ComputeResource:     parent.Self,
		EntityEventArgument: types.EntityEventArgument{Name: parent.Name},
	}
}

func hostParent(host *mo.HostSystem) *mo.ComputeResource {
	switch parent := Map.Get(*host.Parent).(type) {
	case *mo.ComputeResource:
		return parent
	case *ClusterComputeResource:
		return &parent.ComputeResource
	default:
		return nil
	}
}

func addComputeResource(s *types.ComputeResourceSummary, h *HostSystem) {
	s.TotalCpu += h.Summary.Hardware.CpuMhz
	s.TotalMemory += h.Summary.Hardware.MemorySize
	s.NumCpuCores += h.Summary.Hardware.NumCpuCores
	s.NumCpuThreads += h.Summary.Hardware.NumCpuThreads
	s.EffectiveCpu += h.Summary.Hardware.CpuMhz
	s.EffectiveMemory += h.Summary.Hardware.MemorySize
	s.NumHosts++
	s.NumEffectiveHosts++
	s.OverallStatus = types.ManagedEntityStatusGreen
}

// CreateDefaultESX creates a standalone ESX
// Adds objects of type: Datacenter, Network, ComputeResource, ResourcePool and HostSystem
func CreateDefaultESX(ctx *Context, f *Folder) {
	dc := NewDatacenter(ctx, &f.Folder)

	host := NewHostSystem(esx.HostSystem)

	summary := new(types.ComputeResourceSummary)
	addComputeResource(summary, host)

	cr := &mo.ComputeResource{
		Summary: summary,
		Network: esx.Datacenter.Network,
	}
	cr.EnvironmentBrowser = newEnvironmentBrowser()
	cr.Self = *host.Parent
	cr.Name = host.Name
	cr.Host = append(cr.Host, host.Reference())
	host.Network = cr.Network
	ctx.Map.PutEntity(cr, host)

	pool := NewResourcePool()
	cr.ResourcePool = &pool.Self
	ctx.Map.PutEntity(cr, pool)
	pool.Owner = cr.Self

	folderPutChild(ctx, &ctx.Map.Get(dc.HostFolder).(*Folder).Folder, cr)
}

// CreateStandaloneHost uses esx.HostSystem as a template, applying the given spec
// and creating the ComputeResource parent and ResourcePool sibling.
func CreateStandaloneHost(ctx *Context, f *Folder, spec types.HostConnectSpec) (*HostSystem, types.BaseMethodFault) {
	if spec.HostName == "" {
		return nil, &types.NoHost{}
	}

	template := esx.HostSystem
	network := ctx.Map.getEntityDatacenter(f).defaultNetwork()

	if p := ctx.Map.FindByName(spec.UserName, f.ChildEntity); p != nil {
		cr := p.(*mo.ComputeResource)
		h := ctx.Map.Get(cr.Host[0])
		// "clone" an existing host from the inventory
		template = h.(*HostSystem).HostSystem
		template.Vm = nil
		network = cr.Network
	}

	pool := NewResourcePool()
	host := NewHostSystem(template)
	host.configure(spec, false)

	summary := new(types.ComputeResourceSummary)
	addComputeResource(summary, host)

	cr := &mo.ComputeResource{
		ConfigurationEx: &types.ComputeResourceConfigInfo{
			VmSwapPlacement: string(types.VirtualMachineConfigInfoSwapPlacementTypeVmDirectory),
		},
		Summary:            summary,
		EnvironmentBrowser: newEnvironmentBrowser(),
	}

	ctx.Map.PutEntity(cr, ctx.Map.NewEntity(host))
	host.Summary.Host = &host.Self

	ctx.Map.PutEntity(cr, ctx.Map.NewEntity(pool))

	cr.Name = host.Name
	cr.Network = network
	cr.Host = append(cr.Host, host.Reference())
	cr.ResourcePool = &pool.Self

	folderPutChild(ctx, &f.Folder, cr)
	pool.Owner = cr.Self
	host.Network = cr.Network

	return host, nil
}

func (h *HostSystem) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(h, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if len(h.Vm) > 0 {
			return nil, &types.ResourceInUse{}
		}

		ctx.postEvent(&types.HostRemovedEvent{HostEvent: h.event()})

		f := ctx.Map.getEntityParent(h, "Folder").(*Folder)
		folderRemoveChild(ctx, &f.Folder, h.Reference())

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (h *HostSystem) EnterMaintenanceModeTask(ctx *Context, spec *types.EnterMaintenanceMode_Task) soap.HasFault {
	task := CreateTask(h, "enterMaintenanceMode", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.InMaintenanceMode = true
		return nil, nil
	})

	return &methods.EnterMaintenanceMode_TaskBody{
		Res: &types.EnterMaintenanceMode_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (h *HostSystem) ExitMaintenanceModeTask(ctx *Context, spec *types.ExitMaintenanceMode_Task) soap.HasFault {
	task := CreateTask(h, "exitMaintenanceMode", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.InMaintenanceMode = false
		return nil, nil
	})

	return &methods.ExitMaintenanceMode_TaskBody{
		Res: &types.ExitMaintenanceMode_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (h *HostSystem) DisconnectHostTask(ctx *Context, spec *types.DisconnectHost_Task) soap.HasFault {
	task := CreateTask(h, "disconnectHost", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.ConnectionState = types.HostSystemConnectionStateDisconnected
		return nil, nil
	})

	return &methods.DisconnectHost_TaskBody{
		Res: &types.DisconnectHost_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (h *HostSystem) ReconnectHostTask(ctx *Context, spec *types.ReconnectHost_Task) soap.HasFault {
	task := CreateTask(h, "reconnectHost", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.ConnectionState = types.HostSystemConnectionStateConnected
		return nil, nil
	})

	return &methods.ReconnectHost_TaskBody{
		Res: &types.ReconnectHost_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}
