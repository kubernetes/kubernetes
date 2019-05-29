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
	"time"

	"github.com/google/uuid"
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

	id := uuid.New().String()

	hardware := *host.Summary.Hardware
	hs.Summary.Hardware = &hardware
	hs.Summary.Hardware.Uuid = id

	info := *esx.HostHardwareInfo
	info.SystemInfo.Uuid = id
	hs.Hardware = &info

	config := []struct {
		ref **types.ManagedObjectReference
		obj mo.Reference
	}{
		{&hs.ConfigManager.DatastoreSystem, &HostDatastoreSystem{Host: &hs.HostSystem}},
		{&hs.ConfigManager.NetworkSystem, NewHostNetworkSystem(&hs.HostSystem)},
		{&hs.ConfigManager.AdvancedOption, NewOptionManager(nil, esx.Setting)},
		{&hs.ConfigManager.FirewallSystem, NewHostFirewallSystem(&hs.HostSystem)},
	}

	for _, c := range config {
		ref := Map.Put(c.obj).Reference()

		*c.ref = &ref
	}

	return hs
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
func CreateDefaultESX(f *Folder) {
	dc := NewDatacenter(f)

	host := NewHostSystem(esx.HostSystem)

	summary := new(types.ComputeResourceSummary)
	addComputeResource(summary, host)

	cr := &mo.ComputeResource{Summary: summary}
	cr.EnvironmentBrowser = newEnvironmentBrowser()
	cr.Self = *host.Parent
	cr.Name = host.Name
	cr.Host = append(cr.Host, host.Reference())
	Map.PutEntity(cr, host)

	pool := NewResourcePool()
	cr.ResourcePool = &pool.Self
	Map.PutEntity(cr, pool)
	pool.Owner = cr.Self

	Map.Get(dc.HostFolder).(*Folder).putChild(cr)
}

// CreateStandaloneHost uses esx.HostSystem as a template, applying the given spec
// and creating the ComputeResource parent and ResourcePool sibling.
func CreateStandaloneHost(f *Folder, spec types.HostConnectSpec) (*HostSystem, types.BaseMethodFault) {
	if spec.HostName == "" {
		return nil, &types.NoHost{}
	}

	pool := NewResourcePool()
	host := NewHostSystem(esx.HostSystem)

	host.Summary.Config.Name = spec.HostName
	host.Name = host.Summary.Config.Name
	host.Runtime.ConnectionState = types.HostSystemConnectionStateDisconnected

	summary := new(types.ComputeResourceSummary)
	addComputeResource(summary, host)

	cr := &mo.ComputeResource{
		ConfigurationEx: &types.ComputeResourceConfigInfo{
			VmSwapPlacement: string(types.VirtualMachineConfigInfoSwapPlacementTypeVmDirectory),
		},
		Summary:            summary,
		EnvironmentBrowser: newEnvironmentBrowser(),
	}

	Map.PutEntity(cr, Map.NewEntity(host))
	host.Summary.Host = &host.Self

	Map.PutEntity(cr, Map.NewEntity(pool))

	cr.Name = host.Name
	cr.Host = append(cr.Host, host.Reference())
	cr.ResourcePool = &pool.Self

	f.putChild(cr)
	pool.Owner = cr.Self

	return host, nil
}

func (h *HostSystem) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(h, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		if len(h.Vm) > 0 {
			return nil, &types.ResourceInUse{}
		}

		ctx.postEvent(&types.HostRemovedEvent{HostEvent: h.event()})

		f := Map.getEntityParent(h, "Folder").(*Folder)
		f.removeChild(h.Reference())

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (h *HostSystem) EnterMaintenanceModeTask(spec *types.EnterMaintenanceMode_Task) soap.HasFault {
	task := CreateTask(h, "enterMaintenanceMode", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.InMaintenanceMode = true
		return nil, nil
	})

	return &methods.EnterMaintenanceMode_TaskBody{
		Res: &types.EnterMaintenanceMode_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (h *HostSystem) ExitMaintenanceModeTask(spec *types.ExitMaintenanceMode_Task) soap.HasFault {
	task := CreateTask(h, "exitMaintenanceMode", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		h.Runtime.InMaintenanceMode = false
		return nil, nil
	})

	return &methods.ExitMaintenanceMode_TaskBody{
		Res: &types.ExitMaintenanceMode_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
