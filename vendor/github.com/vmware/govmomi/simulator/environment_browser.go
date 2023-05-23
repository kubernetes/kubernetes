/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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

	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type EnvironmentBrowser struct {
	mo.EnvironmentBrowser

	types.QueryConfigOptionResponse
}

func newEnvironmentBrowser() *types.ManagedObjectReference {
	env := new(EnvironmentBrowser)
	Map.Put(env)
	return &env.Self
}

func (b *EnvironmentBrowser) hosts(ctx *Context) []types.ManagedObjectReference {
	ctx.Map.m.Lock()
	defer ctx.Map.m.Unlock()
	for _, obj := range ctx.Map.objects {
		switch e := obj.(type) {
		case *mo.ComputeResource:
			if b.Self == *e.EnvironmentBrowser {
				return e.Host
			}
		case *ClusterComputeResource:
			if b.Self == *e.EnvironmentBrowser {
				return e.Host
			}
		}
	}
	return nil
}

func (b *EnvironmentBrowser) QueryConfigOption(req *types.QueryConfigOption) soap.HasFault {
	body := new(methods.QueryConfigOptionBody)

	opt := b.QueryConfigOptionResponse.Returnval
	if opt == nil {
		opt = &types.VirtualMachineConfigOption{
			Version:       esx.HardwareVersion,
			DefaultDevice: esx.VirtualDevice,
		}
	}

	body.Res = &types.QueryConfigOptionResponse{
		Returnval: opt,
	}

	return body
}

func guestFamily(id string) string {
	// TODO: We could capture the entire GuestOsDescriptor list from EnvironmentBrowser,
	// but it is a ton of data.. this should be good enough for now.
	switch {
	case strings.HasPrefix(id, "win"):
		return string(types.VirtualMachineGuestOsFamilyWindowsGuest)
	case strings.HasPrefix(id, "darwin"):
		return string(types.VirtualMachineGuestOsFamilyDarwinGuestFamily)
	default:
		return string(types.VirtualMachineGuestOsFamilyLinuxGuest)
	}
}

func (b *EnvironmentBrowser) QueryConfigOptionEx(req *types.QueryConfigOptionEx) soap.HasFault {
	body := new(methods.QueryConfigOptionExBody)

	opt := b.QueryConfigOptionResponse.Returnval
	if opt == nil {
		opt = &types.VirtualMachineConfigOption{
			Version:       esx.HardwareVersion,
			DefaultDevice: esx.VirtualDevice,
		}
	}

	if req.Spec != nil {
		// From the SDK QueryConfigOptionEx doc:
		// "If guestId is nonempty, the guestOSDescriptor array of the config option is filtered to match against the guest IDs in the spec.
		//  If there is no match, the whole list is returned."
		for _, id := range req.Spec.GuestId {
			for _, gid := range GuestID {
				if string(gid) == id {
					opt.GuestOSDescriptor = []types.GuestOsDescriptor{{
						Id:     id,
						Family: guestFamily(id),
					}}

					break
				}
			}
		}
	}

	if len(opt.GuestOSDescriptor) == 0 {
		for i := range GuestID {
			id := string(GuestID[i])
			opt.GuestOSDescriptor = append(opt.GuestOSDescriptor, types.GuestOsDescriptor{
				Id:     id,
				Family: guestFamily(id),
			})
		}
	}

	body.Res = &types.QueryConfigOptionExResponse{
		Returnval: opt,
	}

	return body
}

func (b *EnvironmentBrowser) QueryConfigOptionDescriptor(ctx *Context, req *types.QueryConfigOptionDescriptor) soap.HasFault {
	body := &methods.QueryConfigOptionDescriptorBody{
		Res: new(types.QueryConfigOptionDescriptorResponse),
	}

	body.Res.Returnval = []types.VirtualMachineConfigOptionDescriptor{{
		Key:                 esx.HardwareVersion,
		Description:         esx.HardwareVersion,
		Host:                b.hosts(ctx),
		CreateSupported:     types.NewBool(true),
		DefaultConfigOption: types.NewBool(false),
		RunSupported:        types.NewBool(true),
		UpgradeSupported:    types.NewBool(true),
	}}

	return body
}

func (b *EnvironmentBrowser) QueryConfigTarget(ctx *Context, req *types.QueryConfigTarget) soap.HasFault {
	body := &methods.QueryConfigTargetBody{
		Res: &types.QueryConfigTargetResponse{
			Returnval: &types.ConfigTarget{
				SmcPresent: types.NewBool(false),
			},
		},
	}
	target := body.Res.Returnval

	var hosts []types.ManagedObjectReference
	if req.Host == nil {
		hosts = b.hosts(ctx)
	} else {
		hosts = append(hosts, *req.Host)
	}

	seen := make(map[types.ManagedObjectReference]bool)

	for i := range hosts {
		host := ctx.Map.Get(hosts[i]).(*HostSystem)
		target.NumCpus += int32(host.Summary.Hardware.NumCpuPkgs)
		target.NumCpuCores += int32(host.Summary.Hardware.NumCpuCores)
		target.NumNumaNodes++

		for _, ref := range host.Datastore {
			if seen[ref] {
				continue
			}
			seen[ref] = true

			ds := ctx.Map.Get(ref).(*Datastore)
			target.Datastore = append(target.Datastore, types.VirtualMachineDatastoreInfo{
				VirtualMachineTargetInfo: types.VirtualMachineTargetInfo{
					Name: ds.Name,
				},
				Datastore:       ds.Summary,
				Capability:      ds.Capability,
				Mode:            string(types.HostMountModeReadWrite),
				VStorageSupport: string(types.FileSystemMountInfoVStorageSupportStatusVStorageUnsupported),
			})
		}

		for _, ref := range host.Network {
			if seen[ref] {
				continue
			}
			seen[ref] = true

			switch n := ctx.Map.Get(ref).(type) {
			case *mo.Network:
				target.Network = append(target.Network, types.VirtualMachineNetworkInfo{
					VirtualMachineTargetInfo: types.VirtualMachineTargetInfo{
						Name: n.Name,
					},
					Network: n.Summary.GetNetworkSummary(),
				})
			case *DistributedVirtualPortgroup:
				dvs := ctx.Map.Get(*n.Config.DistributedVirtualSwitch).(*DistributedVirtualSwitch)
				target.DistributedVirtualPortgroup = append(target.DistributedVirtualPortgroup, types.DistributedVirtualPortgroupInfo{
					SwitchName:                  dvs.Name,
					SwitchUuid:                  dvs.Uuid,
					PortgroupName:               n.Name,
					PortgroupKey:                n.Key,
					PortgroupType:               n.Config.Type,
					UplinkPortgroup:             false,
					Portgroup:                   n.Self,
					NetworkReservationSupported: types.NewBool(false),
				})
			case *DistributedVirtualSwitch:
				target.DistributedVirtualSwitch = append(target.DistributedVirtualSwitch, types.DistributedVirtualSwitchInfo{
					SwitchName:                  n.Name,
					SwitchUuid:                  n.Uuid,
					DistributedVirtualSwitch:    n.Self,
					NetworkReservationSupported: types.NewBool(false),
				})
			}
		}
	}

	return body
}
