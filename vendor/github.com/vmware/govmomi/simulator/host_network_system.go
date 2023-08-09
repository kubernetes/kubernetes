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
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HostNetworkSystem struct {
	mo.HostNetworkSystem

	Host *mo.HostSystem
}

func NewHostNetworkSystem(host *mo.HostSystem) *HostNetworkSystem {
	return &HostNetworkSystem{
		Host: host,
		HostNetworkSystem: mo.HostNetworkSystem{
			NetworkInfo: &types.HostNetworkInfo{
				Vswitch: []types.HostVirtualSwitch{
					{
						Name:      "vSwitch0",
						Portgroup: []string{"VM Network"},
					},
				},
			},
		},
	}
}

func (s *HostNetworkSystem) folder() *Folder {
	f := Map.getEntityDatacenter(s.Host).NetworkFolder
	return Map.Get(f).(*Folder)
}

func (s *HostNetworkSystem) AddVirtualSwitch(c *types.AddVirtualSwitch) soap.HasFault {
	r := &methods.AddVirtualSwitchBody{}

	for _, vswitch := range s.NetworkInfo.Vswitch {
		if vswitch.Name == c.VswitchName {
			r.Fault_ = Fault("", &types.AlreadyExists{Name: c.VswitchName})
			return r
		}
	}

	s.NetworkInfo.Vswitch = append(s.NetworkInfo.Vswitch, types.HostVirtualSwitch{
		Name: c.VswitchName,
	})

	r.Res = &types.AddVirtualSwitchResponse{}

	return r
}

func (s *HostNetworkSystem) RemoveVirtualSwitch(c *types.RemoveVirtualSwitch) soap.HasFault {
	r := &methods.RemoveVirtualSwitchBody{}

	vs := s.NetworkInfo.Vswitch

	for i, v := range vs {
		if v.Name == c.VswitchName {
			s.NetworkInfo.Vswitch = append(vs[:i], vs[i+1:]...)
			r.Res = &types.RemoveVirtualSwitchResponse{}
			return r
		}
	}

	r.Fault_ = Fault("", &types.NotFound{})

	return r
}

func (s *HostNetworkSystem) AddPortGroup(c *types.AddPortGroup) soap.HasFault {
	var vswitch *types.HostVirtualSwitch

	r := &methods.AddPortGroupBody{}

	if c.Portgrp.Name == "" {
		r.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "name"})
		return r
	}

	for i := range s.NetworkInfo.Vswitch {
		if s.NetworkInfo.Vswitch[i].Name == c.Portgrp.VswitchName {
			vswitch = &s.NetworkInfo.Vswitch[i]
			break
		}
	}

	if vswitch == nil {
		r.Fault_ = Fault("", &types.NotFound{})
		return r
	}

	network := &mo.Network{}
	network.Name = c.Portgrp.Name
	network.Entity().Name = network.Name

	folder := s.folder()

	if obj := Map.FindByName(c.Portgrp.Name, folder.ChildEntity); obj != nil {
		r.Fault_ = Fault("", &types.DuplicateName{
			Name:   c.Portgrp.Name,
			Object: obj.Reference(),
		})

		return r
	}

	folder.putChild(network)

	vswitch.Portgroup = append(vswitch.Portgroup, c.Portgrp.Name)
	r.Res = &types.AddPortGroupResponse{}

	return r
}

func (s *HostNetworkSystem) RemovePortGroup(c *types.RemovePortGroup) soap.HasFault {
	var vswitch *types.HostVirtualSwitch

	r := &methods.RemovePortGroupBody{}

	for i, v := range s.NetworkInfo.Vswitch {
		for j, pg := range v.Portgroup {
			if pg == c.PgName {
				vswitch = &s.NetworkInfo.Vswitch[i]
				vswitch.Portgroup = append(vswitch.Portgroup[:j], vswitch.Portgroup[j+1:]...)
			}
		}
	}

	if vswitch == nil {
		r.Fault_ = Fault("", &types.NotFound{})
		return r
	}

	folder := s.folder()
	e := Map.FindByName(c.PgName, folder.ChildEntity)
	folder.removeChild(e.Reference())

	r.Res = &types.RemovePortGroupResponse{}

	return r
}

func (s *HostNetworkSystem) UpdateNetworkConfig(req *types.UpdateNetworkConfig) soap.HasFault {
	s.NetworkConfig = &req.Config

	return &methods.UpdateNetworkConfigBody{
		Res: &types.UpdateNetworkConfigResponse{
			Returnval: types.HostNetworkConfigResult{},
		},
	}
}
