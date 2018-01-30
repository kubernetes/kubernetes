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

type DistributedVirtualPortgroup struct {
	mo.DistributedVirtualPortgroup
}

func (s *DistributedVirtualPortgroup) ReconfigureDVPortgroupTask(req *types.ReconfigureDVPortgroup_Task) soap.HasFault {
	task := CreateTask(s, "reconfigureDvPortgroup", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		s.Config.DefaultPortConfig = req.Spec.DefaultPortConfig
		s.Config.NumPorts = req.Spec.NumPorts
		s.Config.AutoExpand = req.Spec.AutoExpand
		s.Config.Type = req.Spec.Type
		s.Config.Description = req.Spec.Description
		s.Config.DynamicData = req.Spec.DynamicData
		s.Config.Name = req.Spec.Name
		s.Config.Policy = req.Spec.Policy
		s.Config.PortNameFormat = req.Spec.PortNameFormat
		s.Config.VmVnicNetworkResourcePoolKey = req.Spec.VmVnicNetworkResourcePoolKey

		return nil, nil
	})

	return &methods.ReconfigureDVPortgroup_TaskBody{
		Res: &types.ReconfigureDVPortgroup_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (s *DistributedVirtualPortgroup) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(s, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		vswitch := Map.Get(*s.Config.DistributedVirtualSwitch).(*DistributedVirtualSwitch)
		Map.RemoveReference(vswitch, &vswitch.Portgroup, s.Reference())
		Map.removeString(vswitch, &vswitch.Summary.PortgroupName, s.Name)

		f := Map.getEntityParent(vswitch, "Folder").(*Folder)
		f.removeChild(s.Reference())

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}

}
