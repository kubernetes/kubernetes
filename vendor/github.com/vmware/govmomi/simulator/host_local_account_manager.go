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
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// As of vSphere API 5.1, local groups operations are deprecated, so it's not supported here.

type HostLocalAccountManager struct {
	mo.HostLocalAccountManager
}

func NewHostLocalAccountManager(ref types.ManagedObjectReference) object.Reference {
	m := &HostLocalAccountManager{}
	m.Self = ref

	return m
}

func (h *HostLocalAccountManager) CreateUser(req *types.CreateUser) soap.HasFault {
	spec := req.User.GetHostAccountSpec()
	userDirectory := Map.UserDirectory()

	found := userDirectory.search(true, false, compareFunc(spec.Id, true))
	if len(found) > 0 {
		return &methods.CreateUserBody{
			Fault_: Fault("", &types.AlreadyExists{}),
		}
	}

	userDirectory.addUser(spec.Id)

	return &methods.CreateUserBody{
		Res: &types.CreateUserResponse{},
	}
}

func (h *HostLocalAccountManager) RemoveUser(req *types.RemoveUser) soap.HasFault {
	userDirectory := Map.UserDirectory()

	found := userDirectory.search(true, false, compareFunc(req.UserName, true))

	if len(found) == 0 {
		return &methods.RemoveUserBody{
			Fault_: Fault("", &types.UserNotFound{}),
		}
	}

	userDirectory.removeUser(req.UserName)

	return &methods.RemoveUserBody{
		Res: &types.RemoveUserResponse{},
	}
}

func (h *HostLocalAccountManager) UpdateUser(req *types.UpdateUser) soap.HasFault {
	return &methods.CreateUserBody{
		Res: &types.CreateUserResponse{},
	}
}
