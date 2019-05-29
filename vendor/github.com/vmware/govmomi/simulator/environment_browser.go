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
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type EnvironmentBrowser struct {
	mo.EnvironmentBrowser
}

func newEnvironmentBrowser() *types.ManagedObjectReference {
	env := new(EnvironmentBrowser)
	Map.Put(env)
	return &env.Self
}

func (b *EnvironmentBrowser) QueryConfigOption(req *types.QueryConfigOption) soap.HasFault {
	body := new(methods.QueryConfigOptionBody)

	opt := &types.VirtualMachineConfigOption{
		Version:       esx.HardwareVersion,
		DefaultDevice: esx.VirtualDevice,
	}

	body.Res = &types.QueryConfigOptionResponse{
		Returnval: opt,
	}

	return body
}

func (b *EnvironmentBrowser) QueryConfigOptionEx(req *types.QueryConfigOptionEx) soap.HasFault {
	body := new(methods.QueryConfigOptionExBody)

	opt := &types.VirtualMachineConfigOption{
		Version:       esx.HardwareVersion,
		DefaultDevice: esx.VirtualDevice,
	}

	body.Res = &types.QueryConfigOptionExResponse{
		Returnval: opt,
	}

	return body
}
