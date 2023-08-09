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
	"time"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/simulator/internal"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ServiceInstance struct {
	mo.ServiceInstance
}

func NewServiceInstance(ctx *Context, content types.ServiceContent, folder mo.Folder) *ServiceInstance {
	// TODO: This function ignores the passed in Map and operates on the
	// global Map.
	Map = NewRegistry()
	ctx.Map = Map

	s := &ServiceInstance{}

	s.Self = vim25.ServiceInstance
	s.Content = content

	Map.Put(s)

	f := &Folder{Folder: folder}
	Map.Put(f)

	if content.About.ApiType == "HostAgent" {
		CreateDefaultESX(ctx, f)
	} else {
		content.About.InstanceUuid = uuid.New().String()
	}

	refs := mo.References(content)

	for i := range refs {
		if Map.Get(refs[i]) != nil {
			continue
		}
		content := types.ObjectContent{Obj: refs[i]}
		o, err := loadObject(content)
		if err != nil {
			panic(err)
		}
		Map.Put(o)
	}

	return s
}

func (s *ServiceInstance) RetrieveServiceContent(*types.RetrieveServiceContent) soap.HasFault {
	return &methods.RetrieveServiceContentBody{
		Res: &types.RetrieveServiceContentResponse{
			Returnval: s.Content,
		},
	}
}

func (*ServiceInstance) CurrentTime(*types.CurrentTime) soap.HasFault {
	return &methods.CurrentTimeBody{
		Res: &types.CurrentTimeResponse{
			Returnval: time.Now(),
		},
	}
}

func (s *ServiceInstance) RetrieveInternalContent(*internal.RetrieveInternalContent) soap.HasFault {
	return &internal.RetrieveInternalContentBody{
		Res: &internal.RetrieveInternalContentResponse{
			Returnval: internal.InternalServiceInstanceContent{
				NfcService: types.ManagedObjectReference{Type: "NfcService", Value: "NfcService"},
			},
		},
	}
}
