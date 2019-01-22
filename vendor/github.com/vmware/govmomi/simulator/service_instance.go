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
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ServiceInstance struct {
	mo.ServiceInstance
}

func NewServiceInstance(content types.ServiceContent, folder mo.Folder) *ServiceInstance {
	Map = NewRegistry()

	s := &ServiceInstance{}

	s.Self = vim25.ServiceInstance
	s.Content = content

	Map.Put(s)

	f := &Folder{Folder: folder}
	Map.Put(f)

	var setting []types.BaseOptionValue

	if content.About.ApiType == "HostAgent" {
		CreateDefaultESX(f)
	} else {
		content.About.InstanceUuid = uuid.New().String()
		setting = vpx.Setting
	}

	objects := []object.Reference{
		NewSessionManager(*s.Content.SessionManager),
		NewAuthorizationManager(*s.Content.AuthorizationManager),
		NewPerformanceManager(*s.Content.PerfManager),
		NewPropertyCollector(s.Content.PropertyCollector),
		NewFileManager(*s.Content.FileManager),
		NewVirtualDiskManager(*s.Content.VirtualDiskManager),
		NewLicenseManager(*s.Content.LicenseManager),
		NewSearchIndex(*s.Content.SearchIndex),
		NewViewManager(*s.Content.ViewManager),
		NewEventManager(*s.Content.EventManager),
		NewTaskManager(*s.Content.TaskManager),
		NewUserDirectory(*s.Content.UserDirectory),
		NewOptionManager(s.Content.Setting, setting),
	}

	if s.Content.CustomFieldsManager != nil {
		objects = append(objects, NewCustomFieldsManager(*s.Content.CustomFieldsManager))
	}

	if s.Content.IpPoolManager != nil {
		objects = append(objects, NewIpPoolManager(*s.Content.IpPoolManager))
	}

	if s.Content.AccountManager != nil {
		objects = append(objects, NewHostLocalAccountManager(*s.Content.AccountManager))
	}

	for _, o := range objects {
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
